"""MEAR: candidate-relative ranking of population-exact edit utility."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.relation_decision_regret_v1.legal_oracle import _predicate_options
from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators,
    exact_action_utility,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import _native_pred


UTILITY_SCALE = 100_000.0


def _logits(row: Mapping) -> np.ndarray:
    probabilities = np.asarray(row["pred_scores"], dtype=np.float32)[1:]
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


@dataclass
class PackedActions:
    hidden: np.ndarray
    logits: np.ndarray
    pair_score: np.ndarray
    native: np.ndarray
    candidates: np.ndarray
    utilities: np.ndarray
    image_index: np.ndarray
    row_id: np.ndarray
    frequencies: np.ndarray
    depth: int

    def __len__(self) -> int:
        return int(len(self.native))


def pack_records(
    records: Sequence[Mapping],
    num_predicates: int,
    depth: int,
    frequencies: np.ndarray,
) -> PackedActions:
    """Pack one row per fixed-support action group without duplicating hidden tokens."""
    width = int(depth) - 1
    if width < 1:
        raise ValueError("MEAR needs at least one SWAP candidate")
    count = sum(len(record.get("selected", ())) for record in records)
    hidden = np.empty((count, 384), dtype=np.float16)
    logits = np.empty((count, int(num_predicates)), dtype=np.float16)
    pair_score = np.empty(count, dtype=np.float32)
    native = np.empty(count, dtype=np.int16)
    candidates = np.empty((count, width), dtype=np.int16)
    utilities = np.empty((count, width), dtype=np.float32)
    image_index = np.empty(count, dtype=np.int32)
    row_id = np.empty(count, dtype=np.int32)
    denominators = PopulationDenominators.from_records(records, num_predicates)
    offset = 0
    for current_image, record in enumerate(records):
        for row in record.get("selected", ()):
            native_predicate = _native_pred(row)
            options = [
                int(predicate)
                for predicate in _predicate_options(row, depth)
                if int(predicate) != native_predicate
            ]
            if len(options) != width:
                raise ValueError(
                    f"expected {width} alternatives, got {len(options)} for row {row.get('row')}"
                )
            token = np.asarray(row.get("pair_features", ()), dtype=np.float32).reshape(-1)
            if token.size != 384:
                raise ValueError("MEAR requires the full 384-D pair token")
            hidden[offset] = token.astype(np.float16)
            logits[offset] = _logits(row).astype(np.float16)
            pair_score[offset] = float(row.get("score", 0.0))
            native[offset] = int(native_predicate)
            candidates[offset] = np.asarray(options, dtype=np.int16)
            utilities[offset] = np.asarray(
                [
                    exact_action_utility(
                        record.get("relations", ()), row, predicate, denominators,
                        num_predicates,
                    )["delta_mr"]
                    for predicate in options
                ],
                dtype=np.float32,
            )
            image_index[offset] = int(current_image)
            row_id[offset] = int(row["row"])
            offset += 1
    return PackedActions(
        hidden=hidden,
        logits=logits,
        pair_score=pair_score,
        native=native,
        candidates=candidates,
        utilities=utilities,
        image_index=image_index,
        row_id=row_id,
        frequencies=np.asarray(frequencies, dtype=np.float32),
        depth=int(depth),
    )


class MetricExactActionRanker(nn.Module):
    """Candidate-relative low-rank scorer with an external zero KEEP anchor."""

    def __init__(self, num_predicates: int = 56, hidden_dim: int = 384,
                 predicate_dim: int = 32, interaction_dim: int = 64) -> None:
        super().__init__()
        self.num_predicates = int(num_predicates)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, interaction_dim)
        self.predicate_embedding = nn.Embedding(num_predicates, predicate_dim)
        self.predicate_projection = nn.Linear(predicate_dim, interaction_dim, bias=False)
        self.logit_context = nn.Sequential(
            nn.LayerNorm(num_predicates),
            nn.Linear(num_predicates, 32),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(interaction_dim + 32 + 6, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        pair_score: torch.Tensor,
        native: torch.Tensor,
        candidates: torch.Tensor,
        log_frequencies: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        batch, width = candidates.shape
        projected_hidden = self.hidden_projection(self.hidden_norm(hidden))
        native_embedding = self.predicate_embedding(native)
        candidate_embedding = self.predicate_embedding(candidates)
        relative = candidate_embedding - native_embedding[:, None, :]
        interaction = projected_hidden[:, None, :] * self.predicate_projection(relative)
        context = self.logit_context(logits)[:, None, :].expand(-1, width, -1)
        native_logit = logits.gather(1, native[:, None])
        candidate_logit = logits.gather(1, candidates)
        delta_logit = candidate_logit - native_logit
        top2 = torch.topk(logits, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1])[:, None].expand(-1, width)
        ranks = torch.arange(1, width + 1, device=logits.device, dtype=logits.dtype)
        ranks = (ranks / float(depth))[None, :].expand(batch, -1)
        candidate_frequency = log_frequencies[candidates]
        native_frequency = log_frequencies[native][:, None].expand(-1, width)
        pair = pair_score[:, None].expand(-1, width)
        scalars = torch.stack(
            [delta_logit, candidate_logit, margin, ranks,
             candidate_frequency - native_frequency, pair],
            dim=-1,
        )
        features = torch.cat([interaction, context, scalars], dim=-1)
        return self.scorer(features).squeeze(-1)


def _batch_tensors(data: PackedActions, indices: np.ndarray, device: torch.device) -> dict:
    return {
        # torch.from_numpy is ABI-incompatible with the system's NumPy build;
        # tensor() is an explicit safe copy for these bounded action batches.
        "hidden": torch.tensor(data.hidden[indices].astype(np.float32), device=device),
        "logits": torch.tensor(data.logits[indices].astype(np.float32), device=device),
        "pair_score": torch.tensor(data.pair_score[indices], device=device),
        "native": torch.tensor(data.native[indices].astype(np.int64), device=device),
        "candidates": torch.tensor(data.candidates[indices].astype(np.int64), device=device),
        "utilities": torch.tensor(data.utilities[indices], device=device),
    }


def _ranking_loss(scores: torch.Tensor, utilities: torch.Tensor) -> torch.Tensor:
    keep = torch.zeros((scores.shape[0], 1), device=scores.device, dtype=scores.dtype)
    values = torch.cat([keep, scores], dim=1)
    targets = torch.cat([keep, utilities * UTILITY_SCALE], dim=1)
    target_difference = targets[:, :, None] - targets[:, None, :]
    score_difference = values[:, :, None] - values[:, None, :]
    mask = target_difference > 1e-8
    if not torch.any(mask):
        return scores.sum() * 0.0
    weights = target_difference[mask]
    weights = weights / weights.mean().clamp_min(1e-6)
    return (F.softplus(-score_difference[mask]) * weights).mean()


def train_ranker(
    data: PackedActions,
    num_predicates: int,
    device: str = "cuda",
    epochs: int = 5,
    batch_size: int = 4096,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> tuple[MetricExactActionRanker, list[dict]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    target_device = torch.device(device)
    model = MetricExactActionRanker(num_predicates=num_predicates).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-3)
    positives = int(np.count_nonzero(data.utilities > 0.0))
    total = int(data.utilities.size)
    positive_weight = min(50.0, (total - positives) / max(positives, 1))
    pos_weight = torch.tensor(positive_weight, device=target_device)
    log_frequencies = torch.tensor(np.log1p(data.frequencies), device=target_device).float()
    log_frequencies /= log_frequencies.max().clamp_min(1.0)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(int(epochs)):
        model.train()
        permutation = rng.permutation(len(data))
        total_loss = total_rank = total_sign = 0.0
        batches = 0
        for start in range(0, len(data), int(batch_size)):
            indices = permutation[start:start + int(batch_size)]
            batch = _batch_tensors(data, indices, target_device)
            scores = model(
                batch["hidden"], batch["logits"], batch["pair_score"],
                batch["native"], batch["candidates"], log_frequencies, data.depth,
            )
            rank_loss = _ranking_loss(scores, batch["utilities"])
            signs = (batch["utilities"] > 0.0).float()
            sign_loss = F.binary_cross_entropy_with_logits(
                scores, signs, pos_weight=pos_weight
            )
            loss = rank_loss + 0.2 * sign_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_rank += float(rank_loss.detach())
            total_sign += float(sign_loss.detach())
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(batches, 1),
                "rank_loss": total_rank / max(batches, 1),
                "sign_loss": total_sign / max(batches, 1),
            }
        )
    return model, history


@torch.inference_mode()
def predict_scores(
    model: MetricExactActionRanker,
    data: PackedActions,
    device: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    target_device = torch.device(device)
    model.eval()
    log_frequencies = torch.tensor(np.log1p(data.frequencies), device=target_device).float()
    log_frequencies /= log_frequencies.max().clamp_min(1.0)
    values = []
    for start in range(0, len(data), int(batch_size)):
        indices = np.arange(start, min(start + int(batch_size), len(data)))
        batch = _batch_tensors(data, indices, target_device)
        scores = model(
            batch["hidden"], batch["logits"], batch["pair_score"],
            batch["native"], batch["candidates"], log_frequencies, data.depth,
        )
        values.append(scores.cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.empty_like(data.utilities)
