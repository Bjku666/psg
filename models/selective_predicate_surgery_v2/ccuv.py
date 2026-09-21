"""Leakage-safe utilities for candidate-conditioned verifier controls."""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Mapping, Sequence

import numpy as np

from models.relation_decision_regret_v1.legal_oracle import _predicate_options
from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators,
    exact_action_utility,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import _native_pred


UTILITY_SCALE = 100_000.0
ROW_DIM = 384 + 56 + 5


def stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.sha1(str(value).encode()).digest()
    return int.from_bytes(digest[:4], "little") % int(modulo)


def split_records(records: Sequence[Mapping]) -> dict[str, list[dict]]:
    """Deterministic 80/10/10 physical-file split."""
    output = {"train": [], "select": [], "calib": []}
    for record in records:
        group = str(record.get("bootstrap_group", record.get("file_name")))
        bucket = stable_bucket(group, 10)
        key = "select" if bucket == 0 else "calib" if bucket == 1 else "train"
        output[key].append(record)
    return output


def truth_predicates(relations: Sequence[Sequence[int]], row: Mapping) -> set[int]:
    mapped = row.get("gt_pair")
    if mapped is None:
        return set()
    pair = tuple(map(int, mapped))
    return {
        int(predicate)
        for subject, obj, predicate in relations
        if (int(subject), int(obj)) == pair
    }


def build_teacher(records: Sequence[Mapping], num_predicates: int = 56,
                  depth: int = 3) -> list[dict]:
    """Build exact signed action teachers for one evaluation population."""
    denominators = PopulationDenominators.from_records(records, num_predicates)
    output: list[dict] = []
    for image_index, image in enumerate(records):
        relations = image.get("relations", ())
        for row in image.get("selected", ()):
            native = int(_native_pred(row))
            candidates = [
                int(predicate) for predicate in _predicate_options(row, depth)
                if int(predicate) != native
            ]
            if len(candidates) != int(depth) - 1:
                raise ValueError(
                    f"row {row.get('row')} has {len(candidates)} alternatives"
                )
            action = [
                exact_action_utility(
                    relations, row, predicate, denominators, num_predicates
                )
                for predicate in candidates
            ]
            utilities = np.asarray([item["delta_mr"] for item in action], dtype=np.float64)
            delta_r = np.asarray([item["delta_r"] for item in action], dtype=np.float64)
            truth = truth_predicates(relations, row)
            native_correct = native in truth
            transitions = [
                ("correct" if native_correct else "wrong")
                + "_to_"
                + ("correct" if predicate in truth else "wrong")
                for predicate in candidates
            ]
            output.append({
                "image_index": int(image_index),
                "row_id": int(row["row"]),
                "bootstrap_group": str(
                    image.get("bootstrap_group", image.get("file_name", image_index))
                ),
                "native": native,
                "candidates": np.asarray(candidates, dtype=np.int16),
                "utilities": utilities,
                "delta_r": delta_r,
                "transitions": transitions,
                "edit_label": bool(np.max(utilities) > 0.0),
                "signed_best_utility": float(np.max(utilities)),
                "best_candidate": int(np.argmax(utilities)),
                "pair_features": np.asarray(row["pair_features"], dtype=np.float32).reshape(-1),
                "pred_scores": np.asarray(row["pred_scores"], dtype=np.float32)[1:],
                "pair_score": float(row.get("score", 0.0)),
            })
    return output


def row_matrix(rows: Sequence[Mapping]) -> np.ndarray:
    """Observable 445-D row representation; no teacher fields are read."""
    output = np.empty((len(rows), ROW_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        probabilities = np.clip(
            np.asarray(row["pred_scores"], dtype=np.float32).reshape(-1),
            1e-6,
            1.0 - 1e-6,
        )
        if probabilities.size != 56:
            raise ValueError(f"expected 56 predicate scores, got {probabilities.size}")
        logits = np.log(probabilities / (1.0 - probabilities)).astype(np.float32)
        order = np.sort(logits)[::-1]
        entropy = float(
            -(probabilities * np.log(probabilities)
              + (1.0 - probabilities) * np.log(1.0 - probabilities)).mean()
        )
        scalar = np.asarray([
            float(row.get("pair_score", 0.0)),
            float(order[0] - order[1]),
            float(order[0] - order[2]),
            entropy,
            float(logits[int(row["native"])]),
        ], dtype=np.float32)
        output[index] = np.concatenate([
            np.asarray(row["pair_features"], dtype=np.float32), logits, scalar
        ])
    return output


def chooser_matrix(x_row: np.ndarray, rows: Sequence[Mapping], slots: np.ndarray) -> np.ndarray:
    """p1.8-compatible candidate features for one candidate slot per row."""
    slots = np.asarray(slots, dtype=np.int64)
    candidates = np.asarray([
        int(row["candidates"][slot]) for row, slot in zip(rows, slots)
    ], dtype=np.int64)
    output = np.zeros((len(rows), ROW_DIM + 56 + 1), dtype=np.float32)
    output[:, :ROW_DIM] = x_row
    output[np.arange(len(rows)), ROW_DIM + candidates] = 1.0
    logits = x_row[:, 384:440]
    natives = np.asarray([int(row["native"]) for row in rows], dtype=np.int64)
    output[:, -1] = logits[np.arange(len(rows)), candidates] - logits[np.arange(len(rows)), natives]
    return output


def verifier_matrix(x_row: np.ndarray, rows: Sequence[Mapping], slots: np.ndarray) -> np.ndarray:
    """Candidate-conditioned features with explicit native and candidate IDs."""
    slots = np.asarray(slots, dtype=np.int64)
    candidates = np.asarray([
        int(row["candidates"][slot]) for row, slot in zip(rows, slots)
    ], dtype=np.int64)
    natives = np.asarray([int(row["native"]) for row in rows], dtype=np.int64)
    output = np.zeros((len(rows), ROW_DIM + 2 * 56 + 1), dtype=np.float32)
    output[:, :ROW_DIM] = x_row
    index = np.arange(len(rows))
    output[index, ROW_DIM + natives] = 1.0
    output[index, ROW_DIM + 56 + candidates] = 1.0
    logits = x_row[:, 384:440]
    output[:, -1] = logits[index, candidates] - logits[index, natives]
    return output


def proposal_targets(rows: Sequence[Mapping], slots: np.ndarray) -> dict[str, np.ndarray]:
    slots = np.asarray(slots, dtype=np.int64)
    utilities = np.asarray([
        float(row["utilities"][slot]) for row, slot in zip(rows, slots)
    ], dtype=np.float64)
    delta_r = np.asarray([
        float(row["delta_r"][slot]) for row, slot in zip(rows, slots)
    ], dtype=np.float64)
    transitions = np.asarray([
        str(row["transitions"][slot]) for row, slot in zip(rows, slots)
    ], dtype=object)
    return {"utility": utilities, "delta_r": delta_r, "transition": transitions}


def exact_top_mask(scores: np.ndarray, coverage_percent: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    count = max(1, int(np.ceil(len(scores) * float(coverage_percent) / 100.0)))
    # Stable sort makes tied scores reproducible without silently exceeding q.
    chosen = np.argsort(-scores, kind="stable")[:count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    return mask


def utility_curve(scores: np.ndarray, targets: Mapping[str, np.ndarray],
                  rows: Sequence[Mapping], coverages: Sequence[float]) -> list[dict]:
    utilities = np.asarray(targets["utility"], dtype=np.float64)
    delta_r = np.asarray(targets["delta_r"], dtype=np.float64)
    transitions = np.asarray(targets["transition"], dtype=object)
    oracle_total = float(sum(max(0.0, float(row["signed_best_utility"])) for row in rows))
    curve = []
    for coverage in coverages:
        active = exact_top_mask(scores, coverage)
        selected = utilities[active]
        transition_counts = Counter(map(str, transitions[active]))
        positive = float(selected[selected > 0.0].sum())
        negative = float(selected[selected < 0.0].sum())
        curve.append({
            "q_percent": float(coverage),
            "applied_rows": int(active.sum()),
            "applied_edit_rate_percent": float(100.0 * active.mean()),
            "threshold": float(np.min(np.asarray(scores)[active])),
            "rescue_precision": float(np.mean(selected > 0.0)),
            "harm_rate": float(np.mean(selected < 0.0)),
            "sum_positive_utility": positive,
            "sum_negative_utility": negative,
            "net_utility": float(selected.sum()),
            "delta_mR_pp": float(100.0 * selected.sum()),
            "delta_R_pp": float(100.0 * delta_r[active].sum()),
            "oracle_utility_capture": float(positive / oracle_total) if oracle_total > 0 else 0.0,
            "transitions": dict(transition_counts),
        })
    return curve


def select_curve_point(curve: Sequence[Mapping]) -> dict:
    eligible = [point for point in curve if float(point["delta_R_pp"]) >= -0.5]
    pool = eligible if eligible else list(curve)
    return dict(max(pool, key=lambda point: (
        float(point["delta_mR_pp"]), -float(point["q_percent"])
    )))
