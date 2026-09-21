"""S0--S7 killer controls for fixed-support predicate surgery.

Every fitted parameter and heuristic threshold is selected on fit records.
KEEP is always the fixed zero-utility anchor at decode time.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators,
    population_substitution_teacher,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import _native_pred


UTILITY_SCALE = 100_000.0


def predicate_frequencies(records: Sequence[Mapping], num_predicates: int) -> np.ndarray:
    counts = np.zeros(int(num_predicates), dtype=np.int64)
    for record in records:
        for _, _, predicate in record.get("relations", ()):
            predicate = int(predicate)
            if 0 <= predicate < int(num_predicates):
                counts[predicate] += 1
    return counts


def _logits(row: Mapping) -> np.ndarray:
    probabilities = np.asarray(row["pred_scores"], dtype=np.float32)[1:]
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def _row_statistics(row: Mapping) -> tuple[float, float]:
    values = _logits(row)
    order = np.argsort(-values, kind="stable")
    margin = float(values[order[0]] - values[order[1]]) if len(order) > 1 else 0.0
    shifted = values - float(values.max())
    probabilities = np.exp(shifted)
    probabilities /= max(float(probabilities.sum()), 1e-12)
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    return margin, entropy


def action_features(
    row: Mapping,
    predicate: int,
    frequencies: np.ndarray,
    include_hidden: bool,
) -> np.ndarray:
    """Full logits plus explicit candidate/native identities and optional hidden."""
    logits = _logits(row)
    native = _native_pred(row)
    predicate = int(predicate)
    identity = np.zeros(2 * len(logits), dtype=np.float32)
    identity[native] = 1.0
    identity[len(logits) + predicate] = 1.0
    margin, entropy = _row_statistics(row)
    extras = np.asarray(
        [
            float(row.get("score", 0.0)),
            float(logits[predicate] - logits[native]),
            margin,
            entropy,
            float(np.log1p(frequencies[predicate])),
            float(np.log1p(frequencies[native])),
        ],
        dtype=np.float32,
    )
    parts = [logits, identity, extras]
    if include_hidden:
        parts.insert(0, np.asarray(row.get("pair_features", np.zeros(384)), dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def build_groups(
    records: Sequence[Mapping],
    num_predicates: int,
    depth: int,
    frequencies: np.ndarray,
) -> list[dict]:
    denominators = PopulationDenominators.from_records(records, num_predicates)
    groups = []
    for image_index, record in enumerate(records):
        labels = population_substitution_teacher(
            record.get("relations", ()), record.get("selected", ()), denominators,
            num_predicates, depth,
        )
        by_row = defaultdict(list)
        for action in labels:
            if action["action"] == "SWAP":
                by_row[int(action["row"])].append(action)
        for row in record.get("selected", ()):
            actions = sorted(
                by_row.get(int(row["row"]), ()),
                key=lambda action: int(action["candidate_rank"]),
            )
            if not actions:
                continue
            margin, entropy = _row_statistics(row)
            groups.append(
                {
                    "image_index": int(image_index),
                    "row": row,
                    "native": int(_native_pred(row)),
                    "margin": margin,
                    "entropy": entropy,
                    "actions": actions,
                    "frequencies": frequencies,
                }
            )
    return groups


def _utility(decisions: Sequence[int | None], groups: Sequence[Mapping]) -> float:
    total = 0.0
    for choice, group in zip(decisions, groups):
        if choice is not None:
            total += float(group["actions"][int(choice)]["delta_mr"])
    return total


def _quantile_grid(values: Sequence[float], points: int = 81) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return np.asarray([0.0])
    return np.unique(np.quantile(array, np.linspace(0.0, 1.0, int(points))))


@dataclass
class ThresholdControl:
    mode: str
    threshold: float | dict[int, float]

    @classmethod
    def fit(cls, groups: Sequence[Mapping], mode: str) -> "ThresholdControl":
        mode = str(mode).upper()
        if mode not in {"S0", "S1", "S2"}:
            raise ValueError(mode)

        def best_threshold(subset: Sequence[Mapping], statistic: str, high: bool) -> float:
            values = [float(group[statistic]) for group in subset]
            candidates = _quantile_grid(values)
            no_action = (
                np.nextafter(float(candidates.max()), np.inf)
                if high else np.nextafter(float(candidates.min()), -np.inf)
            )
            candidates = np.unique(np.append(candidates, no_action))
            best_key = (-np.inf, -np.inf)
            best_value = float(no_action)
            for threshold in candidates:
                decisions = [
                    0 if ((float(group[statistic]) >= threshold) if high else
                          (float(group[statistic]) <= threshold)) else None
                    for group in subset
                ]
                key = (_utility(decisions, subset), -float(threshold) if high else float(threshold))
                if key > best_key:
                    best_key = key
                    best_value = float(threshold)
            return best_value

        if mode == "S0":
            return cls(mode, best_threshold(groups, "margin", high=False))
        if mode == "S1":
            return cls(mode, best_threshold(groups, "entropy", high=True))

        fallback = best_threshold(groups, "margin", high=False)
        by_native = defaultdict(list)
        for group in groups:
            by_native[int(group["native"])].append(group)
        thresholds = {
            predicate: best_threshold(rows, "margin", high=False)
            if len(rows) >= 20 else fallback
            for predicate, rows in by_native.items()
        }
        thresholds[-1] = fallback
        return cls(mode, thresholds)

    def choose(self, group: Mapping) -> int | None:
        if self.mode == "S0":
            return 0 if float(group["margin"]) <= float(self.threshold) else None
        if self.mode == "S1":
            return 0 if float(group["entropy"]) >= float(self.threshold) else None
        assert isinstance(self.threshold, dict)
        threshold = self.threshold.get(int(group["native"]), self.threshold[-1])
        return 0 if float(group["margin"]) <= float(threshold) else None


@dataclass
class PriorAdjustedControl:
    alpha: float

    @classmethod
    def fit(cls, groups: Sequence[Mapping]) -> "PriorAdjustedControl":
        best_key = (0.0, 0.0)
        best_alpha = 0.0
        for alpha in np.linspace(0.0, 3.0, 61):
            model = cls(float(alpha))
            decisions = [model.choose(group) for group in groups]
            key = (_utility(decisions, groups), -float(alpha))
            if key > best_key:
                best_key = key
                best_alpha = float(alpha)
        return cls(best_alpha)

    def choose(self, group: Mapping) -> int | None:
        row = group["row"]
        logits = _logits(row)
        frequencies = np.asarray(group["frequencies"])
        native = int(group["native"])
        native_score = logits[native] - self.alpha * np.log1p(frequencies[native])
        scored = []
        for index, action in enumerate(group["actions"]):
            predicate = int(action["predicate"])
            value = logits[predicate] - self.alpha * np.log1p(frequencies[predicate])
            scored.append((float(value), -predicate, index))
        value, _, index = max(scored)
        return int(index) if value > float(native_score) else None


@dataclass
class LearnedActionControl:
    mode: str
    model: object
    include_hidden: bool
    threshold: float

    @classmethod
    def fit(cls, groups: Sequence[Mapping], mode: str, seed: int = 0) -> "LearnedActionControl":
        mode = str(mode).upper()
        if mode not in {"S4", "S5", "S6", "S7"}:
            raise ValueError(mode)
        include_hidden = mode in {"S5", "S6", "S7"}
        features, targets = [], []
        for group in groups:
            for action in group["actions"]:
                features.append(
                    action_features(
                        group["row"], int(action["predicate"]),
                        group["frequencies"], include_hidden,
                    )
                )
                targets.append(float(action["delta_mr"]) * UTILITY_SCALE)
        x = np.stack(features).astype(np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if mode in {"S4", "S5"}:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        elif mode == "S6":
            model = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=int(seed),
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=1e-3,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=80,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=8,
                    random_state=int(seed),
                ),
            )
        model.fit(x, y)
        provisional = cls(mode, model, include_hidden, 0.0)
        best_scores, utilities = [], []
        for group in groups:
            scores = provisional.scores(group)
            best = int(np.argmax(scores))
            best_scores.append(float(scores[best]))
            utilities.append(float(group["actions"][best]["delta_mr"]))
        candidates = _quantile_grid(best_scores)
        no_action = np.nextafter(float(candidates.max()), np.inf)
        candidates = np.unique(np.append(candidates, no_action))
        best_key = (-np.inf, -np.inf)
        best_threshold = no_action
        for threshold in candidates:
            utility = sum(
                value for score, value in zip(best_scores, utilities)
                if score > float(threshold)
            )
            key = (float(utility), float(threshold))
            if key > best_key:
                best_key = key
                best_threshold = float(threshold)
        provisional.threshold = best_threshold
        return provisional

    def scores(self, group: Mapping) -> np.ndarray:
        """Score all legal swap candidates for one physical row."""
        x = np.stack(
            [
                action_features(
                    group["row"], int(action["predicate"]),
                    group["frequencies"], self.include_hidden,
                )
                for action in group["actions"]
            ]
        )
        return np.asarray(self.model.predict(x), dtype=float)

    def choose(self, group: Mapping) -> int | None:
        scores = self.scores(group)
        best = int(np.argmax(scores))
        return best if float(scores[best]) > float(self.threshold) else None


def fit_controls(
    groups: Sequence[Mapping], seed: int = 0, learned_group_cap: int | None = None
) -> dict[str, object]:
    learned_groups = groups
    if learned_group_cap is not None and len(groups) > int(learned_group_cap):
        # Deterministic evenly spaced sampling keeps the carrier/image order
        # fixed while bounding full-population CPU controls.
        indices = np.linspace(0, len(groups) - 1, int(learned_group_cap), dtype=np.int64)
        learned_groups = [groups[int(index)] for index in indices]
    controls: dict[str, object] = {
        "S0": ThresholdControl.fit(groups, "S0"),
        "S1": ThresholdControl.fit(groups, "S1"),
        "S2": ThresholdControl.fit(groups, "S2"),
        "S3": PriorAdjustedControl.fit(groups),
    }
    for mode in ("S4", "S5", "S6", "S7"):
        controls[mode] = LearnedActionControl.fit(learned_groups, mode, seed)
    return controls
