#!/usr/bin/env python3
"""Fit and evaluate frozen-carrier predicate arbitration controls.

The runner is intentionally downstream of the p1.4 compact carrier.  It never
adds a pair, reads ``gt_pair`` at decode time, or changes the pair-slot order.
Only the predicate emitted for each already-supported physical pair is changed.

``ctpa`` is a comparative tournament: for the same pair, every candidate
predicate is compared against the other candidates with an antisymmetric
linear score.  ``hidden_mlp`` is the capacity-matched killer control and
predicts candidate correctness directly from the frozen hidden representation.
Both controls are fit on the fit carrier and evaluated on a separate carrier.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population


def _top_predicates(row: Mapping, depth: int) -> list[int]:
    """Return zero-based relation IDs, excluding NONE column zero."""
    scores = np.asarray(row["pred_scores"], dtype=np.float32)
    order = np.argsort(-scores[1:], kind="stable")[: int(depth)]
    return [int(value) for value in order]


def _target(row: Mapping, relations: Sequence[Sequence[int]]) -> int | None:
    """Return the GT predicate for this mapped pair, if one exists.

    The method is used only for fit labels.  Multiple annotations for one
    directed pair are resolved by the frozen carrier logit so the training
    target remains single-label and deterministic.
    """
    pair = row.get("gt_pair")
    if pair is None:
        return None
    pair = tuple(map(int, pair))
    values = [int(r) for s, o, r in relations if (int(s), int(o)) == pair]
    if not values:
        return None
    scores = np.asarray(row["pred_scores"], dtype=np.float32)
    return max(values, key=lambda value: (float(scores[value + 1]), -value))


@dataclass
class Tournament:
    """Antisymmetric pairwise logistic model."""

    model: LogisticRegression
    predicate_embeddings: np.ndarray
    hidden_projection: np.ndarray
    depth: int
    lambda_logit: float = 0.10

    def _features(self, row: Mapping, first: int, second: int) -> np.ndarray:
        scores = np.asarray(row["pred_scores"], dtype=np.float32)[1:]
        hidden = np.asarray(row.get("pair_features", np.zeros(384)), dtype=np.float32).reshape(-1)
        h = hidden @ self.hidden_projection
        e = self.predicate_embeddings[first] - self.predicate_embeddings[second]
        # Every term is antisymmetric under first <-> second.  With zero
        # intercept this guarantees d(a,b) == -d(b,a), including inference.
        interaction = np.outer(h, e).reshape(-1)
        return np.concatenate((e, interaction,
                               np.asarray([scores[first] - scores[second]], dtype=np.float32)))

    def margin(self, row: Mapping, first: int, second: int) -> float:
        return float(self.model.decision_function(self._features(row, first, second)[None])[0])

    def choose(self, row: Mapping) -> int:
        options = _top_predicates(row, self.depth)
        if not options:
            return 0
        scores = np.asarray(row["pred_scores"], dtype=np.float32)[1:]
        values = []
        for first in options:
            tournament = sum(self.margin(row, first, other) for other in options if other != first)
            values.append((tournament + self.lambda_logit * float(scores[first]), first))
        return int(max(values, key=lambda value: (value[0], -value[1]))[1])


def _fit_tournament(images: Sequence[Mapping], num_predicates: int, depth: int,
                    seed: int, max_pairs: int,
                    regret_aware: bool = False) -> tuple[Tournament, dict]:
    rng = np.random.default_rng(seed)
    # Small fixed random embeddings make the feature map reproducible while
    # allowing pair-feature/predicate interactions without a 384x56 tensor.
    embeddings = rng.normal(0.0, 1.0 / np.sqrt(16), size=(num_predicates, 16)).astype(np.float32)
    projection = rng.normal(0.0, 1.0 / np.sqrt(384), size=(384, 16)).astype(np.float32)
    prototype = Tournament(LogisticRegression(fit_intercept=False, class_weight="balanced",
                                                C=1.0, max_iter=200, random_state=seed),
                           embeddings, projection, depth)
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    predicate_counts = np.ones(num_predicates, dtype=np.float64)
    for image in images:
        for relation in image.get("relations", ()):
            predicate = int(relation[2])
            if 0 <= predicate < num_predicates:
                predicate_counts[predicate] += 1.0
    positive = 0
    for image in images:
        relations = image.get("relations", ())
        for row in image.get("candidates", ()):
            target = _target(row, relations)
            options = _top_predicates(row, depth)
            if target is None or target not in options:
                continue
            for other in options:
                if other == target:
                    continue
                # Add both orientations; this makes the anti-symmetry contract
                # explicit in the training data as well as in the architecture.
                weight = float(1.0 / np.sqrt(predicate_counts[target])) if regret_aware else 1.0
                features.append(prototype._features(row, target, other)); labels.append(1); weights.append(weight)
                features.append(prototype._features(row, other, target)); labels.append(0); weights.append(weight)
                positive += 1
                if len(labels) >= int(max_pairs) * 2:
                    break
            if len(labels) >= int(max_pairs) * 2:
                break
        if len(labels) >= int(max_pairs) * 2:
            break
    if not features or len(set(labels)) < 2:
        raise ValueError("fit carrier did not expose both tournament labels")
    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int8)
    prototype.model.fit(x, y, sample_weight=np.asarray(weights, dtype=np.float64))
    return prototype, {"training_pairs": int(positive), "training_examples": int(len(y)),
                       "positive_examples": int(y.sum()), "feature_dim": int(x.shape[1]),
                       "regret_aware": bool(regret_aware)}


def _candidate_features(row: Mapping, predicate: int) -> np.ndarray:
    scores = np.asarray(row["pred_scores"], dtype=np.float32)[1:]
    hidden = np.asarray(row.get("pair_features", np.zeros(384)), dtype=np.float32).reshape(-1)
    onehot = np.zeros(len(scores), dtype=np.float32)
    onehot[int(predicate)] = 1.0
    return np.concatenate((hidden, scores, onehot,
                           np.asarray([float(row["score"]), float(scores[predicate])], dtype=np.float32)))


def _fit_hidden_mlp(images: Sequence[Mapping], num_predicates: int, depth: int,
                    seed: int, max_examples: int):
    values = []
    rng = np.random.default_rng(seed)
    for image in images:
        relations = image.get("relations", ())
        for row in image.get("candidates", ()):
            target = _target(row, relations)
            options = _top_predicates(row, depth)
            for predicate in options:
                values.append((_candidate_features(row, predicate), int(target is not None and predicate == target)))
    if len(values) > int(max_examples):
        values = [values[int(i)] for i in rng.choice(len(values), int(max_examples), replace=False)]
    x = np.stack([value[0] for value in values]).astype(np.float32)
    y = np.asarray([value[1] for value in values], dtype=np.int8)
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           max_leaf_nodes=31, l2_regularization=1e-2,
                                           random_state=seed)
    model.fit(x, y)
    return model, {"training_examples": int(len(y)), "positive_examples": int(y.sum()),
                   "feature_dim": int(x.shape[1])}


def _decode(images: Sequence[Mapping], mode: str, model, depth: int, budget: int) -> list[dict]:
    records = []
    for image in images:
        candidates = []
        hidden_options: list[tuple[dict, list[int]]] = []
        hidden_features: list[np.ndarray] = []
        for source in image.get("candidates", ()):
            row = dict(source)
            if mode == "ctpa":
                row["pred"] = int(model.choose(row))
            elif mode == "hidden_mlp":
                options = _top_predicates(row, depth)
                hidden_options.append((row, options))
                hidden_features.extend(_candidate_features(row, p) for p in options)
            else:
                row["pred"] = int(_top_predicates(row, 1)[0]) if _top_predicates(row, 1) else 0
            if mode != "hidden_mlp":
                row["pred_score"] = float(np.asarray(row["pred_scores"])[int(row["pred"]) + 1])
                candidates.append(row)
        if mode == "hidden_mlp":
            offset = 0
            if hidden_features:
                all_scores = model.predict_proba(np.stack(hidden_features).astype(np.float32))[:, 1]
            else:
                all_scores = np.zeros(0, dtype=np.float32)
            for row, options in hidden_options:
                count = len(options)
                if options:
                    scores = all_scores[offset: offset + count]
                    row["pred"] = int(options[int(np.argmax(scores))])
                    offset += count
                else:
                    row["pred"] = 0
                row["pred_score"] = float(np.asarray(row["pred_scores"])[int(row["pred"]) + 1])
                candidates.append(row)
        # Keep exactly the baseline physical-pair ranking.  This is the key
        # isolation: CTPA cannot win by changing pair admission or pair order.
        candidates.sort(key=lambda row: (-float(row["score"]), int(row["row"])))
        selected = candidates[: int(budget)]
        records.append({"image_id": str(image["image_id"]), "file_name": str(image["file_name"]),
                        "bootstrap_group": str(image.get("bootstrap_group", image["file_name"])),
                        "relations": image.get("relations", ()), "selected": selected,
                        "budget": int(budget)})
    return records


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if value.get("schema_version") != 1 or "images" not in value:
        raise ValueError(f"unsupported compact carrier: {path}")
    return value


def run(fit: Mapping, dev: Mapping, depth: int, budget: int, seed: int,
        max_pairs: int, max_examples: int, regret_aware: bool = False,
        include_hidden: bool = True) -> dict:
    num_predicates = len(fit["predicate_classes"])
    tournament, ctpa_stats = _fit_tournament(fit["images"], num_predicates, depth, seed, max_pairs,
                                              regret_aware=regret_aware)
    hidden, hidden_stats = (None, {"skipped": True})
    if include_hidden:
        hidden, hidden_stats = _fit_hidden_mlp(fit["images"], num_predicates, depth, seed, max_examples)
    output = {"schema_version": 1, "contract": {
        "fit_trained_dev_evaluated": True,
        "support": "p1.4 compact frozen physical pairs",
        "pair_support_changed": False,
        "gt_pair_used_at_decode": False,
        "single_mpo": True,
        "confirm_locked": True,
    }, "fit_images": len(fit["images"]), "dev_images": len(dev["images"]),
              "depth": int(depth), "budget": int(budget), "seed": int(seed),
              "objective": "regret_aware_pairwise" if regret_aware else "uniform_pairwise",
              "models": {"ctpa": ctpa_stats, "hidden_mlp": hidden_stats}, "results": {}}
    modes = [("baseline", None), ("ctpa", tournament)]
    if include_hidden:
        modes.append(("hidden_mlp", hidden))
    for mode, model in modes:
        records = _decode(dev["images"], mode, model, depth, budget)
        metrics = evaluate_population(records, num_predicates)
        output["results"][mode] = {
            "images": int(len(records)), "mR": float(metrics[f"mR@{budget}"]),
            "R": float(metrics[f"R@{budget}"]),
            "positive_predicate_auc": None,
        }
    # An audit of the antisymmetry contract is cheap and catches accidental
    # intercepts or feature-order regressions before any full run.
    checks = []
    for image in fit["images"][: min(20, len(fit["images"]))]:
        for row in image.get("candidates", ())[:5]:
            opts = _top_predicates(row, depth)
            if len(opts) > 1:
                checks.append(abs(tournament.margin(row, opts[0], opts[1]) + tournament.margin(row, opts[1], opts[0])))
    output["antisymmetry_max_abs"] = float(max(checks) if checks else 0.0)
    output["gates"] = {
        "ctpa_minus_baseline_mR_pp": 100.0 * (output["results"]["ctpa"]["mR"] - output["results"]["baseline"]["mR"]),
        "hidden_mlp_minus_baseline_mR_pp": (None if not include_hidden else 100.0 * (output["results"]["hidden_mlp"]["mR"] - output["results"]["baseline"]["mR"])),
        "ctpa_beats_hidden_mlp": (None if not include_hidden else output["results"]["ctpa"]["mR"] > output["results"]["hidden_mlp"]["mR"]),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-carrier", required=True, type=Path)
    parser.add_argument("--dev-carrier", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--max-examples", type=int, default=250000)
    parser.add_argument("--regret-aware", action="store_true",
                        help="weight pairwise comparisons by inverse fit predicate frequency")
    parser.add_argument("--skip-hidden", action="store_true",
                        help="skip the expensive Hidden-MLP killer control")
    args = parser.parse_args()
    result = run(_load(args.fit_carrier), _load(args.dev_carrier), args.depth, args.budget,
                 args.seed, args.max_pairs, args.max_examples, args.regret_aware,
                 include_hidden=not args.skip_hidden)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
