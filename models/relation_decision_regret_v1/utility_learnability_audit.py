#!/usr/bin/env python3
"""Audit whether exact utility labels are recoverable from frozen carrier scores.

This is a cheap, fit-only diagnostic before introducing a CMUD learner.  It
never evaluates official test images and never changes the candidate support.
For each teacher-labelled candidate it reports per-image utility ranking
metrics for existing score controls and a grouped ridge probe trained on a
subset of the smoke images.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from .official_metric_adapter import evaluate_population, mapped_candidates
    from .legal_oracle import baseline_selection, legal_oracle_selection
except ImportError:  # direct ``python path/to/script.py`` entry point
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population, mapped_candidates
    from models.relation_decision_regret_v1.legal_oracle import baseline_selection, legal_oracle_selection


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(float(e.sum()), np.finfo(float).tiny)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(x), kind="stable")
    out = np.empty(len(order), dtype=float)
    out[order] = np.arange(len(order), dtype=float)
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels > 0
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = _rank(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def _features(row: dict) -> np.ndarray:
    scores = np.asarray(row["pred_scores"], dtype=float)[1:]
    probs = _softmax(scores)
    order = np.argsort(-probs, kind="stable")
    top = float(probs[order[0]])
    second = float(probs[order[1]]) if len(order) > 1 else 0.0
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    pred_rank = int(np.argsort(-np.asarray(row["pred_scores"], dtype=float), kind="stable").tolist().index(int(order[0]) + 1))
    return np.asarray([
        float(row["score"]),
        float(scores[order[0]]) if len(order) else 0.0,
        float(row["score"]) * top,
        top - second,
        entropy,
        float(pred_rank),
    ], dtype=float)


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    a = (x_train - mu) / sigma
    b = (x_eval - mu) / sigma
    a = np.concatenate([np.ones((len(a), 1)), a], axis=1)
    b = np.concatenate([np.ones((len(b), 1)), b], axis=1)
    reg = np.eye(a.shape[1], dtype=float) * 1e-2
    reg[0, 0] = 0.0
    theta = np.linalg.solve(a.T @ a + reg, a.T @ y_train)
    return b @ theta


def _tree_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1e-2, random_state=0,
    )
    model.fit(x_train, y_train)
    return model.predict(x_eval)


def _mass_capture(score: np.ndarray, utility: np.ndarray, fraction: float = 0.1) -> float:
    if len(score) == 0 or float(utility.sum()) <= 0:
        return float("nan")
    k = max(1, int(np.ceil(len(score) * fraction)))
    selected = np.argsort(-score, kind="stable")[:k]
    return float(utility[selected].sum() / utility.sum())


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(float(value)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    annotation = json.loads(args.psg.read_text())
    teacher = json.loads(args.teacher.read_text())
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    by_id = {str(item["img_id"]): item for item in predictions}
    support = {}
    for entry in annotation["data"]:
        image_id = str(entry["image_id"])
        if entry.get("relations") and image_id in by_id:
            support[image_id] = {int(row["row"]): row for row in mapped_candidates(
                entry, by_id[image_id], args.gt_seg_root, include_unmapped=True
            )[0]}

    images = sorted({str(row["image_id"]) for row in teacher["rows"]})
    cut = max(1, int(len(images) * 0.75))
    train_images = set(images[:cut])
    output = {"schema_version": 1, "contract": "fit-only utility ranking audit",
              "images": len(images), "train_images": len(train_images), "budgets": {}}
    for budget in teacher["budgets"]:
        grouped = {}
        labels = {}
        for row in teacher["rows"]:
            if int(row["budget"]) != int(budget):
                continue
            image_id = str(row["image_id"])
            labels.setdefault(image_id, {})
            for label in row["insertion"]:
                candidate = support.get(image_id, {}).get(int(label["row"]))
                if candidate is None:
                    continue
                labels[image_id][int(label["row"])] = float(label["insertion_delta_mr"])
            # For rows already occupying the baseline set, the exact marginal
            # signal is the utility lost on removal.  Combining this with
            # insertion labels covers the complete frozen candidate support.
            for label in row["removal"]:
                candidate = support.get(image_id, {}).get(int(label["row"]))
                if candidate is None:
                    continue
                labels[image_id][int(label["row"])] = float(label["removal_delta_mr"])
        for image_id, row_labels in labels.items():
            grouped[image_id] = [(_features(candidate), target, image_id, int(row_id))
                                 for row_id, target in row_labels.items()
                                 for candidate in [support[image_id][row_id]]]
        labelled = [value for values in grouped.values() for value in values]
        x = np.stack([v[0] for v in labelled])
        y = np.asarray([v[1] for v in labelled], dtype=float)
        ids = np.asarray([v[2] for v in labelled])
        names = ["pair_score", "pred_score", "joint", "margin", "entropy", "predicate_rank"]
        controls = {}
        for index, name in enumerate(names):
            controls[name] = {"auc": _auc(x[:, index], y),
                              "top10pct_utility_mass": _mass_capture(x[:, index], np.maximum(y, 0.0))}
        train = np.isin(ids, list(train_images))
        pred = _ridge_fit_predict(x[train], y[train], x[~train]) if (~train).any() else np.zeros(0)
        controls["ridge_probe_holdout"] = {
            "auc": _auc(pred, y[~train]),
            "top10pct_utility_mass": _mass_capture(pred, np.maximum(y[~train], 0.0)),
            "holdout_images": int(len(set(ids[~train]))),
        }
        tree_pred = _tree_fit_predict(x[train], y[train], x[~train]) if (~train).any() else np.zeros(0)
        controls["hist_gradient_probe_holdout"] = {
            "auc": _auc(tree_pred, y[~train]),
            "top10pct_utility_mass": _mass_capture(tree_pred, np.maximum(y[~train], 0.0)),
            "holdout_images": int(len(set(ids[~train]))),
        }
        # Evaluate the learned ranking as an actual fixed-K decoder on the
        # held-out images.  The GT is used only by the evaluator/oracle, never
        # by the ridge scores.
        holdout_records = []
        ridge_records = []
        oracle_records = []
        for image_id in images:
            if image_id in train_images or image_id not in support:
                continue
            entry = next(item for item in annotation["data"] if str(item["image_id"]) == image_id)
            candidates = list(support[image_id].values())
            candidate_features = np.stack([_features(c) for c in candidates])
            tree_scores = _tree_fit_predict(x[train], y[train], candidate_features)
            ridge_scores = _ridge_fit_predict(x[train], y[train], candidate_features)
            tree_ranked = sorted(zip(tree_scores.tolist(), candidates), key=lambda value: (-value[0], int(value[1]["row"])))
            ridge_ranked = sorted(zip(ridge_scores.tolist(), candidates), key=lambda value: (-value[0], int(value[1]["row"])))
            selected = [dict(candidate) for _, candidate in tree_ranked[: int(budget)]]
            ridge_selected = [dict(candidate) for _, candidate in ridge_ranked[: int(budget)]]
            baseline = baseline_selection(candidates, int(budget))
            oracle = legal_oracle_selection(candidates, entry["relations"], int(budget), "all")
            holdout_records.append({"image_id": image_id, "file_name": entry["file_name"],
                                   "bootstrap_group": entry["file_name"], "relations": entry["relations"],
                                   "selected": selected, "budget": int(budget)})
            ridge_records.append({"image_id": image_id, "file_name": entry["file_name"],
                                 "bootstrap_group": entry["file_name"], "relations": entry["relations"],
                                 "selected": ridge_selected, "budget": int(budget)})
            # Keep baseline and oracle in a parallel list for one common
            # evaluator call below.
            oracle_records.append({"image_id": image_id, "file_name": entry["file_name"],
                                   "bootstrap_group": entry["file_name"], "relations": entry["relations"],
                                   "selected": baseline, "budget": int(budget),
                                   "oracle_selected": oracle})
        learned_metrics = evaluate_population(holdout_records, len(annotation["predicate_classes"]))
        ridge_metrics = evaluate_population(ridge_records, len(annotation["predicate_classes"]))
        baseline_metrics = evaluate_population([
            {**record, "selected": parallel["selected"]}
            for record, parallel in zip(holdout_records, oracle_records)
        ], len(annotation["predicate_classes"]))
        oracle_metrics = evaluate_population([
            {**record, "selected": parallel["oracle_selected"]}
            for record, parallel in zip(holdout_records, oracle_records)
        ], len(annotation["predicate_classes"]))
        output["budgets"][str(budget)] = {"candidates": int(len(y)),
                                           "positive_labels": int(np.count_nonzero(y > 0)),
                                           "utility_sum": float(y.sum()), "controls": controls,
                                           "holdout_selection": {
                                               "images": len(holdout_records),
                                               "baseline": {"mR": _finite_or_none(baseline_metrics[f"mR@{budget}"]),
                                                            "R": float(baseline_metrics[f"R@{budget}"])},
                                               "hist_gradient_probe": {"mR": _finite_or_none(learned_metrics[f"mR@{budget}"]),
                                                                "R": float(learned_metrics[f"R@{budget}"])},
                                               "ridge_probe": {"mR": _finite_or_none(ridge_metrics[f"mR@{budget}"]),
                                                                "R": float(ridge_metrics[f"R@{budget}"])},
                                               "legal_oracle": {"mR": _finite_or_none(oracle_metrics[f"mR@{budget}"]),
                                                                "R": float(oracle_metrics[f"R@{budget}"])},
                                           }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")
    print(json.dumps(output, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
