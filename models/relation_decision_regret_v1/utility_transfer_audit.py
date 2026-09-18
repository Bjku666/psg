#!/usr/bin/env python3
"""Train a score-only utility probe on fit and evaluate it once on dev."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.legal_oracle import baseline_selection, legal_oracle_selection
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population, mapped_candidates
from models.relation_decision_regret_v1.utility_learnability_audit import _auc, _features, _mass_capture


def _load_support(annotation: dict, predictions: list[dict], root: Path) -> dict[str, dict[int, dict]]:
    by_id = {str(item["img_id"]): item for item in predictions}
    result = {}
    for entry in annotation["data"]:
        image_id = str(entry["image_id"])
        if entry.get("relations") and image_id in by_id:
            result[image_id] = {int(row["row"]): row for row in mapped_candidates(
                entry, by_id[image_id], root, include_unmapped=True
            )[0]}
    return result


def _labels(teacher: dict, support: dict[str, dict[int, dict]], budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, image_ids = [], [], []
    for image in teacher["rows"]:
        if int(image["budget"]) != int(budget):
            continue
        image_id = str(image["image_id"])
        row_labels = {}
        for item in image["insertion"]:
            row_labels[int(item["row"])] = float(item["insertion_delta_mr"])
        for item in image["removal"]:
            row_labels[int(item["row"])] = float(item["removal_delta_mr"])
        for row_id, target in row_labels.items():
            candidate = support.get(image_id, {}).get(row_id)
            if candidate is None:
                continue
            features.append(_features(candidate))
            targets.append(target)
            image_ids.append(image_id)
    return np.stack(features), np.asarray(targets, dtype=float), np.asarray(image_ids)


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1e-2, random_state=0,
    )


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1e-2, random_state=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-psg", required=True, type=Path)
    parser.add_argument("--fit-predictions", required=True, type=Path)
    parser.add_argument("--fit-teacher", required=True, type=Path)
    parser.add_argument("--dev-psg", required=True, type=Path)
    parser.add_argument("--dev-predictions", required=True, type=Path)
    parser.add_argument("--dev-teacher", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-set", choices=("f0", "f1"), default="f0")
    args = parser.parse_args()

    fit_annotation = json.loads(args.fit_psg.read_text())
    dev_annotation = json.loads(args.dev_psg.read_text())
    fit_teacher = json.loads(args.fit_teacher.read_text())
    dev_teacher = json.loads(args.dev_teacher.read_text())
    with args.fit_predictions.open("rb") as stream:
        fit_predictions = pickle.load(stream)
    with args.dev_predictions.open("rb") as stream:
        dev_predictions = pickle.load(stream)
    fit_support = _load_support(fit_annotation, fit_predictions, args.gt_seg_root)
    dev_support = _load_support(dev_annotation, dev_predictions, args.gt_seg_root)
    dev_entries = {str(item["image_id"]): item for item in dev_annotation["data"] if item.get("relations")}

    output = {"schema_version": 1, "contract": f"fit-trained, dev-evaluated once; {args.feature_set} features",
              "fit_images": len(fit_support), "dev_images": len(dev_support), "budgets": {}}
    for budget in fit_teacher["budgets"]:
        x_fit, y_fit, _ = _labels(fit_teacher, fit_support, int(budget))
        x_dev, y_dev, dev_ids = _labels(dev_teacher, dev_support, int(budget))
        if args.feature_set == "f1":
            def augment(x, support, ids):
                rows = []
                for image_id, row_id in zip(ids, range(len(x))):
                    rows.append(x[row_id])
                return x
            # The teacher labels are aligned in order with support lookup only
            # through row ids; rebuild the feature matrices with hidden tokens.
            def rebuild(teacher, support, base):
                out = []
                index = 0
                for image in teacher["rows"]:
                    if int(image["budget"]) != int(budget):
                        continue
                    image_id = str(image["image_id"])
                    labels = {}
                    for item in image["insertion"]:
                        labels[int(item["row"])] = float(item["insertion_delta_mr"])
                    for item in image["removal"]:
                        labels[int(item["row"])] = float(item["removal_delta_mr"])
                    for row_id in labels:
                        candidate = support.get(image_id, {}).get(row_id)
                        if candidate is not None and "pair_features" in candidate:
                            out.append(np.concatenate([base[index], np.asarray(candidate["pair_features"]).reshape(-1)]))
                            index += 1
                return np.stack(out)
            x_fit = rebuild(fit_teacher, fit_support, x_fit)
            x_dev = rebuild(dev_teacher, dev_support, x_dev)
        model = _model().fit(x_fit, y_fit)
        label_scores = model.predict(x_dev)
        classifier = _classifier()
        positive = y_fit > 0
        weights = np.where(positive, max(1.0, float((~positive).sum()) / max(1, int(positive.sum()))), 1.0)
        classifier.fit(x_fit, positive.astype(np.int8), sample_weight=weights)
        classifier_scores = classifier.predict_proba(x_dev)[:, 1]

        baseline_records, probe_records, classifier_records, oracle_records = [], [], [], []
        for image_id, candidates_by_row in dev_support.items():
            entry = dev_entries[image_id]
            candidates = list(candidates_by_row.values())
            scores = model.predict(np.stack([np.concatenate([_features(candidate), np.asarray(candidate["pair_features"]).reshape(-1)])
                                             if args.feature_set == "f1" else _features(candidate)
                                             for candidate in candidates]))
            ranked = sorted(zip(scores.tolist(), candidates), key=lambda item: (-item[0], int(item[1]["row"])))
            probe = [dict(candidate) for _, candidate in ranked[: int(budget)]]
            class_scores = classifier.predict_proba(np.stack([np.concatenate([_features(candidate), np.asarray(candidate["pair_features"]).reshape(-1)])
                                                              if args.feature_set == "f1" else _features(candidate)
                                                              for candidate in candidates]))[:, 1]
            class_ranked = sorted(zip(class_scores.tolist(), candidates), key=lambda item: (-item[0], int(item[1]["row"])))
            class_probe = [dict(candidate) for _, candidate in class_ranked[: int(budget)]]
            common = {"image_id": image_id, "file_name": entry["file_name"],
                      "bootstrap_group": entry["file_name"], "relations": entry["relations"],
                      "budget": int(budget)}
            baseline_records.append({**common, "selected": baseline_selection(candidates, int(budget))})
            probe_records.append({**common, "selected": probe})
            classifier_records.append({**common, "selected": class_probe})
            oracle_records.append({**common, "selected": legal_oracle_selection(
                candidates, entry["relations"], int(budget), "all")})
        metrics = {}
        for name, records in (("baseline", baseline_records), ("fit_tree_probe", probe_records),
                              ("fit_classifier_probe", classifier_records),
                              ("legal_oracle", oracle_records)):
            values = evaluate_population(records, len(dev_annotation["predicate_classes"]))
            metrics[name] = {"R": float(values[f"R@{budget}"])}
        output["budgets"][str(budget)] = {
            "fit_candidates": int(len(y_fit)), "dev_candidates": int(len(y_dev)),
            "dev_labelled_images": int(len(set(dev_ids.tolist()))),
            "dev_positive_labels": int(np.count_nonzero(y_dev > 0)),
            "dev_auc": _auc(label_scores, y_dev),
            "dev_top10pct_utility_mass": _mass_capture(label_scores, np.maximum(y_dev, 0.0)),
            "dev_positive_auc": _auc(classifier_scores, y_dev),
            "dev_positive_top10pct_utility_mass": _mass_capture(classifier_scores, np.maximum(y_dev, 0.0)),
            "selection": metrics,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
