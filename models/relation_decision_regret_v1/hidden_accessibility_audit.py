#!/usr/bin/env python3
"""E1: audit whether final relation tokens expose utility beyond scores."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population, mapped_candidates
from models.relation_decision_regret_v1.utility_learnability_audit import _auc, _features, _mass_capture


def feature_vector(candidate: dict, prediction: dict, feature_set: str) -> np.ndarray:
    base = _features(candidate)
    if feature_set == "f0":
        return base
    if "pair_features" not in candidate:
        raise ValueError("prediction has no pair_features; export them before F1/F2")
    hidden = np.asarray(candidate["pair_features"], dtype=np.float32).reshape(-1)
    if feature_set == "f1":
        return np.concatenate([base, hidden])
    pair = np.asarray(candidate["pair"], dtype=np.int64)
    boxes = np.asarray(prediction["bboxes"], dtype=np.float32)
    labels = np.asarray(prediction["box_label"], dtype=np.float32)
    height, width = np.asarray(prediction["mask"]).shape[-2:]
    geometry = []
    for index in pair:
        x1, y1, x2, y2 = boxes[index]
        geometry.extend(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                         (x2 - x1) / width, (y2 - y1) / height))
    endpoint_classes = [labels[pair[0]] / 200.0, labels[pair[1]] / 200.0]
    return np.concatenate([base, hidden, np.asarray(geometry + endpoint_classes, dtype=np.float32)])


def fit_predict(x_train, y_train, x_eval, kind):
    model = Ridge(alpha=1.0) if kind == "ridge" else HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    model.fit(x_train, y_train)
    return model.predict(x_eval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-sets", nargs="+", choices=("f0", "f1", "f2"), default=["f0"])
    args = parser.parse_args()
    annotation = json.loads(args.psg.read_text())
    teacher = json.loads(args.teacher.read_text())
    predictions = pickle.loads(args.predictions.read_bytes())
    by_id = {str(item["img_id"]): item for item in predictions}
    entries = {str(row["image_id"]): row for row in annotation["data"]}
    teacher_ids = {str(row["image_id"]) for row in teacher["rows"]}
    support = {}
    for image_id in sorted(teacher_ids & set(by_id) & set(entries)):
        support[image_id], _ = mapped_candidates(entries[image_id], by_id[image_id], args.gt_seg_root, include_unmapped=True)
    images = sorted(support)
    cut = max(1, int(len(images) * 0.75))
    train_ids, holdout_ids = set(images[:cut]), set(images[cut:])
    output = {"schema_version": 1, "contract": "fit-only hidden accessibility audit",
              "train_images": len(train_ids), "holdout_images": len(holdout_ids), "feature_sets": {}}
    for feature_set in args.feature_sets:
        try:
            labels = {image_id: {} for image_id in images}
            for row in teacher["rows"]:
                image_id = str(row["image_id"])
                if image_id not in labels:
                    continue
                for item in list(row.get("insertion", [])) + list(row.get("removal", [])):
                    # insertion/removal rows share the same row id; insertion is
                    # preferred when both labels exist for a selected candidate.
                    key = int(item["row"])
                    labels[image_id].setdefault(key, float(item.get("insertion_delta_mr", item.get("removal_delta_mr", 0.0))))
            aligned = [(image_id, candidate, labels[image_id][int(candidate["row"])] )
                       for image_id in images for candidate in support[image_id]
                       if int(candidate["row"]) in labels[image_id]]
            if not aligned:
                raise ValueError("no teacher-aligned candidates")
            x = np.stack([feature_vector(candidate, by_id[image_id], feature_set)
                          for image_id, candidate, _ in aligned])
            y = np.asarray([target for _, _, target in aligned], dtype=float)
            ids = np.asarray([image_id for image_id, _, _ in aligned])
            train = np.isin(ids, list(train_ids))
            spec = {"dimensions": int(x.shape[1]), "candidates": int(len(x)),
                    "positive": int((y > 0).sum()), "models": {}}
            for kind in ("ridge", "hist_gradient"):
                pred = fit_predict(x[train], y[train], x[~train], kind)
                model_result = {"auc": _auc(pred, y[~train]),
                                "top10pct_utility_mass": _mass_capture(pred, np.maximum(y[~train], 0.0))}
                for budget in (20, 50):
                    records = []
                    for image_id in sorted(holdout_ids):
                        candidates = support[image_id]
                        cx = np.stack([feature_vector(candidate, by_id[image_id], feature_set) for candidate in candidates])
                        scores = fit_predict(x[train], y[train], cx, kind)
                        selected = [dict(candidate) for _, candidate in sorted(zip(scores, candidates), key=lambda z: (-z[0], int(z[1]["row"])))[:budget]]
                        entry = entries[image_id]
                        records.append({"image_id": image_id, "file_name": entry["file_name"], "bootstrap_group": entry["file_name"], "relations": entry["relations"], "selected": selected, "budget": budget})
                    metrics = evaluate_population(records, len(annotation["predicate_classes"]))
                    model_result[f"selection@{budget}"] = {"mR": metrics.get(f"mR@{budget}"), "R": metrics.get(f"R@{budget}")}
                spec["models"][kind] = model_result
            output["feature_sets"][feature_set] = spec
        except ValueError as exc:
            output["feature_sets"][feature_set] = {"status": "unavailable", "reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")
    print(json.dumps(output, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
