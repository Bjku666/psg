#!/usr/bin/env python3
"""CPU proxy pilot for the MPU/HPA action-unit learner gate.

This is deliberately a fit->dev diagnostic.  It never reads official-test
labels and uses frozen carrier predictions.  ``HPA`` here is a hidden-feature
proxy, not the planned intra-pair attention model.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.legal_oracle import _predicate_options, _predicate_order, baseline_selection, legal_oracle_selection
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population, mapped_candidates


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-psg", required=True, type=Path)
    p.add_argument("--fit-predictions", required=True, type=Path)
    p.add_argument("--dev-psg", required=True, type=Path)
    p.add_argument("--dev-predictions", required=True, type=Path)
    p.add_argument("--gt-seg-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--max-train-candidates", type=int, default=120000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_support(annotation, predictions, gt_root):
    by_id = {str(x["img_id"]): x for x in predictions}
    entries = {str(x["image_id"]): x for x in annotation["data"] if x.get("relations")}
    support = {}
    for image_id, entry in entries.items():
        item = by_id.get(image_id)
        if item is not None:
            support[image_id] = mapped_candidates(entry, item, gt_root, include_unmapped=True)[0]
    return entries, support


def _hypotheses(row, depth, relations):
    scores = np.asarray(row["pred_scores"], dtype=np.float32)
    order = _predicate_order(row)
    options = _predicate_options(row, depth)
    gt = {(int(s), int(o), int(r)) for s, o, r in relations}
    mapped_pair = row.get("gt_pair")
    pair = tuple(map(int, mapped_pair)) if mapped_pair is not None else None
    out = []
    for predicate in options:
        current = dict(row)
        current["pred"] = int(predicate)
        current["pred_score"] = float(scores[predicate + 1])
        current["predicate_rank"] = int(next(i for i, value in enumerate(order) if int(value) == predicate + 1))
        # Keep unmapped carrier pairs as ordinary negatives.  Dropping them
        # would expose evaluator-only GT mask matching at inference time and
        # make the dev score an oracle rather than a deployable decoder.
        current["label"] = int(pair is not None and (pair[0], pair[1], int(predicate)) in gt)
        out.append(current)
    return out


def _features(row, mode):
    scores = np.asarray(row["pred_scores"], dtype=np.float32)[1:]
    pred = int(row["pred"])
    top = float(scores.max()) if len(scores) else 0.0
    second = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0.0
    base = np.concatenate((
        np.asarray([float(row["score"]), float(row["pred_score"]), float(row["predicate_rank"]),
                    float(row["pred_score"]) - second, float(row["pred_score"]) - top], dtype=np.float32),
        scores.astype(np.float32),
        np.eye(56, dtype=np.float32)[pred],
    ))
    if mode == "MPU":
        return base
    hidden = np.asarray(row.get("pair_features", np.zeros(384)), dtype=np.float32).reshape(-1)
    # HPA sees the frozen pair representation and explicit within-pair
    # competition statistics, but still emits one legal hypothesis per pair.
    return np.concatenate((base, hidden, np.asarray([top, second, top - second], dtype=np.float32)))


def _make_training(entries, support, mode, depth, limit, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for image_id in sorted(support):
        for candidate in support[image_id]:
            rows.extend(_hypotheses(candidate, depth, entries[image_id]["relations"]))
    positives = [x for x in rows if x["label"]]
    negatives = [x for x in rows if not x["label"]]
    keep_neg = max(0, int(limit) - len(positives))
    if len(negatives) > keep_neg:
        negatives = [negatives[i] for i in rng.choice(len(negatives), size=keep_neg, replace=False)]
    chosen = positives + negatives
    rng.shuffle(chosen)
    x = np.stack([_features(row, mode) for row in chosen]).astype(np.float32)
    y = np.asarray([row["label"] for row in chosen], dtype=np.int8)
    return x, y, rows


def _decode(entries, support, model, mode, depth, budget):
    records = []
    for image_id in sorted(support):
        entry = entries[image_id]
        hypotheses = [h for row in support[image_id] for h in _hypotheses(row, depth, entry["relations"])]
        if not hypotheses:
            selected = []
        else:
            scores = model.predict_proba(np.stack([_features(row, mode) for row in hypotheses]).astype(np.float32))[:, 1]
            best = {}
            for score, row in zip(scores, hypotheses):
                key = tuple(map(int, row["pair"]))
                old = best.get(key)
                if old is None or (float(score), -int(row["predicate_rank"])) > (old[0], -int(old[1]["predicate_rank"])):
                    best[key] = (float(score), row)
            selected = [dict(row) for _, row in sorted(best.values(), key=lambda x: (-x[0], int(x[1]["row"])))[:budget]]
        common = {"image_id": image_id, "file_name": entry["file_name"], "bootstrap_group": entry["file_name"],
                  "relations": entry["relations"], "budget": budget}
        records.append({**common, "selected": selected})
    return records


def main():
    args = _args()
    fit_ann = json.loads(args.fit_psg.read_text()); dev_ann = json.loads(args.dev_psg.read_text())
    fit_pred = pickle.loads(args.fit_predictions.read_bytes()); dev_pred = pickle.loads(args.dev_predictions.read_bytes())
    fit_entries, fit_support = _load_support(fit_ann, fit_pred, args.gt_seg_root)
    dev_entries, dev_support = _load_support(dev_ann, dev_pred, args.gt_seg_root)
    output = {"schema_version": 1, "contract": {
                  "fit_trained_dev_evaluated": True,
                  "official_test_labels_used": False,
                  "frozen_carrier_support": True,
                  "unmapped_carrier_pairs_retained": True,
                  "single_mpo": True,
              },
              "model_scope": {
                  "MPU": "score/logit calibration-feature gradient-boosting proxy",
                  "HPA": "MPU features plus frozen pair representation; not intra-pair attention",
              },
              "fit_images": len(fit_support), "dev_images": len(dev_support), "depth": args.depth, "budgets": {}}
    for budget in (20, 50):
        output["budgets"][str(budget)] = {}
        for mode in ("MPU", "HPA"):
            x, y, all_fit = _make_training(fit_entries, fit_support, mode, args.depth, args.max_train_candidates, args.seed)
            model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=31,
                                                   l2_regularization=1e-2, random_state=args.seed)
            model.fit(x, y)
            scores = model.predict_proba(x)[:, 1]
            records = _decode(dev_entries, dev_support, model, mode, args.depth, budget)
            metrics = evaluate_population(records, len(dev_ann["predicate_classes"]))
            output["budgets"][str(budget)][mode] = {
                "train_candidates": int(len(x)), "train_positive": int(y.sum()), "train_auc": float(roc_auc_score(y, scores)),
                "dev_images": len(records), "dev_R": float(metrics[f"R@{budget}"]),
                "dev_mR": (float(metrics[f"mR@{budget}"]) if np.isfinite(metrics[f"mR@{budget}"]) else None),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
