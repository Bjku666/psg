#!/usr/bin/env python3
"""Run the zero-GPU P1.3 factorization qualification package."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.legal_oracle import baseline_selection, legal_oracle_selection
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population, mapped_candidates
from models.relation_decision_regret_v1.run_legal_decision_oracle import _bootstrap_delta
from models.relation_decision_regret_v1.oracle_factorization import (
    additivity_check,
    decode_top_l,
    fixed_pair_predicate_oracle,
)


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--psg", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--gt-seg-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", choices=("test", "train", "all"), default="test")
    p.add_argument("--budgets", nargs="+", type=int, default=[20, 50])
    p.add_argument("--depths", nargs="+", type=int, default=[2, 3, 5])
    p.add_argument("--bootstrap-replicates", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-images", type=int)
    return p.parse_args()


def _entries(annotation, split, max_images):
    test_ids = {str(x) for x in annotation.get("test_image_ids", [])}
    values = []
    for item in annotation.get("data", []):
        image_id = str(item["image_id"])
        if split == "test" and image_id not in test_ids:
            continue
        if split == "train" and image_id in test_ids:
            continue
        if item.get("relations"):
            values.append(item)
    values.sort(key=lambda x: str(x["image_id"]))
    return values[: int(max_images)] if max_images is not None else values


def _records(entries, support, selector, budget):
    return [{"image_id": str(e["image_id"]), "file_name": str(e["file_name"]),
             "bootstrap_group": str(e["file_name"]), "relations": e["relations"],
             "selected": selector(support[str(e["image_id"])], e), "budget": int(budget)}
            for e in entries]


def _metrics(records, num_predicates, budget):
    result = evaluate_population(records, num_predicates)
    mr = float(result[f"mR@{budget}"])
    rr = float(result[f"R@{budget}"])
    return {"mR": (mr if np.isfinite(mr) else None), "R": (rr if np.isfinite(rr) else None)}


def run(annotation, predictions, gt_seg_root, split="test", budgets=(20, 50), depths=(2, 3, 5),
        bootstrap_replicates=500, seed=0, max_images=None):
    entries = _entries(annotation, split, max_images)
    by_id = {str(x["img_id"]): x for x in predictions}
    support, missing = {}, []
    for entry in entries:
        item = by_id.get(str(entry["image_id"]))
        if item is None:
            support[str(entry["image_id"])] = []
            missing.append(str(entry["image_id"]))
        else:
            support[str(entry["image_id"])] = mapped_candidates(entry, item, gt_seg_root, include_unmapped=True)[0]
    num_predicates = len(annotation["predicate_classes"])
    output = {"schema_version": 1, "contract": {"support": "all frozen carrier pairs; GT mapping is evaluation-only", "single_mpo": True,
               "bootstrap_unit": "physical_file_name", "official_test": split == "test"},
              "split": split, "images": len(entries), "missing_predictions": missing,
              "budgets": list(budgets), "depths": list(depths), "oracle_2x2": {}, "top_l_controls": {},
              "simple_control_status": {
                  "kind": "fit-free protocol proxies",
                  "valid_for_P13_3_gate": False,
                  "reason": "planned L0/L1/L4 require fit-trained calibration and a full 56-predicate dev population",
              }, "separability": {}}
    # Test-time proxies may use class priors from the official training
    # population, never labels from the evaluated images.
    test_ids = {str(x) for x in annotation.get("test_image_ids", [])}
    prior_counts = Counter(
        int(relation[2])
        for item in annotation.get("data", [])
        if split == "test" and str(item["image_id"]) not in test_ids
        for relation in item.get("relations", [])
    )
    # Build the additive diagnostic records once.  This does not retain image
    # selections in the output, only the compact residual statistics.
    additive_records = [{"image_id": str(e["image_id"]), "relations": e["relations"],
                         "candidates": support[str(e["image_id"])]} for e in entries]
    output["separability"] = additivity_check(additive_records, num_predicates, seed=seed)
    for budget in budgets:
        baseline = _records(entries, support, lambda rows, e: baseline_selection(rows, budget), budget)
        base_metrics = _metrics(baseline, num_predicates, budget)
        for depth in depths:
            # B is the Top-1 predicate oracle for every L; only D changes its
            # predicate evidence depth.  This keeps the 2x2 decomposition
            # interpretable as A/B/C/D rather than changing two axes at once.
            b = _records(entries, support, lambda rows, e: legal_oracle_selection(rows, e["relations"], budget, 1), budget)
            c = _records(entries, support, lambda rows, e, d=depth: fixed_pair_predicate_oracle(rows, e["relations"], budget, d), budget)
            drec = b
            # Recompute D independently to make the 2x2 implementation audit-visible.
            drec = _records(entries, support, lambda rows, e, dep=depth: legal_oracle_selection(rows, e["relations"], budget, dep), budget)
            bm, cm, dm = (_metrics(x, num_predicates, budget) for x in (b, c, drec))
            metric_key = "mR" if all(x["mR"] is not None for x in (base_metrics, bm, cm, dm)) else "R"
            h_slot = (bm[metric_key] - base_metrics[metric_key]) * 100.0
            h_pred = (cm[metric_key] - base_metrics[metric_key]) * 100.0
            h_joint = (dm[metric_key] - cm[metric_key]) * 100.0
            coupling = (dm[metric_key] - bm[metric_key] - cm[metric_key] + base_metrics[metric_key]) * 100.0
            output["oracle_2x2"][f"L{depth}@{budget}"] = {
                "A_baseline_pair_top1": base_metrics, "B_oracle_slots_top1": bm,
                "C_baseline_slots_predicate_topL": cm, "D_oracle_slots_topL": dm,
                "headroom_pp": {"slot_top1": h_slot, "predicate_fixed_pairs": h_pred,
                                 "slot_after_predicate": h_joint, "decision_factorization_coupling": coupling,
                                 "metric_used": metric_key},
                "bootstrap": {
                                     "B_minus_A": _bootstrap_delta(baseline, b, num_predicates, budget, bootstrap_replicates, seed),
                                     "C_minus_A": _bootstrap_delta(baseline, c, num_predicates, budget, bootstrap_replicates, seed),
                                     "D_minus_A": _bootstrap_delta(baseline, drec, num_predicates, budget, bootstrap_replicates, seed),
                                 } if bootstrap_replicates > 0 else {"replicates": 0},
            }
        counts = {}
        for mode in [f"L{i}" for i in range(5)]:
            rec = _records(entries, support, lambda rows, e, m=mode: decode_top_l(
                rows, (), budget, max(depths), m, predicate_counts=prior_counts), budget)
            counts[f"{mode}@{budget}"] = _metrics(rec, num_predicates, budget)
        output["top_l_controls"].update(counts)
    return output


def main():
    args = _args()
    annotation = json.loads(args.psg.read_text())
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    result = run(annotation, predictions, args.gt_seg_root, args.split, tuple(args.budgets), tuple(args.depths),
                 args.bootstrap_replicates, args.seed, args.max_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"split": result["split"], "images": result["images"], "missing": len(result["missing_predictions"]),
                      "oracle_keys": sorted(result["oracle_2x2"]), "separability": result["separability"]}, indent=2))


if __name__ == "__main__":
    main()
