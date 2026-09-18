#!/usr/bin/env python3
"""Run the legal fixed-K decision-regret qualification audit.

The oracle is GT-aware only after strict mask matching.  It can choose a
predicate already present in a candidate row's top-M evidence, but it cannot
create pairs, masks, or duplicate directed pairs.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.legal_oracle import (
    baseline_selection,
    legal_oracle_selection,
)
from models.relation_decision_regret_v1.official_metric_adapter import (
    evaluate_population,
    mapped_candidates,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--budgets", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--depths", nargs="+", default=["1", "2", "3", "5", "all"])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-per-image", action="store_true",
                        help="retain selected candidate rows; off by default for full runs")
    return parser.parse_args()


def _select_entries(annotation: dict, split: str, max_images: int | None) -> list[dict]:
    test_ids = {str(value) for value in annotation.get("test_image_ids", [])}
    entries = []
    for item in annotation.get("data", []):
        image_id = str(item["image_id"])
        if split == "test" and image_id not in test_ids:
            continue
        if split == "train" and image_id in test_ids:
            continue
        if item.get("relations"):
            entries.append(item)
    entries.sort(key=lambda item: str(item["image_id"]))
    return entries[: int(max_images)] if max_images is not None else entries


def _metric(records: list[dict], num_predicates: int, budget: int) -> dict:
    result = evaluate_population(records, num_predicates)
    return {f"mR@{budget}": float(result[f"mR@{budget}"]),
            f"R@{budget}": float(result[f"R@{budget}"])}


def _bootstrap_delta(baseline: list[dict], oracle: list[dict], num_predicates: int,
                     budget: int, replicates: int, seed: int) -> dict:
    groups = sorted({str(row["bootstrap_group"]) for row in baseline})
    if not groups or replicates <= 0:
        return {"unit": "physical_file_name", "groups": len(groups), "replicates": 0,
                "estimate_pp": None, "ci95_pp": None}
    # Evaluate each image once.  The grouped bootstrap resamples these image
    # statistics; it must not rerun mask/relation evaluation per replicate.
    base_eval = evaluate_population(baseline, num_predicates)
    oracle_eval = evaluate_population(oracle, num_predicates)
    base_per = np.stack([row["per_predicate_recall"] for row in base_eval["per_image"]])
    oracle_per = np.stack([row["per_predicate_recall"] for row in oracle_eval["per_image"]])
    group_indices = {group: [index for index, row in enumerate(baseline)
                             if str(row["bootstrap_group"]) == group] for group in groups}

    def metric(per: np.ndarray, indices: list[int]) -> float:
        class_recall = np.nanmean(per[indices], axis=0)
        # A bootstrap draw can omit an ultra-rare predicate entirely.  The
        # official evaluator omits classes with no denominator; mirror that
        # behavior rather than allowing one absent class to make the CI NaN.
        return float(np.nanmean(class_recall))

    point_base = metric(base_per, list(range(len(baseline))))
    point_oracle = metric(oracle_per, list(range(len(oracle))))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(replicates)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = [index for group in sampled for index in group_indices[str(group)]]
        values.append((metric(oracle_per, indices) - metric(base_per, indices)) * 100.0)
    values = np.asarray(values, dtype=float)
    return {"unit": "physical_file_name", "groups": len(groups), "replicates": int(replicates),
            "estimate_pp": float((point_oracle - point_base) * 100.0),
            "bootstrap_mean_pp": float(values.mean()),
            "ci95_pp": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def run(annotation: dict, predictions: list[dict], gt_seg_root: Path, split: str = "test",
        budgets: tuple[int, ...] = (20, 50), depths: tuple[int | str, ...] = (1, 2, 3, 5, "all"),
        max_images: int | None = None, bootstrap_replicates: int = 2000, seed: int = 0,
        include_per_image: bool = False) -> dict:
    entries = _select_entries(annotation, split, max_images)
    by_id = {str(item["img_id"]): item for item in predictions}
    num_predicates = len(annotation["predicate_classes"])
    missing = []
    coverage_failure = []
    mapped = {}
    for entry in entries:
        item = by_id.get(str(entry["image_id"]))
        if item is None:
            missing.append(str(entry["image_id"]))
            mapped[str(entry["image_id"])] = []
        else:
            mapped[str(entry["image_id"])] = mapped_candidates(
                entry, item, gt_seg_root, include_unmapped=True
            )[0]
        has_mapped = any(row.get("gt_pair") is not None for row in mapped[str(entry["image_id"])])
        if not has_mapped and str(entry["image_id"]) not in coverage_failure:
            coverage_failure.append(str(entry["image_id"]))

    strategies = {}
    for budget in budgets:
        baseline_records = []
        for entry in entries:
            rows = mapped[str(entry["image_id"])]
            baseline_records.append({"image_id": str(entry["image_id"]), "file_name": str(entry["file_name"]),
                                     "bootstrap_group": str(entry["file_name"]), "relations": entry["relations"],
                                     "selected": baseline_selection(rows, budget), "budget": budget})
        base_name = f"baseline@{budget}"
        strategies[base_name] = {"metrics": _metric(baseline_records, num_predicates, budget)}
        if include_per_image:
            strategies[base_name]["per_image"] = baseline_records
        for depth in depths:
            oracle_records = []
            for entry in entries:
                rows = mapped[str(entry["image_id"])]
                oracle_records.append({"image_id": str(entry["image_id"]), "file_name": str(entry["file_name"]),
                                       "bootstrap_group": str(entry["file_name"]), "relations": entry["relations"],
                                       "selected": legal_oracle_selection(rows, entry["relations"], budget, depth),
                                       "budget": budget})
            label = f"oracle_top{depth}" if depth != "all" else "oracle_all"
            strategies[f"{label}@{budget}"] = {
                "metrics": _metric(oracle_records, num_predicates, budget),
                "bootstrap": _bootstrap_delta(baseline_records, oracle_records, num_predicates, budget,
                                               bootstrap_replicates, seed),
            }
            if include_per_image:
                strategies[f"{label}@{budget}"]["per_image"] = oracle_records
    return {
        "schema_version": 1,
        "contract": {"evaluator": "image-wise Fair PSG recall", "dedup": "fail", "mask_iou": ">0.5",
                     "support": "frozen mapped carrier pairs", "bootstrap_unit": "physical_file_name"},
        "split": split, "images": len(entries), "missing_predictions": missing,
        "coverage_failure": {"image_ids": coverage_failure, "count": len(coverage_failure),
                             "definition": "no legal mapped candidate support; independent of predicate/ranking failure"},
        "budgets": list(budgets), "depths": list(depths), "strategies": strategies,
        "gates": {"metric_contract_locked": True, "cmud_authorized": False},
    }


def main() -> None:
    args = _args()
    annotation = json.loads(args.psg.read_text())
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    depths = tuple("all" if str(value) == "all" else int(value) for value in args.depths)
    result = run(annotation, predictions, args.gt_seg_root, args.split, tuple(args.budgets), depths,
                 args.max_images, args.bootstrap_replicates, args.seed, args.include_per_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "strategies"}, indent=2))


if __name__ == "__main__":
    main()
