#!/usr/bin/env python3
"""Evaluate D0-D7 score controls on the same frozen mapped support."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population, mapped_candidates
from models.relation_decision_regret_v1.simple_decoders import decode_rows, predicate_histogram


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--budgets", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--include-per-image", action="store_true",
                        help="retain selected candidate rows; off by default for full runs")
    return parser.parse_args()


def _entries(annotation: dict, split: str, max_images: int | None) -> list[dict]:
    test_ids = {str(value) for value in annotation.get("test_image_ids", [])}
    values = []
    for item in annotation.get("data", []):
        image_id = str(item["image_id"])
        if split == "test" and image_id not in test_ids:
            continue
        if split == "train" and image_id in test_ids:
            continue
        if item.get("relations"):
            values.append(item)
    values.sort(key=lambda item: str(item["image_id"]))
    return values[: int(max_images)] if max_images is not None else values


def run(annotation: dict, predictions: list[dict], gt_seg_root: Path, split: str = "test",
        budgets: tuple[int, ...] = (20, 50), max_images: int | None = None,
        include_per_image: bool = False) -> dict:
    entries = _entries(annotation, split, max_images)
    by_id = {str(item["img_id"]): item for item in predictions}
    support = {}
    missing = []
    for entry in entries:
        item = by_id.get(str(entry["image_id"]))
        if item is None:
            support[str(entry["image_id"])] = []
            missing.append(str(entry["image_id"]))
        else:
            support[str(entry["image_id"])] = mapped_candidates(
                entry, item, gt_seg_root, include_unmapped=True
            )[0]
    predicate_counts = predicate_histogram([relation for entry in entries for relation in entry["relations"]])
    output = {}
    num_predicates = len(annotation["predicate_classes"])
    for budget in budgets:
        for mode in [f"D{i}" for i in range(8)]:
            records = []
            for entry in entries:
                records.append({"image_id": str(entry["image_id"]), "file_name": str(entry["file_name"]),
                                "bootstrap_group": str(entry["file_name"]), "relations": entry["relations"],
                                "selected": decode_rows(support[str(entry["image_id"])], budget, mode, predicate_counts),
                                "budget": budget})
            metrics = evaluate_population(records, num_predicates)
            output[f"{mode}@{budget}"] = {"mR": float(metrics[f"mR@{budget}"]),
                                          "R": float(metrics[f"R@{budget}"])}
            if include_per_image:
                output[f"{mode}@{budget}"]["per_image"] = records
    return {"schema_version": 1, "contract": "same frozen support and strict SingleMPO mapping",
            "split": split, "images": len(entries), "missing_predictions": missing,
            "budgets": list(budgets), "controls": output,
            "predicate_counts": {str(key): int(value) for key, value in predicate_counts.items()}}


def main() -> None:
    args = _args()
    annotation = json.loads(args.psg.read_text())
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    result = run(annotation, predictions, args.gt_seg_root, args.split, tuple(args.budgets), args.max_images,
                 args.include_per_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "controls"}, indent=2))


if __name__ == "__main__":
    main()
