#!/usr/bin/env python3
"""Audit where the GT predicate appears in each frozen candidate's evidence."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import mapped_candidates


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def audit(annotation: dict, predictions: list[dict], gt_seg_root: Path, split: str = "test",
          max_images: int | None = None) -> dict:
    by_id = {str(item["img_id"]): item for item in predictions}
    test_ids = {str(value) for value in annotation.get("test_image_ids", [])}
    entries = []
    for entry in annotation.get("data", []):
        image_id = str(entry["image_id"])
        in_test = image_id in test_ids
        if split == "test" and not in_test:
            continue
        if split == "train" and in_test:
            continue
        if entry.get("relations"):
            entries.append(entry)
    entries.sort(key=lambda item: str(item["image_id"]))
    if max_images is not None:
        entries = entries[: int(max_images)]
    depths = (1, 2, 3, 5)
    counts = {str(depth): 0 for depth in depths}
    total = 0
    per_image = []
    missing = []
    for entry in entries:
        item = by_id.get(str(entry["image_id"]))
        if item is None:
            missing.append(str(entry["image_id"]))
            continue
        rows, _ = mapped_candidates(entry, item, gt_seg_root)
        gt = {(int(s), int(o), int(r)) for s, o, r in entry.get("relations", [])}
        image_ranks = []
        by_pair = {tuple(row["gt_pair"]): row for row in rows if row.get("gt_pair") is not None}
        for s, o, predicate in sorted(gt):
            row = by_pair.get((s, o))
            if row is None:
                continue
            scores = np.asarray(row["pred_scores"], dtype=float)[1:]
            order = np.argsort(-scores, kind="stable")
            rank = int(np.flatnonzero(order == int(predicate))[0]) + 1 if int(predicate) in order else None
            if rank is None:
                continue
            total += 1
            image_ranks.append(rank)
            for depth in depths:
                counts[str(depth)] += int(rank <= depth)
        per_image.append({"image_id": str(entry["image_id"]), "file_name": str(entry["file_name"]),
                          "ranks": image_ranks, "n_relations": len(entry.get("relations", []))})
    result = {
        "schema_version": 1,
        "contract": "strict SingleMPO mapped candidate support; predicate rank excludes NONE",
        "split": split,
        "images": len(entries),
        "missing_predictions": missing,
        "relations_with_mapped_candidate": total,
        "fraction_rank_le": {str(depth): (counts[str(depth)] / total if total else None) for depth in depths},
        "counts_rank_le": counts,
        "per_image": per_image,
    }
    return result


def main() -> None:
    args = _args()
    annotation = json.loads(args.psg.read_text())
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    result = audit(annotation, predictions, args.gt_seg_root, args.split, args.max_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
