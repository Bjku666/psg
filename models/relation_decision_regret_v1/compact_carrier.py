#!/usr/bin/env python3
"""Compact one raw P1.4 inference shard into mapped Top-K carrier rows.

The raw Fair-PSG pickle repeats full masks and all directed pairs.  P1.4 only
needs the frozen top pair slots and their predicate hypotheses.  This command
performs the registered SingleMPO mapping once, stores float32 scores and
optional float16 hidden tokens, and drops rebuildable masks/bounding boxes.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import mapped_candidates


def compact(annotation: dict, predictions: list[dict], gt_seg_root: Path,
            top_pairs: int = 50, keep_hidden: bool = True) -> dict:
    by_id = {str(item["img_id"]): item for item in predictions}
    images, missing = [], []
    for entry in annotation["data"]:
        image_id = str(entry["image_id"])
        prediction = by_id.get(image_id)
        if prediction is None and entry.get("relations"):
            missing.append(image_id)
        if prediction is None:
            candidates = []
        else:
            candidates, _ = mapped_candidates(
                entry, prediction, gt_seg_root, include_unmapped=True
            )
        # Match the carrier's pair-slot ordering.  Keep exactly the strongest
        # K physical rows; predicate arbitration never expands this support.
        candidates = sorted(
            candidates, key=lambda row: (-float(row["score"]), int(row["row"]))
        )[: int(top_pairs)]
        compact_rows = []
        for candidate in candidates:
            row = {
                "row": int(candidate["row"]),
                "pair": tuple(map(int, candidate["pair"])),
                "gt_pair": (None if candidate.get("gt_pair") is None
                            else tuple(map(int, candidate["gt_pair"]))),
                "score": float(candidate["score"]),
                "pred_scores": np.asarray(candidate["pred_scores"], dtype=np.float32),
            }
            if keep_hidden and "pair_features" in candidate:
                row["pair_features"] = np.asarray(candidate["pair_features"], dtype=np.float16)
            compact_rows.append(row)
        images.append({
            "image_id": image_id,
            "file_name": str(entry["file_name"]),
            "bootstrap_group": str(entry["file_name"]),
            "relations": [tuple(map(int, relation)) for relation in entry.get("relations", [])],
            "candidates": compact_rows,
        })
    return {
        "schema_version": 1,
        "contract": {
            "support": f"frozen carrier top-{int(top_pairs)} physical pairs",
            "mapping": "class-compatible mask IoU > 0.5; SingleMPO one-to-one",
            "predicate_scores_dtype": "float32",
            "pair_features_dtype": "float16" if keep_hidden else None,
            "confirm_locked": True,
        },
        "predicate_classes": list(annotation["predicate_classes"]),
        "images": images,
        "missing_predictions": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-pairs", type=int, default=50)
    parser.add_argument("--drop-hidden", action="store_true")
    args = parser.parse_args()
    annotation = json.loads(args.annotation.read_text())
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    result = compact(annotation, predictions, args.gt_seg_root, args.top_pairs,
                     keep_hidden=not args.drop_hidden)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({"images": len(result["images"]),
                      "missing": len(result["missing_predictions"]),
                      "bytes": args.output.stat().st_size,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
