#!/usr/bin/env python3
"""Qualify the corrected P1A entity-admission actionability contract.

The oracle is solved over candidates produced by full-pool pixel competition;
it never reruns competition for a selected subset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0_v1.audit import aggregate_counts, image_counts
from models.gssr_p0b_v1.balanced_oracle import gt_set_oracle, inverse_predicate_weights, predicate_counts
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping
from models.gssr_p0c_v1.raw_query_schema import validate_image_record
from models.gssr_p1_v2.entity_admission import (
    assemble_admitted_entities, native_candidate_query_ids, validate_native_assembly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("all", "train"), default="train")
    parser.add_argument("--split-manifest", type=Path,
                        help="grouped fit/dev/confirm JSON produced from official train")
    parser.add_argument("--partition", choices=("fit", "dev", "confirm"),
                        help="evaluate only this registered partition")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-mask-area-threshold", type=float, default=0.8)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--raw-query-gap", type=float,
        help="same-partition 200-raw-query fixed-K oracle gap; omit when unavailable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    psg = json.loads(args.psg.read_text())
    records = {}
    with args.manifest.open() as stream:
        for line in stream:
            record = json.loads(line)
            validate_image_record(record)
            records[str(record["file_name"])] = record
    test_ids = {str(i) for i in psg.get("test_image_ids", [])}
    items = [item for item in psg.get("data", [])
             if item.get("relations") and (args.split == "all" or str(item["image_id"]) not in test_ids)]
    items.sort(key=lambda item: str(item["image_id"]))
    if bool(args.split_manifest) != bool(args.partition):
        raise ValueError("--split-manifest and --partition must be supplied together")
    if args.partition:
        split_document = json.loads(args.split_manifest.read_text())
        partition_ids = {str(value) for value in split_document[args.partition]}
        items = [item for item in items if str(item["image_id"]) in partition_ids]
    if args.max_images is not None:
        items = items[:args.max_images]
    if not items:
        raise ValueError("no eligible relation-bearing images")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    num_predicates = len(psg["predicate_classes"])
    # Predicate weights are always computed on the complete requested
    # partition, so independently executed shards optimize one objective.
    weights = inverse_predicate_weights(
        predicate_counts([np.asarray(i["relations"]) for i in items], num_predicates)
    )
    partition_images = len(items)
    items = items[args.shard_index::args.num_shards]
    native_rows, oracle_rows, full_rows = [], [], []
    sanity_checked = 0
    for item in items:
        filename = str(item["file_name"])
        record = records.get(filename)
        if record is None:
            raise ValueError(f"manifest lacks {filename}")
        artifact = Path(record["artifact_file"])
        if not artifact.is_absolute():
            artifact = args.manifest.parent / artifact
        with np.load(artifact) as data:
            winner_map = np.asarray(data["pre_admission_winner_map"])
            query_ids = np.asarray(data["pre_admission_query_ids"], dtype=int)
            won = np.asarray(data["pre_admission_won_area"], dtype=int)
            original = np.asarray(data["pre_admission_original_area"], dtype=int)
            ratios = np.asarray(data["pre_admission_area_ratio"], dtype=float)
            if "official_panoptic_segmentation_key" in record:
                official_segmentation = np.asarray(
                    data[str(record["official_panoptic_segmentation_key"])], dtype=np.int32
                )
            else:
                official_segmentation = np.asarray(record["official_panoptic_segmentation"])
        by_query = {int(q["query_id"]): q for q in record["queries"]}
        candidates = [{
            "query_id": int(qid), "label_id": int(by_query[int(qid)]["predicted_class"]),
            "score": float(by_query[int(qid)]["class_score"]),
            "won_area": int(wa), "original_area": int(oa), "area_ratio": float(ratio),
            "winning_mask": winner_map == int(qid),
        } for qid, wa, oa, ratio in zip(query_ids, won, original, ratios)]
        native_ids = native_candidate_query_ids(candidates, args.overlap_mask_area_threshold)
        native = assemble_admitted_entities(winner_map, candidates, native_ids)
        official_info = record.get("official_segments_info", [])
        official_query_ids = record.get("official_segment_query_ids", [])
        if "official_segments_info" in record:
            validate_native_assembly(
                official_segmentation, official_info,
                official_query_ids, {"winner_map": winner_map, "candidates": candidates}, native_ids,
            )
            sanity_checked += 1
        gt_mask = load_index_mask(args.gt_seg_root / item["pan_seg_file_name"], item["segments_info"])
        gt_labels = np.asarray([a["category_id"] for a in item["annotations"]], dtype=int)
        candidate_masks = np.stack([c["winning_mask"] for c in candidates]) if candidates else np.zeros((0, *winner_map.shape), bool)
        mapping = single_mpo_binary_mask_mapping(
            gt_mask, candidate_masks, gt_labels,
            np.asarray([c["label_id"] for c in candidates], dtype=int), args.iou_threshold,
        )
        relations = np.asarray(item["relations"], dtype=int)
        scores = np.asarray([by_query[c["query_id"]]["joint_score"] for c in candidates], dtype=float)
        oracle_indices = gt_set_oracle(mapping, scores, relations, len(native_ids), weights)
        oracle_ids = [candidates[int(i)]["query_id"] for i in oracle_indices]
        native_indices = np.asarray([i for i, c in enumerate(candidates) if c["query_id"] in set(native_ids)], dtype=int)
        native_count = image_counts(len(gt_labels), relations, native_indices, mapping, num_predicates)
        oracle_count = image_counts(len(gt_labels), relations, oracle_indices, mapping, num_predicates)
        full_count = image_counts(len(gt_labels), relations, np.arange(len(candidates)), mapping, num_predicates)
        for row, strategy in ((native_count, "native"), (oracle_count, "actionable_oracle"),
                              (full_count, "raw_candidate_upper_bound")):
            row.update({"image_id": str(item["image_id"]), "file_name": filename,
                        "bootstrap_group": filename, "strategy": strategy,
                        "entity_count": len(native_ids), "candidate_count": len(candidates)})
            if strategy == "native":
                native_rows.append(row)
            elif strategy == "actionable_oracle":
                oracle_rows.append(row)
            else:
                full_rows.append(row)
    native_summary = aggregate_counts(native_rows, num_predicates)
    oracle_summary = aggregate_counts(oracle_rows, num_predicates)
    full_summary = aggregate_counts(full_rows, num_predicates)
    delta = float(oracle_summary["predicate_balanced_endpoint_support"] - native_summary["predicate_balanced_endpoint_support"])
    candidate_ceiling_gap = float(
        full_summary["predicate_balanced_endpoint_support"]
        - native_summary["predicate_balanced_endpoint_support"]
    )
    raw_gap = args.raw_query_gap
    document = {
        "schema_version": 2, "contract": "full-pool pixel competition -> candidate entities -> fixed-K admission",
        "split": args.split, "partition": args.partition, "images": len(items),
        "partition_images": partition_images,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "native": native_summary,
        "actionable_oracle": oracle_summary, "delta_actionable": delta,
        "native_sanity": {"passed": sanity_checked == len(items),
                           "checked_images": sanity_checked},
        "candidate_supply_ceiling": full_summary,
        "candidate_supply_ceiling_gap": candidate_ceiling_gap,
        "raw_upper_bound_gap": raw_gap,
        "actionable_fraction": (delta / raw_gap if raw_gap is not None and raw_gap > 0 else None),
        "gates": {
            "delta_actionable_min": 0.06,
            "delta_actionable_passed": delta >= 0.06,
            "actionable_fraction_min": 0.4,
            "actionable_fraction_passed": (
                delta / raw_gap >= 0.4 if raw_gap is not None and raw_gap > 0 else None
            ),
        },
        "note": (
            "The candidate supply ceiling admits every post-competition candidate and is not "
            "the 200-raw-query fixed-K denominator. Actionable fraction remains null unless a "
            "same-partition raw-query gap is supplied. Official test is excluded."
        ),
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(document, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")
    print(json.dumps(document, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))


if __name__ == "__main__":
    main()
