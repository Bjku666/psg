#!/usr/bin/env python3
"""Merge independently evaluated P1A shards into one partition result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge_metric(summaries: list[dict]) -> dict:
    images = sum(int(summary["images"]) for summary in summaries)
    gt_entities = sum(int(summary["gt_entities"]) for summary in summaries)
    matched_entities = sum(int(summary["matched_gt_entities"]) for summary in summaries)
    gt_per_predicate = np.sum(
        [np.asarray(summary["gt_per_predicate"], dtype=np.int64) for summary in summaries], axis=0
    )
    hit_per_predicate = np.sum(
        [np.asarray(summary["hit_per_predicate"], dtype=np.int64) for summary in summaries], axis=0
    )
    gt_relations = int(gt_per_predicate.sum())
    supported_relations = int(hit_per_predicate.sum())
    valid = gt_per_predicate > 0
    per_predicate = np.divide(
        hit_per_predicate, gt_per_predicate,
        out=np.zeros_like(hit_per_predicate, dtype=np.float64), where=valid,
    )
    return {
        "images": images,
        "entity_recall_micro": matched_entities / gt_entities if gt_entities else 0.0,
        "endpoint_support_micro": supported_relations / gt_relations if gt_relations else 0.0,
        "endpoint_support_macro_image": (
            sum(float(summary["endpoint_support_macro_image"]) * int(summary["images"])
                for summary in summaries) / images if images else 0.0
        ),
        "predicate_balanced_endpoint_support": float(per_predicate[valid].mean()) if valid.any() else 0.0,
        "gt_entities": gt_entities,
        "matched_gt_entities": matched_entities,
        "gt_relations": gt_relations,
        "supported_gt_relations": supported_relations,
        "gt_per_predicate": gt_per_predicate.tolist(),
        "hit_per_predicate": hit_per_predicate.tolist(),
        "per_predicate_endpoint_support": per_predicate.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-query-gap", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = [json.loads((path / "summary.json").read_text()) for path in args.inputs]
    expected_count = len(documents)
    indices = sorted(int(document["shard"]["index"]) for document in documents)
    if indices != list(range(expected_count)) or {
        int(document["shard"]["count"]) for document in documents
    } != {expected_count}:
        raise ValueError("inputs do not form one complete shard set")
    invariant_keys = ("contract", "split", "partition", "partition_images")
    for key in invariant_keys:
        if len({json.dumps(document[key], sort_keys=True) for document in documents}) != 1:
            raise ValueError(f"shards disagree on {key}")
    native = merge_metric([document["native"] for document in documents])
    oracle = merge_metric([document["actionable_oracle"] for document in documents])
    ceiling = merge_metric([document["candidate_supply_ceiling"] for document in documents])
    delta = float(
        oracle["predicate_balanced_endpoint_support"]
        - native["predicate_balanced_endpoint_support"]
    )
    ceiling_gap = float(
        ceiling["predicate_balanced_endpoint_support"]
        - native["predicate_balanced_endpoint_support"]
    )
    raw_gap = args.raw_query_gap
    actionable_fraction = delta / raw_gap if raw_gap is not None and raw_gap > 0 else None
    checked_images = sum(int(document["native_sanity"]["checked_images"]) for document in documents)
    partition_images = int(documents[0]["partition_images"])
    output = {
        "schema_version": 2,
        "contract": documents[0]["contract"],
        "split": documents[0]["split"],
        "partition": documents[0]["partition"],
        "images": native["images"],
        "partition_images": partition_images,
        "native": native,
        "actionable_oracle": oracle,
        "delta_actionable": delta,
        "native_sanity": {
            "passed": checked_images == partition_images == native["images"],
            "checked_images": checked_images,
        },
        "candidate_supply_ceiling": ceiling,
        "candidate_supply_ceiling_gap": ceiling_gap,
        "raw_upper_bound_gap": raw_gap,
        "actionable_fraction": actionable_fraction,
        "gates": {
            "delta_actionable_min": 0.06,
            "delta_actionable_passed": delta >= 0.06,
            "actionable_fraction_min": 0.4,
            "actionable_fraction_passed": (
                actionable_fraction >= 0.4 if actionable_fraction is not None else None
            ),
            "p1b_authorized": bool(
                delta >= 0.06 and actionable_fraction is not None and actionable_fraction >= 0.4
            ),
        },
        "source_shards": [str(path.resolve()) for path in args.inputs],
        "note": (
            "The candidate supply ceiling is not the 200-raw-query fixed-K denominator. "
            "Official test is excluded and confirm was not opened."
        ),
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
