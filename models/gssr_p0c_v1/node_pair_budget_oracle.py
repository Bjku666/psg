#!/usr/bin/env python3
"""Compute-equivalent entity-set and pair-set oracle comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0_v1.audit import (
    aggregate_counts,
    image_counts,
    resolve_budget,
    score_topk,
    single_mpo_candidate_mapping,
)
from models.gssr_p0b_v1.balanced_oracle import (
    gt_set_oracle,
    inverse_predicate_weights,
    predicate_counts,
)
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_panoptic_mapping
from models.gssr_p0b_v1.run_formal_audit import bootstrap_paired_delta, score_array


def pair_oracle_relation_hits(
    supplied_gt: set[int],
    relations: np.ndarray,
    pair_budget: int,
    predicate_weights: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Choose ordered GT endpoint pairs exactly under an independent pair budget.

    One selected ordered pair supports every annotated predicate instance on
    that pair.  Since pair choices have no coupling constraints, sorting their
    aggregate balanced utilities is the exact solution.
    """
    relations = np.asarray(relations, dtype=np.int64).reshape(-1, 3)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (subject, obj, _predicate) in enumerate(relations):
        if subject == obj:
            continue
        if int(subject) in supplied_gt and int(obj) in supplied_gt:
            groups[(int(subject), int(obj))].append(index)
    utilities = []
    for pair, indices in groups.items():
        utility = sum(float(predicate_weights[int(relations[index, 2])]) for index in indices)
        utilities.append((utility, len(indices), pair, indices))
    utilities.sort(key=lambda row: (-row[0], -row[1], row[2]))
    chosen = utilities[: max(0, int(pair_budget))]
    hits = np.zeros(len(relations), dtype=bool)
    for _utility, _count, _pair, indices in chosen:
        hits[indices] = True
    return hits, len(chosen)


def pair_oracle_image_counts(
    num_gt: int,
    relations: np.ndarray,
    mapping,
    pair_budget: int,
    predicate_weights: np.ndarray,
    num_predicates: int,
) -> dict:
    supplied = set(int(value) for value in mapping.candidate_to_gt if value >= 0)
    hits, selected_pairs = pair_oracle_relation_hits(
        supplied, relations, pair_budget, predicate_weights
    )
    gt_per_predicate = np.zeros(num_predicates, dtype=np.int64)
    hit_per_predicate = np.zeros(num_predicates, dtype=np.int64)
    for index, (subject, obj, predicate) in enumerate(np.asarray(relations).reshape(-1, 3)):
        if subject == obj:
            continue
        gt_per_predicate[int(predicate)] += 1
        if hits[index]:
            hit_per_predicate[int(predicate)] += 1
    return {
        "num_gt_entities": int(num_gt),
        "matched_gt_entities": len(supplied),
        "num_gt_relations": int(gt_per_predicate.sum()),
        "supported_gt_relations": int(hit_per_predicate.sum()),
        "gt_per_predicate": gt_per_predicate,
        "hit_per_predicate": hit_per_predicate,
        "pair_budget": int(pair_budget),
        "selected_distinct_gt_pairs": selected_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--matching", choices=("box", "mask"), default="mask")
    parser.add_argument("--gt-seg-root", type=Path)
    parser.add_argument("--candidate-seg-root", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.matching == "mask" and (args.gt_seg_root is None or args.candidate_seg_root is None):
        raise ValueError("mask matching requires both segmentation roots")
    with args.psg.open() as stream:
        psg = json.load(stream)
    with args.candidates.open() as stream:
        candidates = json.load(stream)
    test_ids = {str(value) for value in psg["test_image_ids"]}
    gt_items = [
        item for item in psg["data"]
        if str(item["image_id"]) in test_ids and item.get("relations")
    ]
    gt_items.sort(key=lambda item: str(item["image_id"]))
    if args.max_images is not None:
        gt_items = gt_items[:args.max_images]
    candidate_groups: dict[str, list[dict]] = defaultdict(list)
    for item in candidates["data"]:
        candidate_groups[str(item.get("file_name"))].append(item)
    duplicates = [name for name, group in candidate_groups.items() if len(group) > 1]
    if duplicates:
        raise ValueError(f"candidate filenames are not unique: {duplicates[:5]}")
    candidate_by_file = {name: group[0] for name, group in candidate_groups.items()}
    num_predicates = len(psg["predicate_classes"])
    frequencies = predicate_counts(
        [np.asarray(item["relations"]) for item in gt_items], num_predicates
    )
    weights = inverse_predicate_weights(frequencies)
    strategies = ("score_topk", "entity_oracle_balanced", "pair_oracle_balanced")
    rows = {(strategy, float(budget)): [] for strategy in strategies for budget in args.budgets}

    args.output.mkdir(parents=True)
    with (args.output / "per_image.jsonl").open("w") as output:
        for gt in gt_items:
            filename = str(gt["file_name"])
            candidate = candidate_by_file.get(filename, {"annotations": []})
            annotations = candidate.get("annotations", [])
            gt_boxes = np.asarray([row["bbox"] for row in gt["annotations"]], dtype=float)
            gt_labels = np.asarray([row["category_id"] for row in gt["annotations"]], dtype=int)
            boxes = np.asarray([row["bbox"] for row in annotations], dtype=float).reshape(-1, 4)
            labels = np.asarray([row["category_id"] for row in annotations], dtype=int)
            scores = score_array(annotations, "panoptic_score")
            relations = np.asarray(gt["relations"], dtype=int).reshape(-1, 3)
            if args.matching == "box" or not annotations:
                mapping = single_mpo_candidate_mapping(
                    gt_boxes, boxes, gt_labels, labels, args.iou_threshold
                )
            else:
                gt_mask = load_index_mask(
                    args.gt_seg_root / gt["pan_seg_file_name"], gt["segments_info"]
                )
                candidate_mask = load_index_mask(
                    args.candidate_seg_root / candidate["pan_seg_file_name"],
                    candidate["segments_info"],
                )
                mapping = single_mpo_panoptic_mapping(
                    gt_mask, candidate_mask, gt_labels, labels, args.iou_threshold
                )
            for budget in args.budgets:
                budget = float(budget)
                k = resolve_budget(len(annotations), budget)
                pair_budget = k * max(0, k - 1)
                selections = {
                    "score_topk": score_topk(scores, k),
                    "entity_oracle_balanced": gt_set_oracle(
                        mapping, scores, relations, k, weights
                    ),
                }
                image_rows = {
                    strategy: image_counts(
                        len(gt_boxes), relations, selected, mapping, num_predicates
                    )
                    for strategy, selected in selections.items()
                }
                image_rows["pair_oracle_balanced"] = pair_oracle_image_counts(
                    len(gt_boxes), relations, mapping, pair_budget, weights, num_predicates
                )
                for strategy, row in image_rows.items():
                    row.update({
                        "image_id": str(gt["image_id"]),
                        "file_name": filename,
                        "bootstrap_group": filename,
                        "strategy": strategy,
                        "budget": budget,
                        "num_candidates": len(annotations),
                        "entity_budget": k,
                        "pair_budget": pair_budget,
                    })
                    rows[(strategy, budget)].append(row)
                    output.write(json.dumps({
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in row.items()
                    }, sort_keys=True) + "\n")

    summaries = []
    comparisons = []
    for strategy in strategies:
        for budget in args.budgets:
            budget = float(budget)
            strategy_rows = rows[(strategy, budget)]
            summary = aggregate_counts(strategy_rows, num_predicates)
            summary.update({
                "strategy": strategy,
                "budget": budget,
                "mean_entity_budget": float(np.mean([row["entity_budget"] for row in strategy_rows])),
                "mean_pair_budget": float(np.mean([row["pair_budget"] for row in strategy_rows])),
            })
            summaries.append(summary)
    for budget in args.budgets:
        budget = float(budget)
        entity_rows = rows[("entity_oracle_balanced", budget)]
        pair_rows = rows[("pair_oracle_balanced", budget)]
        entity = aggregate_counts(entity_rows, num_predicates)
        pair = aggregate_counts(pair_rows, num_predicates)
        comparison = {
            "budget": budget,
            "contrast": "pair_oracle_balanced-entity_oracle_balanced",
            "predicate_balanced_endpoint_support_delta": (
                pair["predicate_balanced_endpoint_support"]
                - entity["predicate_balanced_endpoint_support"]
            ),
            "endpoint_support_micro_delta": (
                pair["endpoint_support_micro"] - entity["endpoint_support_micro"]
            ),
        }
        comparison.update(bootstrap_paired_delta(
            entity_rows,
            pair_rows,
            args.bootstrap_samples,
            args.seed + int(round(1000 * budget)),
        ))
        comparisons.append(comparison)
    document = {
        "schema_version": 1,
        "carrier_scope": "final-panoptic preliminary" if args.matching == "box" else "mask-matched candidate pool",
        "compute_contract": "entity arm: K nodes and K(K-1) pairs; pair arm: M nodes and at most K(K-1) selected ordered pairs",
        "population_policy": {
            "point_estimate": "official OpenPSG scene rows",
            "bootstrap_unit": "physical image grouped by file_name",
        },
        "summaries": summaries,
        "paired_comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
