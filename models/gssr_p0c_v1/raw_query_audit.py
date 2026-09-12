#!/usr/bin/env python3
"""Audit raw-query supply and native Mask2Former entity admission."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0_v1.audit import aggregate_counts, image_counts, resolve_budget, retained_gt, score_topk
from models.gssr_p0b_v1.balanced_oracle import gt_set_oracle, inverse_predicate_weights, predicate_counts
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping
from models.gssr_p0b_v1.run_formal_audit import bootstrap_paired_delta
from models.gssr_p0c_v1.raw_query_schema import decode_binary_mask, validate_image_record
from models.gssr_p0c_v1.node_pair_budget_oracle import pair_oracle_image_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def failure_decomposition(
    relations: np.ndarray,
    supplied: set[int],
    admitted: set[int],
    num_predicates: int,
) -> dict[str, np.ndarray]:
    output = {
        "supply_failure": np.zeros(num_predicates, dtype=np.int64),
        "entity_admission_failure": np.zeros(num_predicates, dtype=np.int64),
        "survives_entity_admission": np.zeros(num_predicates, dtype=np.int64),
    }
    for subject, obj, predicate in np.asarray(relations).reshape(-1, 3):
        if subject == obj:
            continue
        predicate = int(predicate)
        if int(subject) not in supplied or int(obj) not in supplied:
            output["supply_failure"][predicate] += 1
        elif int(subject) not in admitted or int(obj) not in admitted:
            output["entity_admission_failure"][predicate] += 1
        else:
            output["survives_entity_admission"][predicate] += 1
    return output


def main() -> None:
    args = parse_args()
    if 1.0 not in args.budgets:
        raise ValueError("raw-query audit requires budget 1.0 for the supply ceiling")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    with args.psg.open() as stream:
        psg = json.load(stream)
    records = {}
    with args.manifest.open() as stream:
        for line in stream:
            record = json.loads(line)
            validate_image_record(record)
            filename = str(record["file_name"])
            if filename in records:
                raise ValueError(f"duplicate manifest filename: {filename}")
            records[filename] = record

    test_ids = {str(value) for value in psg["test_image_ids"]}
    gt_items = [
        item for item in psg["data"]
        if str(item["image_id"]) in test_ids and item.get("relations")
    ]
    gt_items.sort(key=lambda item: str(item["image_id"]))
    if args.max_images is not None:
        gt_items = gt_items[:args.max_images]
    num_predicates = len(psg["predicate_classes"])
    frequencies = predicate_counts([np.asarray(item["relations"]) for item in gt_items], num_predicates)
    weights = inverse_predicate_weights(frequencies)
    fixed_strategies = (
        "class_score_topk", "mask_quality_topk", "joint_score_topk", "gt_set_oracle_balanced"
    )
    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    stage_counts = {
        name: np.zeros(num_predicates, dtype=np.int64)
        for name in ("supply_failure", "entity_admission_failure", "survives_entity_admission")
    }

    args.output.mkdir(parents=True)
    started_at = time.monotonic()
    with (args.output / "per_image.jsonl").open("w") as output:
        for image_number, gt in enumerate(gt_items, start=1):
            filename = str(gt["file_name"])
            if filename not in records:
                raise ValueError(f"raw-query manifest lacks {filename}")
            record = records[filename]
            queries = record["queries"]
            # Decode one physical image at a time. Caching the full test split
            # would require well over 100 GB because raw query masks overlap.
            candidate_masks = np.stack([
                decode_binary_mask(query["mask_rle"]) for query in queries
            ]) if queries else np.zeros(
                (0, int(record["height"]), int(record["width"])), dtype=bool
            )
            gt_mask = load_index_mask(
                args.gt_seg_root / gt["pan_seg_file_name"], gt["segments_info"]
            )
            gt_labels = np.asarray([row["category_id"] for row in gt["annotations"]], dtype=int)
            candidate_labels = np.asarray([row["predicted_class"] for row in queries], dtype=int)
            mapping = single_mpo_binary_mask_mapping(
                gt_mask, candidate_masks, gt_labels, candidate_labels, args.iou_threshold
            )
            relations = np.asarray(gt["relations"], dtype=int).reshape(-1, 3)
            class_scores = np.asarray([row["class_score"] for row in queries], dtype=float)
            mask_quality = np.asarray([row["mask_quality"] for row in queries], dtype=float)
            joint_scores = np.asarray([row["joint_score"] for row in queries], dtype=float)
            native = np.asarray([
                index for index, row in enumerate(queries) if row["official_keep_flag"]
            ], dtype=int)
            native_oracle = gt_set_oracle(mapping, joint_scores, relations, len(native), weights)
            strategy_selections = {
                "native_admission": native,
                "gt_set_oracle_balanced_at_native_k": native_oracle,
            }
            for strategy, selected in strategy_selections.items():
                count = image_counts(len(gt_labels), relations, selected, mapping, num_predicates)
                count.update({
                    "image_id": str(gt["image_id"]), "file_name": filename,
                    "bootstrap_group": filename, "strategy": strategy, "budget": "native",
                    "num_raw_queries": len(queries), "selected_nodes": len(selected),
                })
                rows[(strategy, "native")].append(count)
                output.write(json.dumps({
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in count.items()
                }, sort_keys=True) + "\n")
            native_pair_budget = len(native) * max(0, len(native) - 1)
            pair_count = pair_oracle_image_counts(
                len(gt_labels), relations, mapping, native_pair_budget, weights, num_predicates
            )
            pair_count.update({
                "image_id": str(gt["image_id"]), "file_name": filename,
                "bootstrap_group": filename,
                "strategy": "pair_oracle_balanced_at_native_pair_budget",
                "budget": "native",
                "num_raw_queries": len(queries),
                "selected_nodes": len(queries),
            })
            rows[("pair_oracle_balanced_at_native_pair_budget", "native")].append(pair_count)
            output.write(json.dumps({
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in pair_count.items()
            }, sort_keys=True) + "\n")
            for budget in args.budgets:
                budget_key = str(float(budget))
                k = resolve_budget(len(queries), float(budget))
                fixed = {
                    "class_score_topk": score_topk(class_scores, k),
                    "mask_quality_topk": score_topk(mask_quality, k),
                    "joint_score_topk": score_topk(joint_scores, k),
                    "gt_set_oracle_balanced": gt_set_oracle(mapping, joint_scores, relations, k, weights),
                }
                for strategy in fixed_strategies:
                    selected = fixed[strategy]
                    count = image_counts(len(gt_labels), relations, selected, mapping, num_predicates)
                    count.update({
                        "image_id": str(gt["image_id"]), "file_name": filename,
                        "bootstrap_group": filename, "strategy": strategy, "budget": float(budget),
                        "num_raw_queries": len(queries), "selected_nodes": len(selected),
                    })
                    rows[(strategy, budget_key)].append(count)
                    output.write(json.dumps({
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in count.items()
                    }, sort_keys=True) + "\n")
            supplied_gt = set(int(value) for value in mapping.candidate_to_gt if value >= 0)
            admitted_gt = retained_gt(native, mapping)
            stages = failure_decomposition(relations, supplied_gt, admitted_gt, num_predicates)
            for name, values in stages.items():
                stage_counts[name] += values
            if image_number % args.progress_every == 0 or image_number == len(gt_items):
                elapsed = time.monotonic() - started_at
                seconds_per_image = elapsed / image_number
                eta = seconds_per_image * (len(gt_items) - image_number)
                print(
                    f"progress={image_number}/{len(gt_items)} "
                    f"seconds_per_image={seconds_per_image:.2f} eta_seconds={eta:.0f}",
                    flush=True,
                )

    summaries = []
    for (strategy, budget), strategy_rows in rows.items():
        summary = aggregate_counts(strategy_rows, num_predicates)
        summary.update({
            "strategy": strategy,
            "budget": budget if budget == "native" else float(budget),
            "mean_raw_queries": float(np.mean([row["num_raw_queries"] for row in strategy_rows])),
            "mean_selected_nodes": float(np.mean([row["selected_nodes"] for row in strategy_rows])),
        })
        summaries.append(summary)
    comparisons = []
    comparison_pairs = [("native_admission", "gt_set_oracle_balanced_at_native_k", "native")]
    comparison_pairs.append((
        "gt_set_oracle_balanced_at_native_k",
        "pair_oracle_balanced_at_native_pair_budget",
        "native",
    ))
    comparison_pairs.extend(
        ("joint_score_topk", "gt_set_oracle_balanced", str(float(budget)))
        for budget in args.budgets
    )
    for baseline, oracle, budget in comparison_pairs:
        baseline_rows = rows[(baseline, budget)]
        oracle_rows = rows[(oracle, budget)]
        baseline_summary = aggregate_counts(baseline_rows, num_predicates)
        oracle_summary = aggregate_counts(oracle_rows, num_predicates)
        comparison = {
            "budget": budget if budget == "native" else float(budget),
            "contrast": f"{oracle}-{baseline}",
            "predicate_balanced_endpoint_support_delta": (
                oracle_summary["predicate_balanced_endpoint_support"]
                - baseline_summary["predicate_balanced_endpoint_support"]
            ),
        }
        comparison.update(bootstrap_paired_delta(
            baseline_rows, oracle_rows, args.bootstrap_samples,
            args.seed + (5000 if budget == "native" else int(1000 * float(budget))),
        ))
        comparisons.append(comparison)
    denominators = sum(stage_counts.values())
    valid = frequencies > 0
    stages = {}
    for name, counts in stage_counts.items():
        stages[name] = {
            "relations": int(counts.sum()),
            "micro_fraction": float(counts.sum() / denominators.sum()) if denominators.sum() else 0.0,
            "predicate_balanced_fraction": float(np.mean(counts[valid] / frequencies[valid])) if valid.any() else 0.0,
            "per_predicate": counts.tolist(),
        }
    document = {
        "schema_version": 1,
        "matching": "class-compatible mask IoU > threshold",
        "population_policy": {
            "point_estimate": "official OpenPSG scene rows",
            "bootstrap_unit": "physical image grouped by file_name",
        },
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "native_failure_decomposition": stages,
        "decomposition_scope": "Supply + EntityAdmission only; PairAdmission and Predicate require a frozen relation carrier",
    }
    (args.output / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
