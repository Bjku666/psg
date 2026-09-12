#!/usr/bin/env python3
"""Run the formal P0B fixed-budget endpoint-support audit.

This runner intentionally differs from frozen P0 in four ways: it exposes
micro and predicate-balanced exact oracles separately, includes the full-pool
ceiling, rejects fake mask-quality fallbacks, and bootstraps physical images
(``file_name`` clusters) while retaining official OpenPSG scene rows in the
point estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0_v1.audit import (
    aggregate_counts,
    diversity_topk,
    gt_nodewise_topk,
    image_counts,
    resolve_budget,
    retained_gt,
    score_topk,
    single_mpo_candidate_mapping,
)
from models.gssr_p0b_v1.balanced_oracle import (
    gt_set_oracle,
    inverse_predicate_weights,
    predicate_counts,
)
from models.gssr_p0b_v1.mask_matcher import (
    load_index_mask,
    single_mpo_panoptic_mapping,
)
from models.gssr_p0b_v1.supply_decomposition import decompose_summaries


ARMS = (
    "score_topk",
    "mask_quality_topk",
    "joint_score_topk",
    "diversity_topk",
    "gt_nodewise",
    "gt_set_oracle_micro",
    "gt_set_oracle_balanced",
)
DEFAULT_ARMS = (
    "score_topk",
    "diversity_topk",
    "gt_nodewise",
    "gt_set_oracle_micro",
    "gt_set_oracle_balanced",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=DEFAULT_ARMS)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--matching", choices=("box", "mask"), default="mask")
    parser.add_argument("--gt-seg-root", type=Path)
    parser.add_argument("--candidate-seg-root", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--criticality-budget", type=float, default=0.5)
    parser.add_argument("--rare-predicate-max-count", type=int, default=20)
    return parser.parse_args()


def score_array(annotations: list[dict], key: str, required: bool = True) -> np.ndarray:
    aliases = {"panoptic_score": ("panoptic_score", "score")}
    keys = aliases.get(key, (key,))
    values = []
    for index, annotation in enumerate(annotations):
        value = next((annotation[name] for name in keys if annotation.get(name) is not None), None)
        if value is None and key == "joint_score":
            if annotation.get("class_score") is not None and annotation.get("mask_quality") is not None:
                value = float(annotation["class_score"]) * float(annotation["mask_quality"])
        if value is None:
            if required:
                raise ValueError(
                    f"candidate {index} lacks required score field {key}; no fallback is allowed"
                )
            value = np.nan
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def select(
    arm: str,
    mapping,
    scores: np.ndarray,
    mask_quality: np.ndarray,
    joint_score: np.ndarray,
    boxes: np.ndarray,
    relations: np.ndarray,
    num_gt: int,
    k: int,
    balanced_weights: np.ndarray,
) -> np.ndarray:
    if arm == "score_topk":
        return score_topk(scores, k)
    if arm == "mask_quality_topk":
        return score_topk(mask_quality, k)
    if arm == "joint_score_topk":
        return score_topk(joint_score, k)
    if arm == "diversity_topk":
        return diversity_topk(scores, boxes, k)
    if arm == "gt_nodewise":
        return gt_nodewise_topk(mapping, scores, relations, num_gt, k)
    if arm == "gt_set_oracle_micro":
        return gt_set_oracle(mapping, scores, relations, k)
    if arm == "gt_set_oracle_balanced":
        return gt_set_oracle(mapping, scores, relations, k, balanced_weights)
    raise ValueError(arm)


def _cluster_totals(rows: list[dict], value_key: str) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["bootstrap_group"])].append(row)
    keys = sorted(groups)
    gt = np.stack([
        np.stack([row["gt_per_predicate"] for row in groups[key]]).sum(axis=0)
        for key in keys
    ])
    values = np.stack([
        np.stack([row[value_key] for row in groups[key]]).sum(axis=0)
        for key in keys
    ])
    return gt, values


def bootstrap_balanced_endpoint(
    rows: list[dict], samples: int, seed: int
) -> tuple[float, float]:
    if samples <= 0 or not rows:
        return float("nan"), float("nan")
    gt, hit = _cluster_totals(rows, "hit_per_predicate")
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        index = rng.integers(0, len(gt), size=len(gt))
        gt_sum, hit_sum = gt[index].sum(axis=0), hit[index].sum(axis=0)
        valid = gt_sum > 0
        values[sample] = np.mean(hit_sum[valid] / gt_sum[valid]) if valid.any() else 0.0
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def bootstrap_paired_delta(
    baseline_rows: list[dict],
    oracle_rows: list[dict],
    samples: int,
    seed: int,
) -> dict:
    baseline_ids = [(row["image_id"], row["bootstrap_group"]) for row in baseline_rows]
    oracle_ids = [(row["image_id"], row["bootstrap_group"]) for row in oracle_rows]
    if baseline_ids != oracle_ids:
        raise ValueError("paired bootstrap requires identically ordered scene rows and groups")
    gt, baseline_hit = _cluster_totals(baseline_rows, "hit_per_predicate")
    oracle_gt, oracle_hit = _cluster_totals(oracle_rows, "hit_per_predicate")
    if not np.array_equal(gt, oracle_gt):
        raise ValueError("paired bootstrap GT denominators differ")
    if samples <= 0:
        return {
            "bootstrap_samples": 0,
            "bootstrap_groups": len(gt),
            "bootstrap_unit": "physical_image_file_name",
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "probability_delta_gt_zero": float("nan"),
        }
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        index = rng.integers(0, len(gt), size=len(gt))
        gt_sum = gt[index].sum(axis=0)
        valid = gt_sum > 0
        base = np.mean(baseline_hit[index].sum(axis=0)[valid] / gt_sum[valid])
        oracle = np.mean(oracle_hit[index].sum(axis=0)[valid] / gt_sum[valid])
        deltas[sample] = oracle - base
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "bootstrap_samples": samples,
        "bootstrap_groups": len(gt),
        "bootstrap_unit": "physical_image_file_name",
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
    }


def criticality_records(
    gt: dict,
    annotations: list[dict],
    mapping,
    scores: np.ndarray,
    relations: np.ndarray,
    selections: dict[str, np.ndarray],
    predicate_frequency: np.ndarray,
    rare_max_count: int,
) -> list[dict]:
    selected_gt = {
        arm: retained_gt(selected, mapping) for arm, selected in selections.items()
    }
    incident: list[list[int]] = [[] for _ in gt["annotations"]]
    for subject, obj, predicate in relations:
        if subject == obj:
            continue
        incident[int(subject)].append(int(predicate))
        incident[int(obj)].append(int(predicate))
    output = []
    for gt_index, annotation in enumerate(gt["annotations"]):
        candidates = np.flatnonzero(mapping.candidate_to_gt == gt_index)
        best_score = float(scores[candidates].max()) if len(candidates) else None
        best_iou = float(mapping.candidate_iou[candidates].max()) if len(candidates) else None
        predicates = sorted(set(incident[gt_index]))
        frequencies = [int(predicate_frequency[predicate]) for predicate in predicates]
        score_selected = gt_index in selected_gt.get("score_topk", set())
        balanced_selected = gt_index in selected_gt.get("gt_set_oracle_balanced", set())
        bbox = annotation.get("bbox", [0, 0, 0, 0])
        bbox_area = (
            max(0.0, float(bbox[2]) - float(bbox[0]))
            * max(0.0, float(bbox[3]) - float(bbox[1]))
        )
        area = (
            int(gt["segments_info"][gt_index].get("area", bbox_area))
            if gt.get("segments_info") and gt_index < len(gt["segments_info"])
            else bbox_area
        )
        output.append({
            "image_id": str(gt["image_id"]),
            "file_name": str(gt.get("file_name")),
            "gt_entity_index": gt_index,
            "category_id": int(annotation["category_id"]),
            "thing_or_stuff": "thing" if int(annotation["category_id"]) < 80 else "stuff",
            "area": area,
            "relation_degree": len(incident[gt_index]),
            "incident_predicates": predicates,
            "min_incident_predicate_frequency": min(frequencies) if frequencies else None,
            "has_rare_predicate": any(value <= rare_max_count for value in frequencies),
            "candidate_available": bool(len(candidates)),
            "best_candidate_score": best_score,
            "best_candidate_iou": best_iou,
            "score_selected": score_selected,
            "micro_oracle_selected": gt_index in selected_gt.get("gt_set_oracle_micro", set()),
            "balanced_oracle_selected": balanced_selected,
            "score_dropped_but_balanced_oracle_selected": balanced_selected and not score_selected,
        })
    return output


def main() -> None:
    args = parse_args()
    if 1.0 not in args.budgets:
        raise ValueError("formal P0B requires budget 1.0 as the full-pool ceiling")
    if args.matching == "mask" and (args.gt_seg_root is None or args.candidate_seg_root is None):
        raise ValueError("mask matching requires --gt-seg-root and --candidate-seg-root")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output}")
    with args.psg.open() as stream:
        psg = json.load(stream)
    with args.candidates.open() as stream:
        candidates = json.load(stream)

    test_ids = {str(value) for value in psg["test_image_ids"]}
    gt_items = []
    for item in psg["data"]:
        is_test = str(item["image_id"]) in test_ids
        if args.split == "test" and not is_test:
            continue
        if args.split == "train" and is_test:
            continue
        if item.get("relations"):
            gt_items.append(item)
    gt_items.sort(key=lambda item: str(item["image_id"]))
    if args.max_images is not None:
        gt_items = gt_items[:args.max_images]

    candidate_groups: dict[str, list[dict]] = defaultdict(list)
    for item in candidates.get("data", []):
        candidate_groups[str(item.get("file_name"))].append(item)
    duplicate_candidate_files = [key for key, value in candidate_groups.items() if len(value) > 1]
    if duplicate_candidate_files:
        raise ValueError(f"candidate filenames are not unique: {duplicate_candidate_files[:5]}")
    candidate_by_file = {key: value[0] for key, value in candidate_groups.items()}
    gt_files = {str(item.get("file_name")) for item in gt_items}
    if candidates.get("data") and not gt_files.intersection(candidate_by_file):
        raise ValueError("candidate/GT filename overlap is empty")

    num_predicates = len(psg["predicate_classes"])
    predicate_frequency = predicate_counts(
        [np.asarray(item["relations"], dtype=np.int64) for item in gt_items],
        num_predicates,
    )
    balanced_weights = inverse_predicate_weights(predicate_frequency)
    needs_quality = "mask_quality_topk" in args.arms
    needs_joint = "joint_score_topk" in args.arms
    raw_rows = {
        (arm, float(budget)): [] for arm in args.arms for budget in args.budgets
    }
    candidate_counts: list[int] = []
    missing_candidate_files: list[str] = []
    dimension_mismatches: list[dict] = []
    criticality: list[dict] = []

    args.output.mkdir(parents=True)
    with (args.output / "per_image.jsonl").open("w") as per_image_stream:
        for gt in gt_items:
            filename = str(gt.get("file_name"))
            candidate = candidate_by_file.get(filename, {"annotations": []})
            if not candidate.get("annotations"):
                missing_candidate_files.append(filename)
            annotations = candidate.get("annotations", [])
            if annotations and (
                int(candidate.get("width", -1)) != int(gt.get("width", -2))
                or int(candidate.get("height", -1)) != int(gt.get("height", -2))
            ):
                dimension_mismatches.append({
                    "file_name": filename,
                    "gt": [gt.get("width"), gt.get("height")],
                    "candidate": [candidate.get("width"), candidate.get("height")],
                })
            gt_boxes = np.asarray([row["bbox"] for row in gt["annotations"]], dtype=np.float64)
            gt_labels = np.asarray([row["category_id"] for row in gt["annotations"]], dtype=np.int64)
            boxes = np.asarray([row["bbox"] for row in annotations], dtype=np.float64).reshape(-1, 4)
            labels = np.asarray([row["category_id"] for row in annotations], dtype=np.int64)
            scores = score_array(annotations, "panoptic_score")
            quality = score_array(annotations, "mask_quality", required=needs_quality)
            joint = score_array(annotations, "joint_score", required=needs_joint)
            relations = np.asarray(gt["relations"], dtype=np.int64).reshape(-1, 3)
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
            candidate_counts.append(len(annotations))
            primary_selections: dict[str, np.ndarray] = {}
            for budget in args.budgets:
                k = resolve_budget(len(annotations), float(budget))
                for arm in args.arms:
                    selected = select(
                        arm, mapping, scores, quality, joint, boxes, relations,
                        len(gt_boxes), k, balanced_weights,
                    )
                    counts = image_counts(
                        len(gt_boxes), relations, selected, mapping, num_predicates
                    )
                    counts.update({
                        "image_id": str(gt["image_id"]),
                        "file_name": filename,
                        "bootstrap_group": filename,
                        "arm": arm,
                        "budget": float(budget),
                        "num_candidates": len(annotations),
                        "selected_nodes": k,
                        "ordered_pairs": k * max(0, k - 1),
                    })
                    raw_rows[(arm, float(budget))].append(counts)
                    serializable = {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in counts.items()
                    }
                    per_image_stream.write(json.dumps(serializable, sort_keys=True) + "\n")
                    if np.isclose(float(budget), args.criticality_budget):
                        primary_selections[arm] = selected
            if primary_selections:
                criticality.extend(criticality_records(
                    gt, annotations, mapping, scores, relations, primary_selections,
                    predicate_frequency, args.rare_predicate_max_count,
                ))

    summaries = []
    for (arm, budget), rows in raw_rows.items():
        summary = aggregate_counts(rows, num_predicates)
        summary.update({
            "arm": arm,
            "budget": budget,
            "scene_rows": len(rows),
            "bootstrap_groups": len({row["bootstrap_group"] for row in rows}),
            "mean_candidates": float(np.mean([row["num_candidates"] for row in rows])) if rows else 0.0,
            "mean_selected_nodes": float(np.mean([row["selected_nodes"] for row in rows])) if rows else 0.0,
            "mean_ordered_pairs": float(np.mean([row["ordered_pairs"] for row in rows])) if rows else 0.0,
        })
        low, high = bootstrap_balanced_endpoint(
            rows, args.bootstrap_samples, args.seed + int(round(1000 * budget))
        )
        summary["predicate_balanced_endpoint_support_ci95"] = [low, high]
        summaries.append(summary)

    comparisons = []
    for oracle_arm in ("gt_set_oracle_micro", "gt_set_oracle_balanced"):
        if "score_topk" not in args.arms or oracle_arm not in args.arms:
            continue
        for budget in args.budgets:
            base_rows = raw_rows[("score_topk", float(budget))]
            oracle_rows = raw_rows[(oracle_arm, float(budget))]
            base = aggregate_counts(base_rows, num_predicates)
            oracle = aggregate_counts(oracle_rows, num_predicates)
            comparison = {
                "budget": float(budget),
                "contrast": f"{oracle_arm}-score_topk",
                "predicate_balanced_endpoint_support_delta": (
                    oracle["predicate_balanced_endpoint_support"]
                    - base["predicate_balanced_endpoint_support"]
                ),
                "endpoint_support_micro_delta": (
                    oracle["endpoint_support_micro"] - base["endpoint_support_micro"]
                ),
            }
            comparison.update(bootstrap_paired_delta(
                base_rows,
                oracle_rows,
                args.bootstrap_samples,
                args.seed + 10000 + int(round(1000 * float(budget))),
            ))
            comparisons.append(comparison)

    counts = np.asarray(candidate_counts, dtype=np.float64)
    summary_doc = {
        "schema_version": 2,
        "population_policy": {
            "point_estimate": "official OpenPSG scene rows",
            "bootstrap_unit": "physical image grouped by file_name",
        },
        "predicate_frequency": predicate_frequency.tolist(),
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "supply_decomposition": decompose_summaries(summaries)
        if "score_topk" in args.arms and "gt_set_oracle_balanced" in args.arms else [],
        "candidate_count_distribution": {
            "scene_rows": len(counts),
            "physical_images": len(gt_files),
            "missing_candidate_scene_rows": int((counts == 0).sum()),
            "min": int(counts.min()) if len(counts) else 0,
            "q25": float(np.quantile(counts, 0.25)) if len(counts) else 0.0,
            "median": float(np.median(counts)) if len(counts) else 0.0,
            "q75": float(np.quantile(counts, 0.75)) if len(counts) else 0.0,
            "max": int(counts.max()) if len(counts) else 0,
        },
        "data_contract": {
            "gt_scene_rows": len(gt_items),
            "gt_unique_files": len(gt_files),
            "duplicate_scene_rows": len(gt_items) - len(gt_files),
            "candidate_files": len(candidate_by_file),
            "filename_overlap": len(gt_files.intersection(candidate_by_file)),
            "missing_candidate_files": sorted(set(missing_candidate_files)),
            "dimension_mismatches": dimension_mismatches,
            "mask_quality_field_available": all(
                annotation.get("mask_quality") is not None
                for item in candidates.get("data", [])
                for annotation in item.get("annotations", [])
            ),
            "mask_quality_arm_run": "mask_quality_topk" in args.arms,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary_doc, indent=2) + "\n")
    with (args.output / "entity_criticality.jsonl").open("w") as stream:
        for row in criticality:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "seed": args.seed,
        "matching": args.matching,
        "iou_threshold": args.iou_threshold,
        "budget_rule": "K=max(1,ceil(ratio*M)); K=0 when M=0",
        "oracle_objectives": {
            "gt_set_oracle_micro": "unit weight per relation instance",
            "gt_set_oracle_balanced": "1 / split predicate frequency per relation instance",
        },
        "bootstrap_unit": "physical_image_file_name",
        "python": sys.version,
        "platform": platform.platform(),
        "psg": {"path": str(args.psg.resolve()), "sha256": sha256(args.psg)},
        "candidates": {"path": str(args.candidates.resolve()), "sha256": sha256(args.candidates)},
        "gt_seg_root": str(args.gt_seg_root.resolve()) if args.gt_seg_root else None,
        "candidate_seg_root": str(args.candidate_seg_root.resolve()) if args.candidate_seg_root else None,
        "fair_psg_revision": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2] / "third_party/fair_psg"), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES",)
            if os.environ.get(key)
        },
    }
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(summary_doc, indent=2))


if __name__ == "__main__":
    main()
