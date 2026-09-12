#!/usr/bin/env python3
"""Run fixed-budget entity admission diagnostics on PSG-style JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .audit import (
        aggregate_counts, diversity_topk, gt_nodewise_topk, gt_set_oracle,
        image_counts, resolve_budget, score_topk, single_mpo_candidate_mapping,
        single_mpo_mask_mapping,
    )
except ImportError:  # direct script entry point
    from audit import (
        aggregate_counts, diversity_topk, gt_nodewise_topk, gt_set_oracle,
        image_counts, resolve_budget, score_topk, single_mpo_candidate_mapping,
        single_mpo_mask_mapping,
    )


ARMS = ("score_topk", "mask_quality_topk", "diversity_topk", "gt_nodewise", "gt_set_oracle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--matching", choices=("box", "mask"), default="box")
    parser.add_argument("--gt-seg-root", type=Path)
    parser.add_argument("--candidate-seg-root", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def candidate_score(annotation: dict, key: str = "score") -> float:
    value = annotation.get(key)
    if value is None and key == "mask_quality":
        value = annotation.get("score")
    return float(value if value is not None else 0.0)


def rgb2id(color: np.ndarray) -> np.ndarray:
    color = color.astype(np.int64, copy=False)
    return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]


def load_index_mask(path: Path, segments_info: list[dict]) -> np.ndarray:
    ids = rgb2id(np.asarray(Image.open(path).convert("RGB")))
    indexed = np.full(ids.shape, -1, dtype=np.int32)
    for idx, info in enumerate(segments_info):
        indexed[ids == int(info["id"])] = idx
    return indexed


def select(arm, mapping, scores, qualities, boxes, relations, num_gt, k):
    if arm == "score_topk":
        return score_topk(scores, k)
    if arm == "mask_quality_topk":
        return score_topk(qualities, k)
    if arm == "diversity_topk":
        return diversity_topk(scores, boxes, k)
    if arm == "gt_nodewise":
        return gt_nodewise_topk(mapping, scores, relations, num_gt, k)
    if arm == "gt_set_oracle":
        return gt_set_oracle(mapping, scores, relations, k)
    raise ValueError(arm)


def bootstrap_balanced_endpoint(rows: list[dict], samples: int, seed: int) -> tuple[float, float]:
    if samples <= 0 or not rows:
        return float("nan"), float("nan")
    gt = np.stack([r["gt_per_predicate"] for r in rows])
    hit = np.stack([r["hit_per_predicate"] for r in rows])
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for sample in range(samples):
        idx = rng.integers(0, len(rows), size=len(rows))
        gt_sum, hit_sum = gt[idx].sum(0), hit[idx].sum(0)
        valid = gt_sum > 0
        values[sample] = np.mean(hit_sum[valid] / gt_sum[valid]) if valid.any() else 0.0
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def bootstrap_paired_delta(
    rows_a: list[dict], rows_b: list[dict], samples: int, seed: int
) -> dict:
    if len(rows_a) != len(rows_b):
        raise ValueError("paired bootstrap requires aligned image populations")
    gt = np.stack([r["gt_per_predicate"] for r in rows_a])
    hit_a = np.stack([r["hit_per_predicate"] for r in rows_a])
    hit_b = np.stack([r["hit_per_predicate"] for r in rows_b])
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for sample in range(samples):
        idx = rng.integers(0, len(rows_a), size=len(rows_a))
        gt_sum = gt[idx].sum(0)
        valid = gt_sum > 0
        score_a = np.mean(hit_a[idx].sum(0)[valid] / gt_sum[valid]) if valid.any() else 0.0
        score_b = np.mean(hit_b[idx].sum(0)[valid] / gt_sum[valid]) if valid.any() else 0.0
        deltas[sample] = score_b - score_a
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "bootstrap_samples": samples,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
    }


def main() -> None:
    args = parse_args()
    if args.matching == "mask" and (args.gt_seg_root is None or args.candidate_seg_root is None):
        raise ValueError("mask matching requires --gt-seg-root and --candidate-seg-root")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output}")
    args.output.mkdir(parents=True)
    with args.psg.open() as handle:
        psg = json.load(handle)
    with args.candidates.open() as handle:
        candidates = json.load(handle)

    test_ids = {str(x) for x in psg["test_image_ids"]}
    gt_items = []
    for item in psg["data"]:
        is_test = str(item["image_id"]) in test_ids
        if args.split == "test" and not is_test:
            continue
        if args.split == "train" and is_test:
            continue
        if item.get("relations"):
            gt_items.append(item)
    gt_items.sort(key=lambda x: str(x["image_id"]))
    if args.max_images is not None:
        gt_items = gt_items[: args.max_images]
    cand_by_id = {str(x["image_id"]): x for x in candidates["data"]}
    cand_by_file = {str(x["file_name"]): x for x in candidates["data"] if x.get("file_name")}
    # Raw Fair PSG exports use COCO filename stems as image_id, while OpenPSG
    # scene annotations use a separate image_id namespace.  Require filename
    # overlap for the declared raw-export contract and record any ID fallback.
    gt_files = {str(x.get("file_name")) for x in gt_items if x.get("file_name")}
    cand_files = set(cand_by_file)
    filename_overlap = gt_files & cand_files
    if candidates.get("data") and not filename_overlap:
        raise ValueError("candidate/GT filename overlap is empty; refusing ambiguous ID-only join")
    fallback_id_joins = 0
    missing_candidate_files: list[str] = []
    dimension_mismatches: list[dict] = []
    category_values: set[int] = set()
    num_predicates = len(psg["predicate_classes"])

    raw_rows: dict[tuple[str, float], list[dict]] = {
        (arm, budget): [] for arm in args.arms for budget in args.budgets
    }
    candidate_counts = []
    per_image_path = args.output / "per_image.jsonl"
    with per_image_path.open("w") as stream:
        for gt in gt_items:
            # Fair PSG's segmentation exporter uses the COCO filename stem as
            # image_id, whereas OpenPSG uses a separate scene id.  Filename is
            # therefore the authoritative join key for raw exporter output.
            gt_file = str(gt.get("file_name"))
            cand = cand_by_file.get(gt_file)
            if cand is None:
                cand = cand_by_id.get(str(gt["image_id"]), {"annotations": []})
                if cand.get("annotations"):
                    fallback_id_joins += 1
                else:
                    missing_candidate_files.append(gt_file)
            anns = cand.get("annotations", [])
            if anns and (int(cand.get("width", -1)) != int(gt.get("width", -2)) or
                         int(cand.get("height", -1)) != int(gt.get("height", -2))):
                dimension_mismatches.append({"file_name": gt_file,
                                             "gt": [gt.get("width"), gt.get("height")],
                                             "candidate": [cand.get("width"), cand.get("height")]})
            gt_boxes = np.asarray([x["bbox"] for x in gt["annotations"]], dtype=float)
            gt_labels = np.asarray([x["category_id"] for x in gt["annotations"]], dtype=int)
            boxes = np.asarray([x["bbox"] for x in anns], dtype=float).reshape(-1, 4)
            labels = np.asarray([x["category_id"] for x in anns], dtype=int)
            category_values.update(int(x) for x in gt_labels)
            category_values.update(int(x) for x in labels)
            scores = np.asarray([candidate_score(x) for x in anns], dtype=float)
            qualities = np.asarray([candidate_score(x, "mask_quality") for x in anns], dtype=float)
            relations = np.asarray(gt["relations"], dtype=int).reshape(-1, 3)
            if not anns:
                mapping = single_mpo_candidate_mapping(
                    gt_boxes, boxes, gt_labels, labels, args.iou_threshold
                )
            elif args.matching == "box":
                mapping = single_mpo_candidate_mapping(
                    gt_boxes, boxes, gt_labels, labels, args.iou_threshold
                )
            else:
                gt_mask = load_index_mask(args.gt_seg_root / gt["pan_seg_file_name"], gt["segments_info"])
                cand_mask = load_index_mask(
                    args.candidate_seg_root / cand["pan_seg_file_name"], cand["segments_info"]
                )
                mapping = single_mpo_mask_mapping(
                    gt_mask, cand_mask, gt_labels, labels, args.iou_threshold
                )
            candidate_counts.append(len(anns))
            for budget in args.budgets:
                k = resolve_budget(len(anns), budget)
                for arm in args.arms:
                    selected = select(arm, mapping, scores, qualities, boxes, relations, len(gt_boxes), k)
                    counts = image_counts(len(gt_boxes), relations, selected, mapping, num_predicates)
                    counts.update({
                        "image_id": str(gt["image_id"]), "arm": arm, "budget": budget,
                        "num_candidates": len(anns), "selected_nodes": k,
                        "ordered_pairs": k * max(0, k - 1),
                    })
                    raw_rows[(arm, budget)].append(counts)
                    serializable = {k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in counts.items()}
                    stream.write(json.dumps(serializable, sort_keys=True) + "\n")

    summaries = []
    for (arm, budget), rows in raw_rows.items():
        summary = aggregate_counts(rows, num_predicates)
        summary.update({
            "arm": arm, "budget": budget,
            "mean_candidates": float(np.mean([r["num_candidates"] for r in rows])) if rows else 0,
            "mean_selected_nodes": float(np.mean([r["selected_nodes"] for r in rows])) if rows else 0,
            "mean_ordered_pairs": float(np.mean([r["ordered_pairs"] for r in rows])) if rows else 0,
        })
        ci_low, ci_high = bootstrap_balanced_endpoint(
            rows, args.bootstrap_samples, args.seed + int(round(1000 * budget))
        )
        summary["predicate_balanced_endpoint_support_ci95"] = [ci_low, ci_high]
        summaries.append(summary)

    counts = np.asarray(candidate_counts, dtype=float)
    comparisons = []
    if "score_topk" in args.arms and "gt_set_oracle" in args.arms:
        for budget in args.budgets:
            base = aggregate_counts(raw_rows[("score_topk", budget)], num_predicates)
            oracle = aggregate_counts(raw_rows[("gt_set_oracle", budget)], num_predicates)
            comparison = {
                "budget": budget,
                "contrast": "gt_set_oracle-score_topk",
                "predicate_balanced_endpoint_support_delta": (
                    oracle["predicate_balanced_endpoint_support"]
                    - base["predicate_balanced_endpoint_support"]
                ),
            }
            comparison.update(bootstrap_paired_delta(
                raw_rows[("score_topk", budget)],
                raw_rows[("gt_set_oracle", budget)],
                args.bootstrap_samples,
                args.seed + 10000 + int(round(1000 * budget)),
            ))
            comparisons.append(comparison)
    summary_doc = {
        "schema_version": 1,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "candidate_count_distribution": {
            "images": len(counts),
            "missing_candidate_images": int((counts == 0).sum()),
            "min": int(counts.min()) if len(counts) else 0,
            "q25": float(np.quantile(counts, 0.25)) if len(counts) else 0,
            "median": float(np.median(counts)) if len(counts) else 0,
            "q75": float(np.quantile(counts, 0.75)) if len(counts) else 0,
            "max": int(counts.max()) if len(counts) else 0,
        },
        "data_contract": {
            "gt_files": len(gt_files),
            "candidate_files": len(cand_files),
            "filename_overlap": len(filename_overlap),
            "fallback_id_joins": fallback_id_joins,
            "missing_candidate_files": missing_candidate_files,
            "dimension_mismatches": dimension_mismatches,
            "category_id_min": min(category_values) if category_values else None,
            "category_id_max": max(category_values) if category_values else None,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary_doc, indent=2) + "\n")
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "seed": args.seed,
        "matching": args.matching,
        "python": sys.version,
        "platform": platform.platform(),
        "psg": {"path": str(args.psg.resolve()), "sha256": sha256(args.psg)},
        "candidates": {"path": str(args.candidates.resolve()), "sha256": sha256(args.candidates)},
        "fair_psg_revision": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2] / "third_party/fair_psg"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES",) if os.environ.get(k)},
    }
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(summary_doc, indent=2))


if __name__ == "__main__":
    main()
