#!/usr/bin/env python3
"""GT-guided minimum-change fixed-K rescue oracle.

This is a qualification oracle, not a deployable selector.  For each
relation-critical semantic query it transfers the lowest-margin pixels to the
query, swaps it into the native K-set, and reports the best graph-support gain
under changed-pixel budgets.  The implementation intentionally exposes the
visual cost rather than hiding it in a learned score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0_v1.audit import image_counts
from models.gssr_p0b_v1.balanced_oracle import inverse_predicate_weights, predicate_counts
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping
from models.gssr_p0c_v1.native_admission import pre_admission_state, upsampled_query_probabilities
from models.gssr_p0c_v1.raw_query_schema import validate_image_record


def minimum_change_curve(
    winner_map: np.ndarray,
    weighted_scores: np.ndarray,
    semantic_masks: np.ndarray,
    semantic_ids: np.ndarray,
    semantic_labels: np.ndarray,
    native_ids: np.ndarray,
    native_labels: np.ndarray,
    gt_mask: np.ndarray,
    gt_labels: np.ndarray,
    relations: np.ndarray,
    predicate_weights: np.ndarray,
    epsilons: tuple[float, ...],
    iou_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
) -> list[dict]:
    """Evaluate a small exhaustive swap oracle for one image."""
    winner_map = np.asarray(winner_map, dtype=np.int32)
    native_ids = np.asarray(native_ids, dtype=int)
    native_labels = np.asarray(native_labels, dtype=int)
    native_masks = np.stack([winner_map == query_id for query_id in native_ids]) if len(native_ids) else np.zeros((0, *winner_map.shape), bool)
    native_mapping = single_mpo_binary_mask_mapping(
        gt_mask, native_masks, gt_labels, native_labels, iou_threshold
    )
    native_count = image_counts(len(gt_labels), relations, np.arange(len(native_ids)), native_mapping, len(predicate_weights))
    native_support = float(
        native_count["supported_gt_relations"] / native_count["num_gt_relations"]
        if native_count["num_gt_relations"] else 0.0
    )
    native_set = set(int(value) for value in native_ids)
    rows = []
    height, width = winner_map.shape
    for epsilon in epsilons:
        max_pixels = int(np.floor(float(epsilon) * height * width))
        best = {"endpoint_support": native_support, "changed_pixels": 0, "swap": None}
        for local_q, query_id in enumerate(np.asarray(semantic_ids, dtype=int)):
            if int(query_id) in native_set:
                continue
            confidence = np.asarray(semantic_masks[local_q], dtype=bool)
            lost = confidence & (winner_map != int(query_id))
            owners = winner_map[lost]
            if not len(owners):
                continue
            margins = weighted_scores[int(query_id)][lost]
            margins = weighted_scores[owners.clip(min=0), np.where(lost)[0], np.where(lost)[1]] - margins
            won_area = int((winner_map == int(query_id)).sum())
            original_area = int(confidence.sum())
            deficit = max(
                0,
                int(np.floor(overlap_mask_area_threshold * original_area)) + 1 - won_area,
            )
            if deficit <= 0 or deficit > len(margins):
                continue
            order = np.argsort(np.maximum(margins, 0.0), kind="stable")[:deficit]
            rescue_mask = (winner_map == int(query_id)).copy()
            lost_positions = np.argwhere(lost)
            if len(order):
                rescue_mask[tuple(lost_positions[order].T)] = True
            for drop_index, drop_id in enumerate(native_ids):
                keep = np.arange(len(native_ids)) != drop_index
                baseline_owner = np.full(winner_map.shape, -1, dtype=np.int32)
                for native_query_id, mask in zip(native_ids, native_masks):
                    baseline_owner[mask] = int(native_query_id)
                proposed_owner = baseline_owner.copy()
                proposed_owner[proposed_owner == int(drop_id)] = -1
                proposed_owner[rescue_mask] = int(query_id)
                changed = int(np.count_nonzero(proposed_owner != baseline_owner))
                if changed > max_pixels:
                    continue
                masks = np.concatenate([
                    np.stack([proposed_owner == int(value) for value in native_ids[keep]])
                    if int(keep.sum()) else np.zeros((0, *winner_map.shape), bool),
                    (proposed_owner == int(query_id))[None],
                ], axis=0)
                labels = np.concatenate([native_labels[keep], [int(semantic_labels[local_q])]])
                mapping = single_mpo_binary_mask_mapping(gt_mask, masks, gt_labels, labels, iou_threshold)
                selected = np.arange(len(masks), dtype=int)
                count = image_counts(len(gt_labels), relations, selected, mapping, len(predicate_weights))
                support = float(count["supported_gt_relations"] / count["num_gt_relations"] if count["num_gt_relations"] else 0.0)
                if support > best["endpoint_support"] or (
                    support == best["endpoint_support"] and changed < best["changed_pixels"]
                ):
                    best = {"endpoint_support": support, "changed_pixels": changed,
                            "swap": {"drop_query_id": int(drop_id), "add_query_id": int(query_id)}}
        best.update({"epsilon": float(epsilon), "native_endpoint_support": native_support,
                     "changed_fraction": float(best["changed_pixels"] / (height * width))})
        rows.append(best)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("all", "train", "test"), default="train")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-mask-area-threshold", type=float, default=0.8)
    parser.add_argument("--epsilons", type=float, nargs="+", default=(0.0, 0.001, 0.0025, 0.005, 0.01, 0.02))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    psg = json.loads(args.psg.read_text())
    test_ids = {str(value) for value in psg.get("test_image_ids", [])}
    records = {}
    with args.manifest.open() as stream:
        for line in stream:
            record = json.loads(line)
            validate_image_record(record)
            records[str(record["file_name"])] = record
    items = [item for item in psg["data"] if item.get("relations") and (
        args.split == "all" or (args.split == "test") == (str(item["image_id"]) in test_ids)
    )]
    items.sort(key=lambda item: str(item["image_id"]))
    if args.max_images is not None:
        items = items[:args.max_images]
    weights = inverse_predicate_weights(predicate_counts([np.asarray(item["relations"]) for item in items], len(psg["predicate_classes"])))
    rows = []
    for item in items:
        record = records[str(item["file_name"])]
        if int(record["schema_version"]) != 1:
            raise ValueError("minimum-change oracle requires p0c_full schema v1 artifacts")
        artifact = Path(record["artifact_file"])
        if not artifact.is_absolute():
            artifact = args.manifest.parent / artifact
        with np.load(artifact) as data:
            class_logits = np.asarray(data["class_logits"], dtype=np.float32)
            mask_logits = np.asarray(data["mask_logits"], dtype=np.float32)
        probs = np.exp(class_logits - class_logits.max(axis=1, keepdims=True)); probs /= probs.sum(axis=1, keepdims=True)
        labels = probs.argmax(axis=1); scores = probs.max(axis=1)
        no_object = class_logits.shape[1] - 1
        semantic_ids = np.flatnonzero((labels != no_object) & (scores > args.threshold))
        state = pre_admission_state(torch.tensor(class_logits), torch.tensor(mask_logits), (int(record["height"]), int(record["width"])), args.threshold, args.mask_threshold, args.overlap_mask_area_threshold)
        candidates = state["candidates"]
        native_ids = np.asarray([int(c["query_id"]) for c in candidates if c["native_keep"]], dtype=int)
        native_labels = np.asarray([int(labels[q]) for q in native_ids], dtype=int)
        target_size = (int(record["height"]), int(record["width"]))
        weighted_probs = upsampled_query_probabilities(torch.tensor(mask_logits), target_size).numpy() * scores[:, None, None]
        gt_mask = load_index_mask(args.gt_seg_root / item["pan_seg_file_name"], item["segments_info"])
        gt_labels = np.asarray([row["category_id"] for row in item["annotations"]], dtype=int)
        curves = minimum_change_curve(
            np.asarray(state["winner_map"]), weighted_probs,
            weighted_probs[semantic_ids] >= args.mask_threshold, semantic_ids,
            labels[semantic_ids], native_ids, native_labels, gt_mask, gt_labels,
            np.asarray(item["relations"], dtype=int), weights, tuple(args.epsilons),
            args.iou_threshold, args.overlap_mask_area_threshold,
        )
        rows.append({"image_id": str(item["image_id"]), "file_name": str(item["file_name"]), "curve": curves})
    args.output.mkdir(parents=True)
    (args.output / "per_image.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")
    aggregate = []
    for index, epsilon in enumerate(args.epsilons):
        points = [row["curve"][index] for row in rows]
        aggregate.append({"epsilon": float(epsilon), "images": len(points),
                          "mean_endpoint_support": float(np.mean([p["endpoint_support"] for p in points])) if points else None,
                          "mean_native_endpoint_support": float(np.mean([p["native_endpoint_support"] for p in points])) if points else None,
                          "mean_changed_fraction": float(np.mean([p["changed_fraction"] for p in points])) if points else None})
    document = {"schema_version": 1, "contract": "fixed-K GT minimum-change relation rescue oracle", "split": args.split,
                "official_test_used_for_model_selection": False, "curve": aggregate,
                "note": "GT-guided qualification oracle; no selector or learner is trained."}
    (args.output / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
