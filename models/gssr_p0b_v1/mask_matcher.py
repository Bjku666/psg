"""Validated class-compatible mask matching for formal P0B audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from models.gssr_p0_v1.audit import MatchResult


def rgb2id(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.int64)
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"expected an RGB mask, got shape {color.shape}")
    return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]


def load_index_mask(path: Path, segments_info: list[dict]) -> np.ndarray:
    """Load a panoptic RGB PNG and map segment IDs to annotation indices."""
    if not path.is_file():
        raise FileNotFoundError(path)
    segment_ids = [int(info["id"]) for info in segments_info]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError(f"duplicate segment IDs in {path}")
    ids = rgb2id(np.asarray(Image.open(path).convert("RGB")))
    indexed = np.full(ids.shape, -1, dtype=np.int32)
    for index, segment_id in enumerate(segment_ids):
        indexed[ids == segment_id] = index
    return indexed


def single_mpo_binary_mask_mapping(
    gt_index_mask: np.ndarray,
    candidate_masks: np.ndarray,
    gt_labels: np.ndarray,
    candidate_labels: np.ndarray,
    threshold: float = 0.5,
) -> MatchResult:
    """Map overlapping binary candidate masks to their best compatible GT."""
    gt_index_mask = np.asarray(gt_index_mask)
    candidate_masks = np.asarray(candidate_masks, dtype=bool)
    if candidate_masks.ndim != 3:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    if tuple(candidate_masks.shape[1:]) != tuple(gt_index_mask.shape):
        raise ValueError(
            f"mask dimensions differ: GT {gt_index_mask.shape}, candidates {candidate_masks.shape[1:]}"
        )
    num_candidates = len(candidate_masks)
    num_gt = len(gt_labels)
    if len(candidate_labels) != num_candidates:
        raise ValueError("candidate mask/label counts differ")
    gt_areas = np.bincount(gt_index_mask[gt_index_mask >= 0], minlength=num_gt)
    ious = np.zeros((num_candidates, num_gt), dtype=np.float64)
    for candidate_index, pixels in enumerate(candidate_masks):
        area = int(pixels.sum())
        if area == 0:
            continue
        overlap = gt_index_mask[pixels]
        intersections = np.bincount(overlap[overlap >= 0], minlength=num_gt)
        unions = area + gt_areas - intersections
        ious[candidate_index] = np.divide(
            intersections,
            unions,
            out=np.zeros(num_gt, dtype=np.float64),
            where=unions > 0,
        )
    compatible = np.asarray(candidate_labels)[:, None] == np.asarray(gt_labels)[None, :]
    ious = np.where(compatible, ious, 0.0)
    if ious.size == 0:
        return MatchResult(
            np.full(num_candidates, -1, dtype=np.int64),
            np.zeros(num_candidates, dtype=np.float64),
        )
    best_gt = ious.argmax(axis=1)
    best_iou = ious[np.arange(num_candidates), best_gt]
    return MatchResult(
        np.where(best_iou > threshold, best_gt, -1).astype(np.int64),
        best_iou,
    )


def single_mpo_panoptic_mapping(
    gt_index_mask: np.ndarray,
    candidate_index_mask: np.ndarray,
    gt_labels: np.ndarray,
    candidate_labels: np.ndarray,
    threshold: float = 0.5,
) -> MatchResult:
    """Map a non-overlapping candidate panoptic index mask to GT."""
    candidate_index_mask = np.asarray(candidate_index_mask)
    masks = np.stack(
        [candidate_index_mask == index for index in range(len(candidate_labels))],
        axis=0,
    ) if len(candidate_labels) else np.zeros((0, *candidate_index_mask.shape), dtype=bool)
    return single_mpo_binary_mask_mapping(
        gt_index_mask, masks, gt_labels, candidate_labels, threshold
    )

