"""Reproduce Hugging Face Mask2Former native panoptic query admission."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


def upsampled_query_probabilities(
    mask_logits: torch.Tensor, target_size: tuple[int, int]
) -> torch.Tensor:
    """Match the processor's logits->384->sigmoid->target resize sequence."""
    logits_384 = functional.interpolate(
        mask_logits.unsqueeze(0), size=(384, 384), mode="bilinear", align_corners=False
    )[0]
    probabilities = logits_384.sigmoid()
    if tuple(target_size) != (384, 384):
        probabilities = functional.interpolate(
            probabilities.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0]
    return probabilities


def native_panoptic_admission(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    target_size: tuple[int, int],
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
) -> tuple[np.ndarray, list[dict]]:
    """Return native segmentation and segment records including raw query IDs.

    Stuff fusion is deliberately disabled, matching the frozen P0 carrier.
    ``official_keep_flag`` therefore means the query survived confidence,
    no-object, pixel competition, and overlap-area checks.
    """
    if class_logits.ndim != 2 or mask_logits.ndim != 3:
        raise ValueError("expected class logits [Q,C+1] and mask logits [Q,H,W]")
    probabilities = upsampled_query_probabilities(mask_logits, target_size)
    scores, labels = class_logits.softmax(dim=-1).max(dim=-1)
    num_labels = class_logits.shape[-1] - 1
    keep = labels.ne(num_labels) & scores.gt(threshold)
    query_ids = torch.arange(len(labels), device=labels.device)[keep]
    probabilities = probabilities[keep]
    scores = scores[keep]
    labels = labels[keep]
    segmentation = torch.zeros(target_size, dtype=torch.int32, device=mask_logits.device)
    if not len(probabilities):
        return segmentation.cpu().numpy(), []

    weighted = probabilities * scores[:, None, None]
    winning_query = weighted.argmax(dim=0)
    segments: list[dict] = []
    segment_id = 0
    for kept_index in range(len(labels)):
        pixels = winning_query == kept_index
        won_area = pixels.sum()
        original_area = (weighted[kept_index] >= mask_threshold).sum()
        exists = won_area > 0 and original_area > 0
        if exists:
            area_ratio = won_area / original_area
            exists = bool(area_ratio.item() > overlap_mask_area_threshold)
        if not exists:
            continue
        segment_id += 1
        segmentation[pixels] = segment_id
        segments.append({
            "id": segment_id,
            "query_id": int(query_ids[kept_index].item()),
            "label_id": int(labels[kept_index].item()),
            "score": round(float(scores[kept_index].item()), 6),
            "was_fused": False,
        })
    return segmentation.cpu().numpy(), segments

