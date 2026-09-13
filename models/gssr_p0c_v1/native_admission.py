"""Reproduce Hugging Face Mask2Former native panoptic query admission."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


def pre_admission_state(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    target_size: tuple[int, int],
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
) -> dict:
    """Compute the fixed full-pool pixel competition state.

    The returned ``winner_map`` contains decoder query ids (``-1`` means no
    semantically eligible query won the pixel).  Candidate records are the
    queries with non-empty winning support.  This is the causal boundary used
    by P1 v2: selectors may change entity admission after this function, but
    never the competition itself.
    """
    if class_logits.ndim != 2 or mask_logits.ndim != 3:
        raise ValueError("expected class logits [Q,C+1] and mask logits [Q,H,W]")
    if len(class_logits) != len(mask_logits):
        raise ValueError("class/mask query counts differ")
    probabilities = upsampled_query_probabilities(mask_logits, target_size)
    scores, labels = class_logits.softmax(dim=-1).max(dim=-1)
    num_labels = class_logits.shape[-1] - 1
    keep = labels.ne(num_labels) & scores.gt(threshold)
    eligible_ids = torch.arange(len(labels), device=labels.device)[keep]
    eligible_probabilities = probabilities[keep]
    eligible_scores = scores[keep]
    eligible_labels = labels[keep]
    winner_map = torch.full(target_size, -1, dtype=torch.int32, device=mask_logits.device)
    candidates: list[dict] = []
    if len(eligible_probabilities):
        weighted = eligible_probabilities * eligible_scores[:, None, None]
        winner_index = weighted.argmax(dim=0)
        winner_map = eligible_ids[winner_index].to(torch.int32)
        for index in range(len(eligible_labels)):
            pixels = winner_index == index
            won_area = int(pixels.sum().item())
            if won_area <= 0:
                continue
            original_area = int((weighted[index] >= mask_threshold).sum().item())
            # Match torch's float32 comparison in the official processor;
            # boundary cases such as 48/60 can be represented just above .8.
            area_ratio = (float(torch.tensor(won_area, dtype=torch.float32)
                                / torch.tensor(original_area, dtype=torch.float32))
                          if original_area else 0.0)
            candidates.append({
                "query_id": int(eligible_ids[index].item()),
                "label_id": int(eligible_labels[index].item()),
                "score": float(eligible_scores[index].item()),
                "won_area": won_area,
                "original_area": original_area,
                "area_ratio": area_ratio,
                "native_keep": bool(area_ratio > overlap_mask_area_threshold),
                "winning_mask": pixels.cpu().numpy(),
            })
    return {"winner_map": winner_map.cpu().numpy(), "candidates": candidates}


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
    state = pre_admission_state(class_logits, mask_logits, target_size, threshold,
                                mask_threshold, overlap_mask_area_threshold)
    winner_map = state["winner_map"]
    candidates = state["candidates"]
    # The official processor emits an all-`-1` map when no query survives
    # semantic eligibility; otherwise its initialized void label is zero.
    segmentation = (np.full(target_size, -1, dtype=np.int32)
                    if not candidates else np.zeros(target_size, dtype=np.int32))
    segments: list[dict] = []
    segment_id = 0
    for candidate in candidates:
        if not candidate["native_keep"]:
            continue
        segment_id += 1
        segmentation[candidate["winning_mask"]] = segment_id
        segments.append({
            "id": segment_id,
            "query_id": candidate["query_id"],
            "label_id": candidate["label_id"],
            "score": round(candidate["score"], 6),
            "was_fused": False,
        })
    return segmentation, segments
