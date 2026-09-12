"""Core fixed-budget entity admission audit.

The matching rule mirrors Fair PSG's SingleMPO box matcher: each candidate is
assigned to its best compatible GT above threshold, then at most one candidate
is retained per GT.  The GT set oracle is exact for this declared mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class MatchResult:
    candidate_to_gt: np.ndarray
    candidate_iou: np.ndarray


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for arrays shaped [N,4] and [M,4] in inclusive-free XYXY."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def single_mpo_candidate_mapping(
    gt_boxes: np.ndarray,
    candidate_boxes: np.ndarray,
    gt_labels: np.ndarray,
    candidate_labels: np.ndarray,
    threshold: float = 0.5,
) -> MatchResult:
    """Map each candidate to its best GT, using strict IoU threshold like Fair PSG."""
    ious = box_iou_xyxy(candidate_boxes, gt_boxes)
    if ious.size == 0:
        return MatchResult(
            np.full(len(candidate_boxes), -1, dtype=np.int64),
            np.zeros(len(candidate_boxes), dtype=np.float64),
        )
    compatible = np.asarray(candidate_labels)[:, None] == np.asarray(gt_labels)[None, :]
    ious = np.where(compatible, ious, 0.0)
    best_gt = ious.argmax(axis=1)
    best_iou = ious[np.arange(len(ious)), best_gt]
    best_gt = np.where(best_iou > threshold, best_gt, -1).astype(np.int64)
    return MatchResult(best_gt, best_iou)


def single_mpo_mask_mapping(
    gt_mask: np.ndarray,
    candidate_mask: np.ndarray,
    gt_labels: np.ndarray,
    candidate_labels: np.ndarray,
    threshold: float = 0.5,
) -> MatchResult:
    """SingleMPO mapping for integer index masks with background encoded as -1."""
    num_candidates, num_gt = len(candidate_labels), len(gt_labels)
    ious = np.zeros((num_candidates, num_gt), dtype=np.float64)
    gt_areas = np.bincount(gt_mask[gt_mask >= 0], minlength=num_gt)
    for cand_idx in range(num_candidates):
        pixels = candidate_mask == cand_idx
        cand_area = int(pixels.sum())
        if cand_area == 0:
            continue
        overlap_gt = gt_mask[pixels]
        intersections = np.bincount(overlap_gt[overlap_gt >= 0], minlength=num_gt)
        unions = cand_area + gt_areas - intersections
        ious[cand_idx] = np.divide(
            intersections, unions, out=np.zeros(num_gt, dtype=float), where=unions > 0
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
    best_gt = np.where(best_iou > threshold, best_gt, -1).astype(np.int64)
    return MatchResult(best_gt, best_iou)


def retained_gt(selected: Sequence[int], mapping: MatchResult) -> set[int]:
    """GT entities retained after SingleMPO deduplication."""
    selected = np.asarray(selected, dtype=np.int64)
    valid = mapping.candidate_to_gt[selected] >= 0
    if not valid.any():
        return set()
    by_gt: dict[int, tuple[float, int]] = {}
    for cand in selected[valid]:
        gt = int(mapping.candidate_to_gt[cand])
        key = (float(mapping.candidate_iou[cand]), -int(cand))
        if gt not in by_gt or key > (by_gt[gt][0], -by_gt[gt][1]):
            by_gt[gt] = (float(mapping.candidate_iou[cand]), int(cand))
    return set(by_gt)


def resolve_budget(num_candidates: int, budget: float) -> int:
    if num_candidates <= 0:
        return 0
    if 0 < budget <= 1:
        return min(num_candidates, max(1, int(ceil(budget * num_candidates))))
    if budget >= 1 and float(budget).is_integer():
        return min(num_candidates, int(budget))
    raise ValueError(f"budget must be a ratio in (0,1] or a positive integer; got {budget}")


def score_topk(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="stable")[:k].astype(np.int64)


def diversity_topk(
    scores: np.ndarray, boxes: np.ndarray, k: int, diversity_weight: float = 0.5
) -> np.ndarray:
    """Greedy score/diversity control using normalized score and max box IoU."""
    scores = np.asarray(scores, dtype=np.float64)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    lo, hi = scores.min(), scores.max()
    quality = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)
    overlaps = box_iou_xyxy(boxes, boxes)
    chosen: list[int] = []
    remaining = set(range(len(scores)))
    while remaining and len(chosen) < k:
        def key(i: int) -> tuple[float, float, int]:
            redundancy = max((overlaps[i, j] for j in chosen), default=0.0)
            utility = (1 - diversity_weight) * quality[i] + diversity_weight * (1 - redundancy)
            return float(utility), float(scores[i]), -i
        pick = max(remaining, key=key)
        chosen.append(pick)
        remaining.remove(pick)
    return np.asarray(chosen, dtype=np.int64)


def node_degrees(num_gt: int, relations: np.ndarray) -> np.ndarray:
    degree = np.zeros(num_gt, dtype=np.int64)
    for subject, obj, _predicate in np.asarray(relations, dtype=np.int64).reshape(-1, 3):
        if subject != obj:
            degree[subject] += 1
            degree[obj] += 1
    return degree


def gt_nodewise_topk(
    mapping: MatchResult, scores: np.ndarray, relations: np.ndarray, num_gt: int, k: int
) -> np.ndarray:
    """Independent GT degree utility; duplicates deliberately receive equal utility."""
    degree = node_degrees(num_gt, relations)
    utility = np.zeros(len(scores), dtype=np.float64)
    valid = mapping.candidate_to_gt >= 0
    utility[valid] = degree[mapping.candidate_to_gt[valid]]
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores), -utility))
    return order[:k].astype(np.int64)


def _oracle_gt_nodes(
    supplied: Sequence[int], relations: np.ndarray, k: int
) -> set[int]:
    supplied = sorted(set(int(x) for x in supplied))
    if k <= 0 or not supplied:
        return set()
    if len(supplied) <= k:
        return set(supplied)
    rels = np.asarray(relations, dtype=np.int64).reshape(-1, 3)
    usable = [tuple(map(int, r)) for r in rels if int(r[0]) in supplied and int(r[1]) in supplied]
    if not usable:
        return set(supplied[:k])
    pos = {gt: i for i, gt in enumerate(supplied)}
    n_g, n_r = len(supplied), len(usable)
    # Variables are [v_gt, y_relation]. Maximize relation instances covered.
    c = np.r_[np.zeros(n_g), -np.ones(n_r)]
    rows, lower, upper = [], [], []
    budget_row = np.r_[np.ones(n_g), np.zeros(n_r)]
    rows.append(budget_row); lower.append(-np.inf); upper.append(k)
    for ridx, (subject, obj, _predicate) in enumerate(usable):
        for endpoint in (subject, obj):
            row = np.zeros(n_g + n_r)
            row[n_g + ridx] = 1
            row[pos[endpoint]] = -1
            rows.append(row); lower.append(-np.inf); upper.append(0)
    result = milp(
        c=c,
        integrality=np.ones(n_g + n_r),
        bounds=Bounds(np.zeros(n_g + n_r), np.ones(n_g + n_r)),
        constraints=LinearConstraint(np.stack(rows), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact set-oracle MILP failed: {result.message}")
    chosen = {supplied[i] for i, value in enumerate(result.x[:n_g]) if value > 0.5}
    # Solvers may omit zero-degree nodes; deterministic padding preserves <=K GT nodes.
    for gt in supplied:
        if len(chosen) >= k:
            break
        chosen.add(gt)
    return chosen


def gt_set_oracle(
    mapping: MatchResult, scores: np.ndarray, relations: np.ndarray, k: int
) -> np.ndarray:
    """Exact fixed-K endpoint-support oracle with deterministic candidate padding."""
    supplied = mapping.candidate_to_gt[mapping.candidate_to_gt >= 0]
    chosen_gt = _oracle_gt_nodes(supplied, relations, k)
    selected: list[int] = []
    order = np.argsort(-np.asarray(scores), kind="stable")
    for gt in sorted(chosen_gt):
        choices = [int(i) for i in order if int(mapping.candidate_to_gt[i]) == gt]
        if choices:
            selected.append(choices[0])
    used = set(selected)
    selected.extend(int(i) for i in order if int(i) not in used and len(selected) < k)
    assert len(selected) == k and len(set(selected)) == k
    return np.asarray(selected, dtype=np.int64)


def image_counts(
    num_gt: int,
    relations: np.ndarray,
    selected: Sequence[int],
    mapping: MatchResult,
    num_predicates: int,
) -> dict:
    matched = retained_gt(selected, mapping)
    rels = np.asarray(relations, dtype=np.int64).reshape(-1, 3)
    gt_per_pred = np.zeros(num_predicates, dtype=np.int64)
    hit_per_pred = np.zeros(num_predicates, dtype=np.int64)
    for subject, obj, predicate in rels:
        if subject == obj:
            continue
        gt_per_pred[predicate] += 1
        if int(subject) in matched and int(obj) in matched:
            hit_per_pred[predicate] += 1
    return {
        "num_gt_entities": int(num_gt),
        "matched_gt_entities": len(matched),
        "num_gt_relations": int(gt_per_pred.sum()),
        "supported_gt_relations": int(hit_per_pred.sum()),
        "gt_per_predicate": gt_per_pred,
        "hit_per_predicate": hit_per_pred,
    }


def aggregate_counts(rows: Iterable[dict], num_predicates: int) -> dict:
    rows = list(rows)
    gt_pred = np.zeros(num_predicates, dtype=np.int64)
    hit_pred = np.zeros(num_predicates, dtype=np.int64)
    for row in rows:
        gt_pred += row["gt_per_predicate"]
        hit_pred += row["hit_per_predicate"]
    entity_den = sum(r["num_gt_entities"] for r in rows)
    entity_num = sum(r["matched_gt_entities"] for r in rows)
    rel_den = int(gt_pred.sum())
    rel_num = int(hit_pred.sum())
    valid = gt_pred > 0
    per_pred = np.divide(hit_pred, gt_pred, out=np.zeros_like(hit_pred, dtype=float), where=valid)
    image_recall = [
        r["supported_gt_relations"] / r["num_gt_relations"]
        for r in rows if r["num_gt_relations"] > 0
    ]
    return {
        "images": len(rows),
        "entity_recall_micro": entity_num / entity_den if entity_den else 0.0,
        "endpoint_support_micro": rel_num / rel_den if rel_den else 0.0,
        "endpoint_support_macro_image": float(np.mean(image_recall)) if image_recall else 0.0,
        "predicate_balanced_endpoint_support": float(per_pred[valid].mean()) if valid.any() else 0.0,
        "gt_entities": entity_den,
        "matched_gt_entities": entity_num,
        "gt_relations": rel_den,
        "supported_gt_relations": rel_num,
        "gt_per_predicate": gt_pred.tolist(),
        "hit_per_predicate": hit_pred.tolist(),
        "per_predicate_endpoint_support": per_pred.tolist(),
    }
