from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from models.gssr_p0_v1.audit import aggregate_counts
from models.gssr_p0c_v1.native_admission import upsampled_query_probabilities
from models.gssr_p0c_v1.raw_query_schema import decode_binary_mask


def raw_masks(record: dict, data: np.lib.npyio.NpzFile,
              mask_logits: np.ndarray | None = None) -> np.ndarray:
    if record["queries"] and "mask_rle" in record["queries"][0]:
        return np.stack([decode_binary_mask(query["mask_rle"]) for query in record["queries"]])
    source = data["mask_logits"] if mask_logits is None else mask_logits
    logits = torch.as_tensor(np.asarray(source, dtype=np.float32))
    probabilities = upsampled_query_probabilities(
        logits, (int(record["height"]), int(record["width"])),
    ).numpy()
    return probabilities > 0.5


def score_ranks(queries: list[dict]) -> np.ndarray:
    """Return stable, one-based joint-score ranks (not decoder positions)."""
    scores = np.asarray([float(query["joint_score"]) for query in queries])
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def official_pq_image_stats(
    gt_mask: np.ndarray,
    gt_segments_info: list[dict],
    state: dict,
) -> dict[int, dict[str, float | int]]:
    """COCO panopticapi-equivalent category sufficient statistics.

    Matching, void removal, crowd handling, and the strict IoU > 0.5 rule
    follow the official COCO panoptic evaluator. The returned sufficient
    statistics are intended to be summed over the full population before PQ
    is computed.
    """
    gt_mask = np.asarray(gt_mask, dtype=np.int32)
    kept = [candidate for candidate in state["candidates"] if candidate.get("native_keep", False)]
    pred_masks = [np.asarray(candidate["winning_mask"], dtype=bool) for candidate in kept]
    pred_labels = [int(candidate["label_id"]) for candidate in kept]
    stats: dict[int, dict[str, float | int]] = {}

    def bucket(category: int) -> dict[str, float | int]:
        return stats.setdefault(category, {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0})

    # panopticapi uses the declared GT segment area in its union.
    gt_areas = np.asarray([int(info["area"]) for info in gt_segments_info], dtype=np.int64)
    gt_used: set[int] = set()
    pred_used: set[int] = set()
    void = gt_mask < 0

    # With panoptic partitions and IoU > .5, valid matches are unique.
    for pred_index, (pred_mask, pred_category) in enumerate(zip(pred_masks, pred_labels)):
        pred_area = int(pred_mask.sum())
        if pred_area == 0:
            continue
        void_overlap = int(np.logical_and(pred_mask, void).sum())
        overlaps = gt_mask[pred_mask]
        intersections = np.bincount(overlaps[overlaps >= 0], minlength=len(gt_segments_info))
        for gt_index, gt_info in enumerate(gt_segments_info):
            if gt_index in gt_used or int(gt_info["category_id"]) != pred_category:
                continue
            if int(gt_info.get("iscrowd", 0)):
                continue
            intersection = int(intersections[gt_index])
            union = pred_area + int(gt_areas[gt_index]) - intersection - void_overlap
            iou = intersection / union if union > 0 else 0.0
            if iou > 0.5:
                gt_used.add(gt_index)
                pred_used.add(pred_index)
                current = bucket(pred_category)
                current["tp"] = int(current["tp"]) + 1
                current["iou"] = float(current["iou"]) + iou
                break

    for gt_index, gt_info in enumerate(gt_segments_info):
        if gt_index not in gt_used and not int(gt_info.get("iscrowd", 0)):
            current = bucket(int(gt_info["category_id"]))
            current["fn"] = int(current["fn"]) + 1

    for pred_index, (pred_mask, pred_category) in enumerate(zip(pred_masks, pred_labels)):
        if pred_index in pred_used:
            continue
        area = int(pred_mask.sum())
        if area == 0:
            continue
        ignored = int(np.logical_and(pred_mask, void).sum())
        for gt_index, gt_info in enumerate(gt_segments_info):
            if int(gt_info.get("iscrowd", 0)) and int(gt_info["category_id"]) == pred_category:
                ignored += int(np.logical_and(pred_mask, gt_mask == gt_index).sum())
        if ignored / area <= 0.5:
            current = bucket(pred_category)
            current["fp"] = int(current["fp"]) + 1
    return stats


def aggregate_official_pq(
    rows: Iterable[dict[int, dict[str, float | int]]],
    num_things: int = 80,
) -> dict[str, float | int]:
    combined: dict[int, dict[str, float | int]] = {}
    for row in rows:
        for category, values in row.items():
            current = combined.setdefault(int(category), {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0})
            for key in ("tp", "fp", "fn"):
                current[key] = int(current[key]) + int(values[key])
            current["iou"] = float(current["iou"]) + float(values["iou"])

    def average(categories: list[int]) -> tuple[float, float, float, int]:
        pq_values, sq_values, rq_values = [], [], []
        for category in categories:
            value = combined[category]
            tp, fp, fn = int(value["tp"]), int(value["fp"]), int(value["fn"])
            denominator = tp + 0.5 * fp + 0.5 * fn
            if denominator <= 0:
                continue
            pq_values.append(float(value["iou"]) / denominator)
            sq_values.append(float(value["iou"]) / tp if tp else 0.0)
            rq_values.append(tp / denominator)
        return (
            float(np.mean(pq_values)) if pq_values else 0.0,
            float(np.mean(sq_values)) if sq_values else 0.0,
            float(np.mean(rq_values)) if rq_values else 0.0,
            len(pq_values),
        )

    all_categories = sorted(combined)
    pq, sq, rq, n = average(all_categories)
    pq_th, _, _, n_th = average([category for category in all_categories if category < num_things])
    pq_st, _, _, n_st = average([category for category in all_categories if category >= num_things])
    return {
        "pq": pq, "sq": sq, "rq": rq, "pq_th": pq_th, "pq_st": pq_st,
        "pq_categories": n, "pq_thing_categories": n_th, "pq_stuff_categories": n_st,
        "category_stats": {str(key): value for key, value in sorted(combined.items())},
    }


def intervention_counts(native: dict, changed: dict, native_eligible: np.ndarray,
                        changed_eligible: np.ndarray) -> dict[str, int]:
    native_winner = np.asarray(native["winner_map"])
    changed_winner = np.asarray(changed["winner_map"])
    changed_pixels = native_winner != changed_winner
    affected_queries = set(int(value) for value in native_winner[changed_pixels])
    affected_queries.update(int(value) for value in changed_winner[changed_pixels])
    affected_queries.discard(-1)
    native_segments = sum(bool(candidate.get("native_keep", False)) for candidate in native["candidates"])
    changed_segments = sum(bool(candidate.get("native_keep", False)) for candidate in changed["candidates"])
    return {
        "pixels": int(native_winner.size),
        "changed_pixels": int(changed_pixels.sum()),
        "changed_queries": len(affected_queries),
        "semantic_eligibility_flips": int(np.count_nonzero(native_eligible != changed_eligible)),
        "competition_winner_flips": int(changed_pixels.sum()),
        "native_segment_count": int(native_segments),
        "changed_segment_count": int(changed_segments),
        "segment_count_delta": int(changed_segments - native_segments),
    }


def aggregate_population(count_rows: list[dict], pq_rows: list[dict],
                         intervention_rows: list[dict], num_predicates: int) -> dict:
    graph = aggregate_counts(count_rows, num_predicates)
    pq = aggregate_official_pq(pq_rows)
    pixels = sum(row["pixels"] for row in intervention_rows)
    changed_pixels = sum(row["changed_pixels"] for row in intervention_rows)
    metrics = {
        "endpoint_support": graph["endpoint_support_micro"],
        "predicate_balanced_endpoint_support": graph["predicate_balanced_endpoint_support"],
        **{key: pq[key] for key in ("pq", "sq", "rq", "pq_th", "pq_st", "pq_categories", "pq_thing_categories", "pq_stuff_categories")},
        "changed_pixel_fraction": changed_pixels / pixels if pixels else 0.0,
        "changed_pixels": changed_pixels,
        "total_pixels": pixels,
        "changed_queries": sum(row["changed_queries"] for row in intervention_rows),
        "semantic_eligibility_flips": sum(row["semantic_eligibility_flips"] for row in intervention_rows),
        "competition_winner_flips": sum(row["competition_winner_flips"] for row in intervention_rows),
        "native_segment_count": sum(row["native_segment_count"] for row in intervention_rows),
        "changed_segment_count": sum(row["changed_segment_count"] for row in intervention_rows),
        "native_segment_count_change": sum(row["segment_count_delta"] for row in intervention_rows),
    }
    return {"metrics": metrics, "graph_counts": graph, "pq_category_stats": pq["category_stats"]}
