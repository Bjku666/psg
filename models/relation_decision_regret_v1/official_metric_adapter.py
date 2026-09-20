"""Small, dependency-light adapter for the pinned Fair PSG recall contract.

The adapter deliberately evaluates already selected triplets.  It does not
invent masks, pairs, or labels, and therefore is safe for legal fixed-budget
oracle experiments.  Matching uses the same class-compatible, strict
``SingleMPO`` rule as ``relation_failure_decomp_v1``.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from models.relation_failure_decomp_v1.run_decomposition import (
    _as_array,
    _pair_rows,
    load_gt_index_mask,
    single_mpo_mapping,
)


def _relations(entry: Mapping) -> list[tuple[int, int, int]]:
    return [(int(s), int(o), int(r)) for s, o, r in entry.get("relations", []) if int(s) != int(o)]


def mapped_candidates(entry: Mapping, prediction: Mapping, gt_seg_root: Path,
                      iou_threshold: float = 0.5, include_unmapped: bool = False) -> tuple[list[dict], dict[int, int]]:
    """Convert a carrier prediction into unique, GT-indexed candidate rows."""
    pred_mask = _as_array(prediction["mask"])
    pred_labels = _as_array(prediction["box_label"]).astype(np.int64)
    gt_labels = np.asarray([x["category_id"] for x in entry["annotations"]], dtype=np.int64)
    gt_mask = load_gt_index_mask(gt_seg_root / entry["pan_seg_file_name"], entry["segments_info"])
    mapping, quality = single_mpo_mapping(pred_mask, pred_labels, gt_mask, gt_labels, iou_threshold)
    rows = _pair_rows(prediction)
    relation_scores = _as_array(prediction["rel_scores"]).astype(np.float64)
    pair_features = prediction.get("pair_features")
    if pair_features is not None:
        pair_features = _as_array(pair_features)
        if len(pair_features) != len(rows):
            raise ValueError("pair_features must align with prediction pairs")
    by_pair: dict[tuple[int, int], dict] = {}
    unmapped: list[dict] = []
    for row in rows:
        a, b = row["pair"]
        if a not in mapping or b not in mapping:
            if include_unmapped:
                current = dict(row)
                current["pred_scores"] = relation_scores[int(row["row"])]
                if pair_features is not None:
                    current["pair_features"] = pair_features[int(row["row"])]
                current["gt_pair"] = None
                unmapped.append(current)
            continue
        pair = (mapping[a], mapping[b])
        if pair[0] == pair[1]:
            continue
        # A strict one-to-one mask mapping normally makes this unique.  Keep
        # the strongest row if a malformed upstream artifact collides after
        # mapping, while making the decision explicit and deterministic.
        current = dict(row)
        current["pred_scores"] = relation_scores[int(row["row"])]
        if pair_features is not None:
            current["pair_features"] = pair_features[int(row["row"])]
        current["gt_pair"] = pair
        current["endpoint_iou"] = float(min(quality.get(a, 0.0), quality.get(b, 0.0)))
        old = by_pair.get(pair)
        if old is None or (current["score"], -current["row"]) > (old["score"], -old["row"]):
            by_pair[pair] = current
    values = list(by_pair.values())
    if include_unmapped:
        values.extend(unmapped)
    return list(sorted(values, key=lambda x: x["row"])), mapping


def evaluate_image(relations: Sequence[tuple[int, int, int]], selected: Sequence[Mapping],
                   num_predicates: int, *, nogc_selected: Sequence[Mapping] | None = None) -> dict:
    """Return image-level hit/denominator counts under Fair PSG semantics."""
    gt_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    gt_counts = np.zeros(num_predicates, dtype=np.int64)
    for s, o, predicate in relations:
        if 0 <= int(predicate) < num_predicates:
            gt_by_pair[(int(s), int(o))].append(int(predicate))
            gt_counts[int(predicate)] += 1
    hits = np.zeros(num_predicates, dtype=np.int64)
    seen: set[tuple[int, int]] = set()
    for candidate in selected:
        physical_pair = tuple(map(int, candidate.get("pair", candidate["gt_pair"])))
        if physical_pair in seen:
            raise ValueError("selected set violates SingleMPO unique-pair constraint")
        seen.add(physical_pair)
        if candidate.get("gt_pair") is None:
            continue
        pair = tuple(map(int, candidate["gt_pair"]))
        predicate = int(candidate["pred"])
        # Fair PSG increments once per matching GT relation.  OpenPSG can
        # contain repeated predicate annotations for one directed pair, so a
        # legal candidate may receive more than one hit here.
        hits[predicate] += sum(1 for value in gt_by_pair.get(pair, ()) if value == predicate)
    valid = gt_counts > 0
    per_predicate = np.full(num_predicates, np.nan, dtype=float)
    per_predicate[valid] = hits[valid] / gt_counts[valid]
    total = int(gt_counts.sum())
    r = float(hits.sum() / total) if total else float("nan")
    result = {
        "gt_counts": gt_counts,
        "hit_counts": hits,
        "per_predicate_recall": per_predicate,
        "r": r,
        "selected": [dict(x) for x in selected],
    }
    if nogc_selected is not None:
        # A supplied no-graph selection may contain repeated pairs.  It is
        # evaluated only as a diagnostic and never used by legal oracles.
        ng_hits = np.zeros(num_predicates, dtype=np.int64)
        for candidate in nogc_selected:
            pair = tuple(map(int, candidate["gt_pair"]))
            predicate = int(candidate["pred"])
            ng_hits[predicate] += sum(1 for value in gt_by_pair.get(pair, ()) if value == predicate)
        ng_recall = np.full(num_predicates, np.nan, dtype=float)
        ng_recall[valid] = ng_hits[valid] / gt_counts[valid]
        result["nogc_hit_counts"] = ng_hits
        result["nogc_per_predicate_recall"] = ng_recall
        result["ngr"] = float(ng_hits.sum() / total) if total else float("nan")
    return result


def evaluate_population(images: Iterable[Mapping], num_predicates: int) -> dict:
    """Aggregate image-wise R/mR exactly as the pinned evaluator does.

    Each image mapping must contain ``relations`` and ``selected``.  Missing
    predictions are represented by an empty selection, which is the official
    evaluator's zero-hit behavior.
    """
    rows = []
    for image in images:
        row = evaluate_image(image.get("relations", ()), image.get("selected", ()), num_predicates,
                             nogc_selected=image.get("nogc_selected"))
        if "budget" in image:
            row["budget"] = int(image["budget"])
        row["image_id"] = str(image.get("image_id", ""))
        row["file_name"] = str(image.get("file_name", ""))
        row["bootstrap_group"] = str(image.get("bootstrap_group", image.get("file_name", "")))
        rows.append(row)
    if not rows:
        return {"images": 0, "mR@20": float("nan"), "mR@50": float("nan"),
                "R@20": float("nan"), "R@50": float("nan"), "per_image": []}
    recalls = np.stack([row["per_predicate_recall"] for row in rows])
    r_values = np.asarray([row["r"] for row in rows], dtype=float)
    # One selected set is normally evaluated once per K.  Callers can place
    # ``k`` in the image records to make the output key explicit.
    result: dict[str, object] = {"images": len(rows), "per_image": rows}
    # Average only over predicates that occur in the evaluated population.
    # A grouped fit/dev slice can legitimately omit one of the 56 classes;
    # taking ``mean`` over the resulting NaN columns would make the whole
    # mR undefined even though the pinned evaluator reports the supported
    # predicate mean.
    per_predicate_mean = np.asarray([
        float(np.mean(column[np.isfinite(column)]))
        if np.isfinite(column).any() else np.nan
        for column in recalls.T
    ], dtype=float)
    result["mR"] = float(np.mean(per_predicate_mean[np.isfinite(per_predicate_mean)]))
    result["R"] = float(np.nanmean(r_values))
    # Stable aliases make one-budget runs easy to consume.
    budgets = sorted({int(row.get("budget", len(row["selected"]))) for row in rows})
    if len(budgets) == 1:
        result[f"mR@{budgets[0]}"] = result["mR"]
        result[f"R@{budgets[0]}"] = result["R"]
    return result


def aggregate_budget_rows(rows: Sequence[Mapping], num_predicates: int, budget: int) -> dict:
    """Convenience wrapper for rows produced by an oracle/decoder."""
    normalized = []
    for row in rows:
        current = dict(row)
        current["budget"] = int(budget)
        normalized.append(current)
    result = evaluate_population(normalized, num_predicates)
    result[f"mR@{budget}"] = result.pop("mR")
    result[f"R@{budget}"] = result.pop("R")
    return result
