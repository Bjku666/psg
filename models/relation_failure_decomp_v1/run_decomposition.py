"""Measure endpoint, pair, predicate, and final-ranking relation failures.

The input prediction format is the pinned Fair PSG inference pickle.  This
module deliberately keeps the evaluator contract explicit: mask matching is
class-compatible mask IoU > 0.5 with one-to-one SingleMPO deduplication, and
predicate IDs are the zero-based OpenPSG IDs returned by ``scores[:, 1:]``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def rgb2id(color: np.ndarray) -> np.ndarray:
    color = color.astype(np.int32, copy=False)
    return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]


def load_gt_index_mask(path: Path, segments_info: list[dict]) -> np.ndarray:
    encoded = rgb2id(np.asarray(Image.open(path)))
    out = np.full(encoded.shape, -1, dtype=np.int32)
    for i, info in enumerate(segments_info):
        out[encoded == int(info["id"])] = i
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def single_mpo_mapping(
    pred_mask: np.ndarray,
    pred_labels: np.ndarray,
    gt_mask: np.ndarray,
    gt_labels: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[int, int], dict[int, float]]:
    """Return prediction-index -> unique GT-index under the strict contract."""
    pred_ids = [int(x) for x in np.unique(pred_mask) if x >= 0]
    gt_assign: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pi in pred_ids:
        pm = pred_mask == pi
        scores = np.zeros(len(gt_labels), dtype=np.float64)
        for gi in range(len(gt_labels)):
            if int(pred_labels[pi]) != int(gt_labels[gi]):
                continue
            scores[gi] = iou(pm, gt_mask == gi)
        gi = int(scores.argmax())
        if scores[gi] > threshold:
            gt_assign[gi].append((pi, float(scores[gi])))
    # Match Fair PSG exactly: prediction -> best GT, then best prediction per GT.
    mapping: dict[int, int] = {}
    quality: dict[int, float] = {}
    for gi, candidates in gt_assign.items():
        pi, score = sorted(candidates, key=lambda x: (-x[1], x[0]))[0]
        mapping[pi] = gi
        quality[pi] = score
    return mapping, quality


def _as_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pair_rows(item: dict) -> list[dict]:
    pairs = _as_array(item["pairs"]).astype(np.int64)
    scores = _as_array(item["rel_scores"]).astype(np.float64)
    rank = _as_array(item.get("rel_rank", 1.0 - scores[:, 0])).astype(np.float64)
    if len(pairs) != len(np.unique(pairs, axis=0)):
        raise RuntimeError("Carrier contains duplicate directed pairs; --dedup fail contract violated")
    return [
        {
            "row": i,
            "pair": (int(p[0]), int(p[1])),
            "score": float(rank[i]),
            "pred": int(scores[i, 1:].argmax()),
            "pred_score": float(scores[i, 1:].max()),
        }
        for i, p in enumerate(pairs)
    ]


def _mapped_pairs(rows: list[dict], mapping: dict[int, int]) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for row in rows:
        p0, p1 = row["pair"]
        if p0 not in mapping or p1 not in mapping:
            continue
        pair = (mapping[p0], mapping[p1])
        if pair[0] == pair[1]:
            continue
        row = dict(row)
        row["gt_pair"] = pair
        old = out.get(pair)
        if old is None or (row["score"], -row["row"]) > (old["score"], -old["row"]):
            out[pair] = row
    return out


def _top_rows(rows: list[dict], budget: int | str) -> list[dict]:
    ordered = sorted(rows, key=lambda x: (-x["score"], x["row"]))
    if budget != "all":
        ordered = ordered[: int(budget)]
    return ordered


def _retained_mapped_pairs(rows: list[dict], mapping: dict[int, int], budget: int | str) -> set[tuple[int, int]]:
    return set(_mapped_pairs(_top_rows(rows, budget), mapping))


def _relations(entry: dict) -> list[tuple[int, int, int]]:
    return [(int(s), int(o), int(r)) for s, o, r in entry.get("relations", []) if s != o]


def decompose(
    annotation: dict,
    predictions: Iterable[dict],
    gt_seg_root: Path,
    pair_budgets: list[int | str],
    final_budgets: list[int],
) -> dict:
    by_id = {str(x["img_id"]): x for x in predictions}
    entries = [
        x for x in annotation["data"] if str(x["image_id"]) in set(map(str, annotation["test_image_ids"])) and _relations(x)
    ]
    pair_stats = {str(b): Counter() for b in pair_budgets}
    pair_by_predicate = {str(b): defaultdict(Counter) for b in pair_budgets}
    final_stats = {str(k): Counter() for k in final_budgets}
    final_by_predicate = {str(k): defaultdict(Counter) for k in final_budgets}
    bucket_stats = {str(k): Counter() for k in final_budgets}
    rows = []
    missing = []
    for entry in entries:
        key = str(entry["image_id"])
        item = by_id.get(key)
        relations = _relations(entry)
        if item is None:
            missing.append(key)
            for budget in pair_budgets:
                for _, _, predicate in relations:
                    pair_stats[str(budget)]["total"] += 1
                    pair_stats[str(budget)]["endpoint_failure"] += 1
                    pair_by_predicate[str(budget)][predicate]["total"] += 1
                    pair_by_predicate[str(budget)][predicate]["endpoint_failure"] += 1
            for k in final_budgets:
                for _, _, predicate in relations:
                    final_stats[str(k)]["total"] += 1
                    bucket_stats[str(k)]["endpoint_failure"] += 1
                    final_by_predicate[str(k)][predicate]["total"] += 1
                    final_by_predicate[str(k)][predicate]["endpoint_failure"] += 1
            rows.append({"image_id": key, "file_name": entry["file_name"], "n_relations": len(relations), "missing_prediction": True})
            continue
        pred_mask = _as_array(item["mask"])
        pred_labels = _as_array(item["box_label"]).astype(np.int64)
        gt_labels = np.asarray([x["category_id"] for x in entry["annotations"]], dtype=np.int64)
        gt_mask = load_gt_index_mask(gt_seg_root / entry["pan_seg_file_name"], entry["segments_info"])
        mapping, match_iou = single_mpo_mapping(pred_mask, pred_labels, gt_mask, gt_labels)
        endpoint_gt = set(mapping.values())
        carrier_rows = _pair_rows(item)
        mapped = _mapped_pairs(carrier_rows, mapping)
        row_base = {"image_id": key, "file_name": entry["file_name"], "n_relations": len(relations), "n_pred": len(pred_labels), "n_matched": len(mapping)}
        for budget in pair_budgets:
            retained = _retained_mapped_pairs(carrier_rows, mapping, budget)
            for s, o, r in relations:
                pair_stats[str(budget)]["total"] += 1
                pair_by_predicate[str(budget)][r]["total"] += 1
                pair = (s, o)
                if s not in endpoint_gt or o not in endpoint_gt:
                    event = "endpoint_failure"
                elif pair not in retained:
                    event = "pair_failure"
                else:
                    event = "pair_retained"
                pair_stats[str(budget)][event] += 1
                pair_by_predicate[str(budget)][r][event] += 1
        for k in final_budgets:
            retained = _retained_mapped_pairs(carrier_rows, mapping, "all")
            ranked = _retained_mapped_pairs(carrier_rows, mapping, k)
            for s, o, r in relations:
                final_stats[str(k)]["total"] += 1
                final_by_predicate[str(k)][r]["total"] += 1
                pair = (s, o)
                if s not in endpoint_gt or o not in endpoint_gt:
                    bucket = "endpoint_failure"
                elif pair not in retained:
                    bucket = "pair_failure"
                elif mapped[pair]["pred"] != r:
                    bucket = "predicate_failure"
                elif pair not in ranked:
                    bucket = "ranking_failure"
                else:
                    bucket = "success"
                bucket_stats[str(k)][bucket] += 1
                final_by_predicate[str(k)][r][bucket] += 1
                if bucket == "success":
                    final_stats[str(k)]["success"] += 1
        rows.append({**row_base, "matched_gt": sorted(endpoint_gt), "match_iou": match_iou})
    def summarize(counts: Counter, by_predicate: dict[int, Counter], success_key: str) -> dict:
        total = counts["total"]
        micro = counts[success_key] / total if total else None
        class_recalls = {
            str(p): c[success_key] / c["total"] for p, c in sorted(by_predicate.items()) if c["total"]
        }
        balanced = float(np.mean(list(class_recalls.values()))) if class_recalls else None
        return {"counts": dict(counts), "micro_recall": micro, "predicate_balanced_recall": balanced, "per_predicate_recall": class_recalls}

    pair_summary = {
        b: summarize(c, pair_by_predicate[b], "pair_retained") for b, c in pair_stats.items()
    }
    final_summary = {
        k: summarize(c, final_by_predicate[k], "success") for k, c in final_stats.items()
    }
    balanced_buckets = {}
    for k, pred_counts in final_by_predicate.items():
        keys = ("endpoint_failure", "pair_failure", "predicate_failure", "ranking_failure", "success")
        balanced_buckets[k] = {
            key: float(np.mean([c[key] / c["total"] for c in pred_counts.values() if c["total"]]))
            for key in keys
        }
    conditional_oracles = {}
    for k, pred_counts in final_by_predicate.items():
        keys = ("endpoint_failure", "pair_failure", "predicate_failure", "ranking_failure", "success")
        oracle = {}
        for stage in keys[:-1]:
            vals = []
            for c in pred_counts.values():
                if c["total"]:
                    vals.append((c["success"] + c[stage]) / c["total"])
            oracle[stage] = {
                "predicate_balanced_recall": float(np.mean(vals)) if vals else None,
                "gain_vs_observed_pp": (float(np.mean(vals)) - balanced_buckets[k]["success"]) * 100 if vals else None,
            }
        conditional_oracles[k] = oracle
    pair_failure_gate = {}
    for b, pred_counts in pair_by_predicate.items():
        vals = []
        for c in pred_counts.values():
            miss = c["endpoint_failure"] + c["pair_failure"]
            if miss:
                vals.append(c["pair_failure"] / miss)
        pair_failure_gate[b] = {
            "predicate_balanced_fraction_of_endpoint_or_pair_misses": float(np.mean(vals)) if vals else None,
            "micro_fraction_of_endpoint_or_pair_misses": pair_stats[b]["pair_failure"] / (pair_stats[b]["endpoint_failure"] + pair_stats[b]["pair_failure"]) if pair_stats[b]["endpoint_failure"] + pair_stats[b]["pair_failure"] else None,
        }
    return {
        "contract": {"mask_iou": ">0.5", "class_compatible": True, "deduplication": "single_mpo"},
        "n_entries": len(entries),
        "n_prediction_entries": len(predictions) if hasattr(predictions, "__len__") else None,
        "missing_prediction_ids": missing,
        "pair_stats": pair_summary,
        "final_stats": final_summary,
        "failure_buckets": {
            k: {"micro_counts": dict(v), "predicate_balanced_fraction": balanced_buckets[k]}
            for k, v in bucket_stats.items()
        },
        "conditional_oracles": conditional_oracles,
        "pair_failure_gate": pair_failure_gate,
        "per_image": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psg", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt-seg-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pair-budgets", nargs="+", default=["20", "50", "100", "all"])
    ap.add_argument("--final-budgets", nargs="+", type=int, default=[20, 50])
    args = ap.parse_args()
    pair_budgets = [x if x == "all" else int(x) for x in args.pair_budgets]
    with open(args.psg) as f:
        annotation = json.load(f)
    with open(args.predictions, "rb") as f:
        predictions = pickle.load(f)
    result = decompose(annotation, predictions, Path(args.gt_seg_root), pair_budgets, args.final_budgets)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
