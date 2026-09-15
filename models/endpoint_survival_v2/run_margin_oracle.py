#!/usr/bin/env python3
"""Corrected native/semantic/competition endpoint-survival oracle."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.endpoint_survival_v1.common import (
    artifact_path, biased_state, criticality, load_records,
    relation_endpoints, select_items,
)
from models.endpoint_survival_v2.common import (
    aggregate_population, intervention_counts, official_pq_image_stats, raw_masks,
)
from models.gssr_p0_v1.audit import image_counts
from models.gssr_p0b_v1.balanced_oracle import predicate_counts
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping


def semantic_eligibility(class_logits: np.ndarray, threshold: float,
                         semantic_bias: np.ndarray | None = None) -> np.ndarray:
    logits = torch.as_tensor(np.asarray(class_logits, dtype=np.float32)).clone()
    if semantic_bias is not None:
        labels = logits[:, :-1].argmax(-1)
        logits[torch.arange(len(logits)), labels] += torch.as_tensor(semantic_bias, dtype=torch.float32)
    scores, labels = logits.softmax(-1).max(-1)
    return ((labels != logits.shape[1] - 1) & (scores > threshold)).numpy()


def run(args: argparse.Namespace) -> dict:
    if args.mode == "native" and args.margin != 0:
        raise ValueError("native baseline requires --margin 0")
    psg = json.loads(args.psg.read_text())
    records = load_records(args.manifest)
    items = select_items(psg, args.population_manifest, args.max_images)
    if not items:
        raise ValueError("no eligible relation-bearing images")
    num_predicates = len(psg["predicate_classes"])
    frequencies = predicate_counts([np.asarray(item["relations"]) for item in items], num_predicates)
    count_rows, pq_rows, effect_rows, output_rows = [], [], [], []

    started_at = time.monotonic()
    for image_index, item in enumerate(items, start=1):
        record = records.get(str(item["file_name"]))
        if record is None:
            raise KeyError(f"manifest lacks {item['file_name']}")
        with np.load(artifact_path(record, args.manifest)) as data:
            class_logits = np.asarray(data["class_logits"], dtype=np.float32)
            mask_logits = np.asarray(data["mask_logits"], dtype=np.float32)
            masks = raw_masks(record, data, mask_logits)
            stored_winner = np.asarray(data["pre_admission_winner_map"], dtype=np.int32) if "pre_admission_winner_map" in data else None
            if "official_panoptic_segmentation_key" in record:
                stored_official = np.asarray(data[str(record["official_panoptic_segmentation_key"])], dtype=np.int32)
            else:
                stored_official = np.asarray(record["official_panoptic_segmentation"], dtype=np.int32)
        shape = (int(item["height"]), int(item["width"]))
        gt_mask = load_index_mask(args.gt_seg_root / item["pan_seg_file_name"], item["segments_info"])
        gt_labels = np.asarray([segment["category_id"] for segment in item["segments_info"]], dtype=int)
        pred_labels = np.asarray([query["predicted_class"] for query in record["queries"]], dtype=int)
        raw_mapping = single_mpo_binary_mask_mapping(gt_mask, masks, gt_labels, pred_labels, args.iou_threshold)
        critical_gt = relation_endpoints(np.asarray(item["relations"]), len(gt_labels))
        query_is_critical = np.asarray([int(gt) in critical_gt for gt in raw_mapping.candidate_to_gt], dtype=float)
        bias = query_is_critical * float(args.margin)

        native = biased_state(class_logits, mask_logits, shape, threshold=args.threshold,
                              mask_threshold=args.mask_threshold, overlap=args.overlap, device=args.device)
        if stored_winner is not None and not torch.equal(torch.as_tensor(native["winner_map"]), torch.as_tensor(stored_winner)):
            raise AssertionError(f"native winner-map sanity failed for {item['file_name']}")
        assembled_native = np.full(shape, -1, dtype=np.int32) if not native["candidates"] else np.zeros(shape, dtype=np.int32)
        segment_id = 0
        for candidate in native["candidates"]:
            if candidate.get("native_keep", False):
                segment_id += 1
                assembled_native[candidate["winning_mask"]] = segment_id
        if not torch.equal(torch.as_tensor(assembled_native), torch.as_tensor(stored_official)):
            raise AssertionError(f"native panoptic assembly sanity failed for {item['file_name']}")
        changed = native if args.mode == "native" else biased_state(
            class_logits, mask_logits, shape,
            semantic_bias=bias if args.mode == "semantic" else None,
            competition_bias=bias if args.mode == "competition" else None,
            threshold=args.threshold, mask_threshold=args.mask_threshold,
            overlap=args.overlap, device=args.device,
        )
        candidates = changed["candidates"]
        candidate_masks = np.stack([candidate["winning_mask"] for candidate in candidates]) if candidates else np.zeros((0, *gt_mask.shape), dtype=bool)
        candidate_labels = np.asarray([candidate["label_id"] for candidate in candidates], dtype=int)
        mapping = single_mpo_binary_mask_mapping(gt_mask, candidate_masks, gt_labels, candidate_labels, args.iou_threshold)
        selected = np.asarray([index for index, candidate in enumerate(candidates) if candidate.get("native_keep", False)], dtype=int)
        counts = image_counts(len(gt_labels), np.asarray(item["relations"]), selected, mapping, num_predicates)
        native_eligible = semantic_eligibility(class_logits, args.threshold)
        changed_eligible = semantic_eligibility(class_logits, args.threshold, bias if args.mode == "semantic" else None)
        effects = intervention_counts(native, changed, native_eligible, changed_eligible)
        pq_stats = official_pq_image_stats(gt_mask, item["segments_info"], changed)
        count_rows.append(counts)
        pq_rows.append(pq_stats)
        effect_rows.append(effects)
        output_rows.append({
            "image_id": str(item["image_id"]), "file_name": str(item["file_name"]),
            "mode": args.mode, "margin": float(args.margin),
            "gt_per_predicate": counts["gt_per_predicate"].tolist(),
            "hit_per_predicate": counts["hit_per_predicate"].tolist(),
            "num_gt_relations": counts["num_gt_relations"],
            "supported_gt_relations": counts["supported_gt_relations"],
            "critical_gt": len(critical_gt), "critical_queries": int(query_is_critical.sum()),
            **effects,
        })
        if args.progress_every and (image_index % args.progress_every == 0 or image_index == len(items)):
            elapsed = time.monotonic() - started_at
            seconds_per_image = elapsed / image_index
            eta_seconds = seconds_per_image * (len(items) - image_index)
            print(f"progress={image_index}/{len(items)} seconds_per_image={seconds_per_image:.3f} eta_seconds={eta_seconds:.0f}", file=sys.stderr, flush=True)

    aggregate = aggregate_population(count_rows, pq_rows, effect_rows, num_predicates)
    return {
        "schema_version": 2,
        "contract": "full-population predicate-balanced endpoint support + official COCO category-aggregated PQ",
        "mode": args.mode, "margin": float(args.margin), "images": len(items),
        "native_sanity": {"passed": True, "checked_images": len(items)},
        **aggregate,
        "per_image": output_rows,
        "relation_metrics_note": "Only Endpoint Support and Balanced Endpoint Support are reported; no R@K/mR@K proxy is labeled as a relation result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-seg-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--mode", choices=("native", "semantic", "competition"), required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args)
    rows = result.pop("per_image")
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "per_image.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
