#!/usr/bin/env python3
"""Run structured semantic- or competition-margin qualification oracles.

The oracle is deliberately bounded and query-level.  It re-runs the frozen
Mask2Former native eligibility/competition/admission path and reports both
segmentation (PQ) and endpoint/graph proxy metrics.  Relation proxy metrics
are endpoint-support upper bounds when no frozen relation-head predictions are
provided; this is recorded explicitly in the output contract.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping
from models.gssr_p0b_v1.balanced_oracle import predicate_counts, inverse_predicate_weights
from models.gssr_p0_v1.audit import image_counts
from models.endpoint_survival_v1.common import (select_items, load_records, artifact_path, raw_masks,
    relation_endpoints, criticality, biased_state, panoptic_quality, graph_proxy)

def run(args: argparse.Namespace) -> dict:
    psg = json.loads(args.psg.read_text()); records = load_records(args.manifest)
    items = select_items(psg, args.population_manifest, args.max_images)
    if not items: raise ValueError("no eligible relation-bearing images")
    n_pred = len(psg["predicate_classes"])
    freq = predicate_counts([np.asarray(i["relations"]) for i in items], n_pred)
    weights = inverse_predicate_weights(freq)
    rows = []
    for item in items:
        rec = records.get(str(item["file_name"]))
        if rec is None: raise KeyError(f"manifest lacks {item['file_name']}")
        with np.load(artifact_path(rec, args.manifest)) as data:
            cls = np.asarray(data["class_logits"], dtype=np.float32); masks_logits = np.asarray(data["mask_logits"], dtype=np.float32)
            masks = raw_masks(rec, data)
        gt_mask = load_index_mask(args.gt_seg_root / item["pan_seg_file_name"], item["segments_info"])
        gt_labels = np.asarray([a["category_id"] for a in item["annotations"]], dtype=int); pred_labels = np.asarray([q["predicted_class"] for q in rec["queries"]], dtype=int)
        raw_map = single_mpo_binary_mask_mapping(gt_mask, masks, gt_labels, pred_labels, args.iou_threshold)
        crit_gt = relation_endpoints(np.asarray(item["relations"]), len(gt_labels)); qcrit = np.zeros(len(pred_labels), dtype=float)
        c = criticality(np.asarray(item["relations"]), len(gt_labels), freq)
        for q, g in enumerate(raw_map.candidate_to_gt):
            if int(g) in crit_gt: qcrit[q] = 1.0
        bias = qcrit * float(args.margin)
        state = biased_state(cls, masks_logits, (int(item["height"]), int(item["width"])),
                             semantic_bias=bias if args.mode == "semantic" else None,
                             competition_bias=bias if args.mode == "competition" else None,
                             threshold=args.threshold, mask_threshold=args.mask_threshold, overlap=args.overlap,
                             device=args.device)
        candidates = state["candidates"]; cmasks = np.stack([x["winning_mask"] for x in candidates]) if candidates else np.zeros((0, *gt_mask.shape), bool)
        clabels = np.asarray([x["label_id"] for x in candidates], dtype=int); mapping = single_mpo_binary_mask_mapping(gt_mask, cmasks, gt_labels, clabels, args.iou_threshold)
        selected = np.asarray([i for i, x in enumerate(candidates) if x.get("native_keep", False)], dtype=int)
        count = image_counts(len(gt_labels), np.asarray(item["relations"]), selected, mapping, n_pred)
        support = float(np.dot(count["hit_per_predicate"], weights) / np.count_nonzero(weights)) if np.count_nonzero(weights) else 0.
        pq = panoptic_quality(gt_mask, gt_labels, state)
        graph = graph_proxy(np.asarray(item["relations"]), mapping, selected, n_pred)
        rows.append({"image_id": str(item["image_id"]), "file_name": str(item["file_name"]), "margin": float(args.margin), "mode": args.mode,
                     "predicate_balanced_endpoint_support": support, "endpoint_support": float(graph["supported_relations"] / graph["total_relations"] if graph["total_relations"] else 0.),
                     **pq, **graph, "critical_gt": len(crit_gt), "critical_queries": int(qcrit.sum()), "native_k": int(len(selected))})
    def mean(key): return float(np.mean([r[key] for r in rows])) if rows else 0.
    return {"schema_version": 1, "contract": f"bounded {args.mode}-margin oracle -> frozen native pipeline -> PQ + endpoint graph proxy",
            "mode": args.mode, "margin": float(args.margin), "images": len(rows),
            "metrics": {k: mean(k) for k in ("predicate_balanced_endpoint_support", "endpoint_support", "pq", "pq_th", "pq_st", "r@20", "r@50", "mr@20", "mr@50")},
            "per_image": rows, "relation_metrics_note": "R@K/mR@K are endpoint-support upper-bound proxies unless --relation-predictions is integrated; no relation head is altered."}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--psg", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--gt-seg-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--population-manifest", type=Path); p.add_argument("--max-images", type=int); p.add_argument("--mode", choices=("semantic", "competition"), required=True); p.add_argument("--margin", type=float, required=True)
    p.add_argument("--iou-threshold", type=float, default=.5); p.add_argument("--threshold", type=float, default=.5); p.add_argument("--mask-threshold", type=float, default=.5); p.add_argument("--overlap", type=float, default=.8)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    d = run(a); a.output.mkdir(parents=True); (a.output / "summary.json").write_text(json.dumps(d, indent=2) + "\n"); (a.output / "per_image.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in d.pop("per_image")) + "\n"); print(json.dumps(d, indent=2))
if __name__ == "__main__": main()
