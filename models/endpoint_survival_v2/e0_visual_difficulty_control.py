#!/usr/bin/env python3
"""E0 with categorical controls and true score ranks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.endpoint_survival_v1.common import artifact_path, criticality, load_records, select_items
from models.endpoint_survival_v2.common import raw_masks, score_ranks
from models.gssr_p0b_v1.balanced_oracle import predicate_counts
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-seg-root", type=Path, required=True)
    parser.add_argument("--stage-audit", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    psg = json.loads(args.psg.read_text())
    records = load_records(args.manifest)
    items = {str(item["image_id"]): item for item in select_items(psg, args.population_manifest)}
    audit = {}
    for line in args.stage_audit.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            audit[str(row["image_id"])] = row
    frequencies = predicate_counts([np.asarray(item["relations"]) for item in items.values()], len(psg["predicate_classes"]))
    rows = []
    for image_id, item in items.items():
        record, stage = records.get(str(item["file_name"])), audit.get(image_id)
        if record is None or stage is None:
            raise KeyError(f"missing frozen input for image {image_id}")
        ranks = score_ranks(record["queries"])
        gt_mask = load_index_mask(args.gt_seg_root / item["pan_seg_file_name"], item["segments_info"])
        with np.load(artifact_path(record, args.manifest)) as data:
            raw_query_masks = raw_masks(record, data)
        raw_mapping = single_mpo_binary_mask_mapping(
            gt_mask, raw_query_masks,
            np.asarray([segment["category_id"] for segment in item["segments_info"]], dtype=int),
            np.asarray([query["predicted_class"] for query in record["queries"]], dtype=int),
            0.5,
        )
        endpoints = {int(endpoint["gt_index"]): endpoint for endpoint in stage["endpoints"]}
        graph_mass = criticality(np.asarray(item["relations"]), len(item["segments_info"]), frequencies)
        image_pixels = int(item["height"]) * int(item["width"])
        for gt_index, segment in enumerate(item["segments_info"]):
            endpoint = endpoints.get(gt_index, {})
            query_ids = [int(query_id) for query_id in endpoint.get("semantic_queries", []) if 0 <= int(query_id) < len(record["queries"])]
            query_id = max(query_ids, key=lambda value: float(record["queries"][value]["joint_score"])) if query_ids else None
            query = record["queries"][query_id] if query_id is not None else None
            rows.append({
                "image_id": image_id, "gt_index": gt_index,
                "relation_criticality": float(graph_mass[gt_index]),
                "relation_critical": bool(graph_mass[gt_index] > 0),
                "survived": int(bool(endpoint.get("admission_queries"))),
                "category_id": int(segment["category_id"]),
                "log_area": float(np.log1p(int(segment.get("area", 0))) - np.log(max(image_pixels, 1))),
                "mask_iou": float(raw_mapping.candidate_iou[query_id]) if query_id is not None else 0.0,
                "class_score": float(query["class_score"]) if query is not None else 0.0,
                "mask_quality": float(query["mask_quality"]) if query is not None else 0.0,
                "score_rank": int(ranks[query_id]) if query_id is not None else len(ranks) + 1,
                "score_rank_fraction": float(ranks[query_id] / len(ranks)) if query_id is not None and len(ranks) else 1.0,
            })
    if not rows:
        raise ValueError("no endpoint rows")

    continuous = ["log_area", "mask_iou", "class_score", "mask_quality", "score_rank_fraction"]
    numeric = StandardScaler().fit_transform(np.asarray([[row[key] for key in continuous] for row in rows], dtype=float))
    categories = np.asarray([[row["category_id"]] for row in rows])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse=False).fit_transform(categories)
    graph = np.asarray([[row["relation_criticality"]] for row in rows])
    target = np.asarray([row["survived"] for row in rows], dtype=int)
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.c_[numeric, categorical, graph], target)
    beta = float(model.coef_[0, -1])

    matched = []
    by_image: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_image.setdefault(row["image_id"], []).append(index)
    for image_id, indices in by_image.items():
        critical = [index for index in indices if rows[index]["relation_critical"]]
        controls = [index for index in indices if not rows[index]["relation_critical"]]
        used: set[int] = set()
        for critical_index in critical:
            pool = [index for index in controls if index not in used and rows[index]["category_id"] == rows[critical_index]["category_id"]]
            if not pool:
                continue
            control_index = min(pool, key=lambda index: float(np.linalg.norm(numeric[critical_index] - numeric[index])))
            used.add(control_index)
            matched.append({
                "image_id": image_id, "critical_row": critical_index, "control_row": control_index,
                "category_id": rows[critical_index]["category_id"],
                "distance": float(np.linalg.norm(numeric[critical_index] - numeric[control_index])),
                "survival_delta": rows[critical_index]["survived"] - rows[control_index]["survived"],
            })
    deltas = [pair["survival_delta"] for pair in matched]
    output = {
        "schema_version": 2,
        "contract": "image-within exact-category matching + categorical-fixed-effect residual model",
        "images": len(by_image), "entities": len(rows), "matched_pairs": len(matched),
        "continuous_covariates": continuous, "categorical_covariates": ["category_id"],
        "rank_definition": "stable one-based descending joint_score rank",
        "logistic": {"beta_graph_criticality": beta, "odds_ratio": float(np.exp(beta)), "intercept": float(model.intercept_[0])},
        "matched_survival_delta_mean": float(np.mean(deltas)) if deltas else None,
        "matched_survival_delta_median": float(np.median(deltas)) if deltas else None,
        "relation_survival": float(np.mean([row["survived"] for row in rows if row["relation_critical"]])),
        "control_survival": float(np.mean([row["survived"] for row in rows if not row["relation_critical"]])),
        "matched": matched, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("matched", "rows")}, indent=2))


if __name__ == "__main__":
    main()
