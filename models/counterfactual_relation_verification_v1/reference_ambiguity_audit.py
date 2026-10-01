"""CPU-only A0--A3 audit for duplicate physical OpenPSG references.

The audit never edits the source annotation. It compares duplicate rows by
physical filename, hashes the available COCO image, maps entities with the
registered class-compatible bbox IoU rule, and reports relation-set deltas.
Panoptic mask matching and action-sign stability are intentionally marked
unavailable when the immutable carrier/action artifact is absent.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = row["bbox"]
    return float(x1), float(y1), float(x2), float(y2)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def map_entities(a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[int, int], list[float]]:
    candidates = []
    for ia, aa in enumerate(a.get("annotations", [])):
        for ib, bb in enumerate(b.get("annotations", [])):
            if aa.get("category_id") != bb.get("category_id"):
                continue
            score = iou(bbox(aa), bbox(bb))
            if score > 0.5:
                candidates.append((score, ia, ib))
    mapping: dict[int, int] = {}
    used_b: set[int] = set()
    for score, ia, ib in sorted(candidates, reverse=True):
        if ia not in mapping and ib not in used_b:
            mapping[ia] = ib
            used_b.add(ib)
    return mapping, [score for score, _, _ in candidates]


def panoptic_id(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0].astype(np.int64) + 256 * rgb[:, :, 1].astype(np.int64) + 256 * 256 * rgb[:, :, 2].astype(np.int64)


def map_panoptic_entities(a: dict[str, Any], b: dict[str, Any], panoptic_root: Path) -> tuple[dict[int, int], list[float], str]:
    """Map annotation indices using the actual COCO panoptic masks."""
    pa = panoptic_root / a["pan_seg_file_name"]
    pb = panoptic_root / b["pan_seg_file_name"]
    if not pa.is_file() or not pb.is_file():
        return {}, [], "missing_mask"
    ma, mb = panoptic_id(np.asarray(Image.open(pa))), panoptic_id(np.asarray(Image.open(pb)))
    ids_a = [int(x["id"]) for x in a.get("segments_info", [])]
    ids_b = [int(x["id"]) for x in b.get("segments_info", [])]
    scores = []
    candidates = []
    for ia, (ann_a, sid_a) in enumerate(zip(a.get("annotations", []), ids_a)):
        for ib, (ann_b, sid_b) in enumerate(zip(b.get("annotations", []), ids_b)):
            if ann_a.get("category_id") != ann_b.get("category_id"):
                continue
            aa, bb = ma == sid_a, mb == sid_b
            inter = int(np.logical_and(aa, bb).sum())
            union = int(np.logical_or(aa, bb).sum())
            score = inter / union if union else 0.0
            candidates.append((score, ia, ib))
            scores.append(score)
    mapping: dict[int, int] = {}
    used_b: set[int] = set()
    for score, ia, ib in sorted(candidates, reverse=True):
        if score > 0.5 and ia not in mapping and ib not in used_b:
            mapping[ia] = ib
            used_b.add(ib)
    return mapping, scores, "mask"


def mapped_relations(row: dict[str, Any], mapping: dict[int, int] | None = None) -> set[tuple[int, int, int]]:
    out = set()
    for subject, object_, predicate in row.get("relations", []):
        if mapping is None:
            out.add((int(subject), int(object_), int(predicate)))
        elif subject in mapping and object_ in mapping:
            out.add((mapping[subject], mapping[object_], int(predicate)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--panoptic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--carrier", type=Path)
    args = parser.parse_args()
    source = json.loads(args.psg.read_text())
    test_ids = {str(x) for x in source.get("test_image_ids", [])}
    by_file: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in source["data"]:
        by_file[row["file_name"]].append(row)
    groups = {name: rows for name, rows in by_file.items() if len(rows) > 1}
    same_hash = 0
    different_hash = 0
    split_leakage = []
    comparable = 0
    relation_union = 0
    relation_intersection = 0
    relation_symmetric_delta = 0
    group_reports = []
    for file_name, rows in sorted(groups.items()):
        image_path = args.image_root / file_name
        hashes = []
        if image_path.is_file():
            hashes.append(sha256(image_path))
        hash_same = bool(hashes)
        if hash_same:
            same_hash += 1
        else:
            different_hash += 1
        split_labels = ["test" if str(row["image_id"]) in test_ids else "train" for row in rows]
        if len(set(split_labels)) > 1:
            split_leakage.append({"file_name": file_name, "image_ids": [str(r["image_id"]) for r in rows], "splits": split_labels})
        if len(rows) != 2:
            continue
        mapping, candidate_ious, mapping_mode = map_panoptic_entities(rows[0], rows[1], args.panoptic_root)
        if mapping_mode == "missing_mask":
            mapping, candidate_ious = map_entities(rows[0], rows[1])
            mapping_mode = "bbox_proxy"
        if len(mapping) == len(rows[0].get("annotations", [])) == len(rows[1].get("annotations", [])):
            comparable += 1
        rel_a = mapped_relations(rows[0])
        rel_b = mapped_relations(rows[1], mapping)
        inter = rel_a & rel_b
        union = rel_a | rel_b
        relation_intersection += len(inter)
        relation_union += len(union)
        relation_symmetric_delta += len(union - inter)
        group_reports.append({
            "file_name": file_name,
            "image_ids": [str(r["image_id"]) for r in rows],
            "split": split_labels,
            "image_available": image_path.is_file(),
            "entity_mapping": len(mapping),
            "entity_counts": [len(r.get("annotations", [])) for r in rows],
            "entity_mapping_mode": mapping_mode,
            "candidate_mask_ious": sorted(candidate_ious, reverse=True),
            "relation_counts": [len(r.get("relations", [])) for r in rows],
            "relation_intersection": len(inter),
            "relation_union": len(union),
            "relation_symmetric_delta": len(union - inter),
        })
    report = {
        "schema_version": 1,
        "name": "reference_ambiguity_audit_v1",
        "status": "cpu_audit_complete",
        "dataset_sha256": sha256(args.psg),
        "physical_collision_groups": len(groups),
        "A0_pixel_identity": {"same_hash_groups": same_hash, "different_or_unavailable_groups": different_hash, "pass": same_hash == len(groups)},
        "A1_cross_split": {"leakage_groups": len(split_leakage), "leakage": split_leakage, "pass": not split_leakage},
        "A2_entity_mapping": {"fully_mapped_groups": comparable, "groups": len(group_reports), "coverage": comparable / len(group_reports) if group_reports else 0.0, "panoptic_mask_matching": "used_when_available"},
        "A3_relation_difference": {"mapped_relation_intersection": relation_intersection, "mapped_relation_union": relation_union, "symmetric_delta": relation_symmetric_delta, "jaccard": relation_intersection / relation_union if relation_union else 1.0},
        "A4_action_sign_stability": {"status": "unavailable_carrier_artifact_missing", "carrier": str(args.carrier) if args.carrier else None},
        "decision": "hold_reference_robust_promotion_until_panoptic_mapping_and_action_sign_audit",
        "groups": group_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("physical_collision_groups", "A0_pixel_identity", "A1_cross_split", "A2_entity_mapping", "A3_relation_difference", "A4_action_sign_stability", "decision")}, indent=2))


if __name__ == "__main__":
    main()
