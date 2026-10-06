#!/usr/bin/env python3
"""Measure interface, resolution, and scene-context signal on frozen pairs.

This is a grouped, CPU-only mechanism diagnostic.  It never changes carrier
predictions and reports predicate classification signal rather than official
PSG metrics.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_failure_decomp_v1.run_decomposition import (
    load_gt_index_mask,
    single_mpo_mapping,
)


def _box_features(boxes: np.ndarray, labels: np.ndarray, a: int, b: int,
                  h: int, w: int) -> np.ndarray:
    out: list[float] = []
    for i in (a, b):
        x1, y1, x2, y2 = boxes[i]
        bw, bh = max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))
        out.extend(((x1 + x2) / (2 * w), (y1 + y2) / (2 * h),
                    bw / w, bh / h, bw * bh / max(w * h, 1)))
    ax1, ay1, ax2, ay2 = boxes[a]
    bx1, by1, bx2, by2 = boxes[b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
    aa = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    bb = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    out.extend((float(labels[a]) / 200.0, float(labels[b]) / 200.0,
                inter / max(aa + bb - inter, 1e-8),
                inter / max(aa, 1e-8), inter / max(bb, 1e-8),
                ((bx1 + bx2) - (ax1 + ax2)) / (2 * w),
                ((by1 + by2) - (ay1 + ay2)) / (2 * h)))
    return np.asarray(out, dtype=np.float32)


def _interface(mask: np.ndarray, a: int, b: int) -> np.ndarray:
    ma, mb = mask == int(a), mask == int(b)
    h, w = mask.shape[-2:]
    ba = ma ^ ndimage.binary_erosion(ma)
    bb = mb ^ ndimage.binary_erosion(mb)
    da = ndimage.distance_transform_edt(~ba) if ba.any() else np.full(mask.shape, np.inf)
    ca = np.argwhere(ma).mean(axis=0) if ma.any() else np.zeros(2)
    cb = np.argwhere(mb).mean(axis=0) if mb.any() else np.zeros(2)
    aa, ab = float(ma.sum()), float(mb.sum())
    inter = float(np.logical_and(ma, mb).sum())
    union = aa + ab - inter
    contact = float(np.logical_and(ndimage.binary_dilation(ma), mb).sum() +
                    np.logical_and(ndimage.binary_dilation(mb), ma).sum())
    dx, dy = float(cb[1] - ca[1]) / max(w, 1), float(cb[0] - ca[0]) / max(h, 1)
    boundary_dist = float(da[bb].min()) / max(float(max(h, w)), 1.0) if bb.any() else 1.0
    return np.asarray((aa / max(h * w, 1), ab / max(h * w, 1),
                       inter / max(union, 1.0), inter / max(aa, 1.0),
                       inter / max(ab, 1.0), contact / max(aa + ab, 1.0),
                       boundary_dist, dx, dy, float(np.hypot(dx, dy))), dtype=np.float32)


def _downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    h, w = mask.shape
    small = np.asarray(Image.fromarray(mask.astype(np.int32), mode="I").resize(
        (max(1, w // factor), max(1, h // factor)), resample=Image.Resampling.NEAREST))
    return small.astype(mask.dtype, copy=False)


def _context(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values, counts = np.unique(mask[mask >= 0], return_counts=True)
    area = counts.astype(np.float32) / max(mask.size, 1)
    return np.asarray((len(values), float(area.mean()) if len(area) else 0.0,
                       float(area.std()) if len(area) else 0.0,
                       float(area.max()) if len(area) else 0.0,
                       len(np.unique(labels))), dtype=np.float32)


def _collect(annotation: dict, raw_items: list[dict], seg_root: Path):
    entries = {str(x["image_id"]): x for x in annotation["data"]}
    rows, groups = [], []
    for item in raw_items:
        key = str(item["img_id"])
        entry = entries.get(key)
        if entry is None:
            continue
        gt_labels = np.asarray([x["category_id"] for x in entry["annotations"]], dtype=np.int64)
        gt_mask = load_gt_index_mask(seg_root / entry["pan_seg_file_name"], entry["segments_info"])
        pred_mask = np.asarray(item["mask"])
        pred_labels = np.asarray(item["box_label"], dtype=np.int64)
        mapping, _ = single_mpo_mapping(pred_mask, pred_labels, gt_mask, gt_labels)
        inverse = {int(gt): int(pred) for pred, gt in mapping.items()}
        pair_by_gt = {}
        pairs = np.asarray(item["pairs"], dtype=np.int64)
        h, w = pred_mask.shape[-2:]
        scene = _context(pred_mask, pred_labels)
        for rid, (pa, pb) in enumerate(pairs):
            if int(pa) not in mapping or int(pb) not in mapping:
                continue
            gt_pair = (mapping[int(pa)], mapping[int(pb)])
            pair_by_gt.setdefault(gt_pair, (int(pa), int(pb), rid))
        for s, o, predicate in entry.get("relations", []):
            hit = pair_by_gt.get((int(s), int(o)))
            if hit is None:
                continue
            pa, pb, _ = hit
            base = _box_features(np.asarray(item["bboxes"], dtype=np.float32), pred_labels, pa, pb, h, w)
            inter = _interface(pred_mask, pa, pb)
            rows.append({"geometry": base, "interface": inter,
                         "interface2": _interface(_downsample(pred_mask, 2), pa, pb),
                         "interface4": _interface(_downsample(pred_mask, 4), pa, pb),
                         "context": scene, "label": int(predicate)})
            groups.append(key)
    return rows, np.asarray(groups)


def _evaluate(rows: list[dict], groups: np.ndarray, seed: int) -> dict:
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    features = {
        "geometry": lambda r: r["geometry"],
        "geometry_interface": lambda r: np.concatenate((r["geometry"], r["interface"])),
        "geometry_interface_2x": lambda r: np.concatenate((r["geometry"], r["interface2"])),
        "geometry_interface_4x": lambda r: np.concatenate((r["geometry"], r["interface4"])),
        "geometry_context": lambda r: np.concatenate((r["geometry"], r["context"])),
    }
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(rows)), y, groups))
    out = {"n_rows": int(len(rows)), "n_groups": int(len(np.unique(groups))),
           "train_rows": int(len(train)), "test_rows": int(len(test)), "features": {}}
    for name, fn in features.items():
        x = np.stack([fn(r) for r in rows]).astype(np.float32)
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                               max_leaf_nodes=31, random_state=seed)
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        out["features"][name] = {
            "dimensions": int(x.shape[1]),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0)),
        }
    base = out["features"]["geometry"]["balanced_accuracy"]
    for name, value in out["features"].items():
        value["delta_balanced_accuracy_pp"] = 100.0 * (value["balanced_accuracy"] - base)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--seg-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    annotation = json.loads(args.annotation.read_text())
    raw = pickle.loads(args.raw.read_bytes())
    rows, groups = _collect(annotation, raw, args.seg_root)
    if len(rows) < 100 or len(np.unique(groups)) < 10:
        raise RuntimeError(f"insufficient matched rows: {len(rows)} rows / {len(np.unique(groups))} groups")
    result = {"schema_version": 1, "contract": "frozen grouped geometry/interface diagnostic",
              "seed": args.seed, "status": "completed", "result": _evaluate(rows, groups, args.seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
