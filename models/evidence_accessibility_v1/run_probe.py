#!/usr/bin/env python3
"""Run the registered E0/E1/E2 accessibility probe on a frozen fit carrier.

This is deliberately a fit-only CPU diagnostic.  It keeps the existing
concrete native->candidate proposal support, learns the chooser on train rows,
and fits only Ridge and HistGradientBoosting utility probes.  No dev/test
carrier is read.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action
from models.relation_decision_regret_v2.run_p17_validity import compact_records
from models.selective_predicate_surgery_v2.ccuv import (
    build_teacher,
    proposal_targets,
    row_matrix,
    split_records,
)
from models.selective_predicate_surgery_v2.run_ccuv_controls import (
    COVERAGES,
    _fit_chooser_oof,
    _predict_chooser,
)


def _endpoint_geometry(raw: Mapping, row_id: int) -> np.ndarray:
    pairs = np.asarray(raw["pairs"], dtype=np.int64)
    boxes = np.asarray(raw["bboxes"], dtype=np.float32)
    labels = np.asarray(raw["box_label"], dtype=np.float32)
    height, width = np.asarray(raw["mask"]).shape[-2:]
    a, b = pairs[int(row_id)]
    out = []
    for index in (int(a), int(b)):
        x1, y1, x2, y2 = boxes[index]
        w = max(0.0, float(x2 - x1)); h = max(0.0, float(y2 - y1))
        cx = (float(x1) + float(x2)) / (2.0 * width)
        cy = (float(y1) + float(y2)) / (2.0 * height)
        out.extend((cx, cy, w / width, h / height, (w * h) / (width * height)))
    ax1, ay1, ax2, ay2 = boxes[int(a)]
    bx1, by1, bx2, by2 = boxes[int(b)]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
    aa = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    bb = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    union = max(aa + bb - inter, 1e-8)
    out.extend([
        float(labels[int(a)]) / 200.0, float(labels[int(b)]) / 200.0,
        inter / union, inter / max(aa, 1e-8), inter / max(bb, 1e-8),
        float((bx1 + bx2) - (ax1 + ax2)) / (2.0 * width),
        float((by1 + by2) - (ay1 + ay2)) / (2.0 * height),
    ])
    return np.asarray(out, dtype=np.float32)


def _mask_geometry_cache(raw: Mapping) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute per-query mask statistics once for one image."""
    mask = np.asarray(raw["mask"])
    cache = {}
    for query in np.unique(mask):
        query = int(query)
        m = mask == query
        boundary = m ^ ndimage.binary_erosion(m)
        distance = ndimage.distance_transform_edt(~boundary) if boundary.any() else np.full(mask.shape, np.inf)
        center = np.argwhere(m).mean(axis=0) if m.any() else np.asarray([0.0, 0.0])
        cache[query] = (m, boundary, distance, center, np.asarray(float(m.sum())))
    return cache


def _mask_geometry(raw: Mapping, row_id: int, cache: Mapping) -> np.ndarray:
    pairs = np.asarray(raw["pairs"], dtype=np.int64)
    mask = np.asarray(raw["mask"])
    a, b = pairs[int(row_id)]
    ma, boundary_a, distance_a, ca, area_a_value = cache[int(a)]
    mb, boundary_b, distance_b, cb, area_b_value = cache[int(b)]
    ma = mask == int(a); mb = mask == int(b)
    area_a = float(area_a_value); area_b = float(area_b_value)
    inter = float(np.logical_and(ma, mb).sum())
    union = area_a + area_b - inter
    height, width = mask.shape[-2:]
    dy = float(cb[0] - ca[0]) / max(float(height), 1.0)
    dx = float(cb[1] - ca[1]) / max(float(width), 1.0)
    dist = float(np.hypot(dx, dy))
    # Boundary/contact statistics are computed from binary morphology, so no
    # learned visual feature is introduced.
    dil_a = ndimage.binary_dilation(ma)
    dil_b = ndimage.binary_dilation(mb)
    contact = float(np.logical_and(dil_a, mb).sum() + np.logical_and(dil_b, ma).sum())
    if boundary_a.any() and boundary_b.any():
        min_boundary = float(distance_a[boundary_b].min()) / max(float(max(height, width)), 1.0)
    else:
        min_boundary = 1.0
    return np.asarray((area_a / (height * width), area_b / (height * width),
                       inter / max(union, 1.0), inter / max(area_a, 1.0),
                       inter / max(area_b, 1.0), min_boundary,
                       contact / max(area_a + area_b, 1.0), dx, dy, dist), dtype=np.float32)


def _feature_rows(rows: Sequence[Mapping], records: Sequence[Mapping], raw_by_id: Mapping[str, Mapping], level: str) -> np.ndarray:
    base = row_matrix(rows)
    if level == "e0":
        return base
    extra = []
    current_image_id = None
    current_mask_cache = None
    for row in rows:
        image_id = str(records[int(row["image_index"])] ["image_id"])
        raw = raw_by_id[image_id]
        if level == "e2" and image_id != current_image_id:
            current_image_id = image_id
            current_mask_cache = _mask_geometry_cache(raw)
        endpoint = _endpoint_geometry(raw, int(row["row_id"]))
        if level == "e1":
            extra.append(endpoint)
        elif level == "e2":
            extra.append(np.concatenate((endpoint, _mask_geometry(raw, int(row["row_id"]), current_mask_cache))))
        else:
            raise ValueError(level)
    return np.concatenate((base, np.stack(extra)), axis=1)


def _fit_predict(kind: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if kind == "ridge":
        model = Ridge(alpha=10.0)
    else:
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05,
                                              max_leaf_nodes=15, l2_regularization=1.0,
                                              random_state=0)
    model.fit(x, y)
    return model.predict(z)


def _apply(records: Sequence[Mapping], teachers: Sequence[Mapping], slots: np.ndarray,
           scores: np.ndarray, q: float) -> list[dict]:
    count = max(1, int(np.ceil(len(scores) * float(q) / 100.0)))
    active = np.zeros(len(scores), dtype=bool)
    active[np.argsort(-scores, kind="stable")[:count]] = True
    decisions = {(int(row["image_index"]), int(row["row_id"])): int(row["candidates"][slot])
                 for row, slot, on in zip(teachers, slots, active) if on}
    out = []
    for image_index, image in enumerate(records):
        selected = [dict(row) for row in image["selected"]]
        for row in image["selected"]:
            predicate = decisions.get((image_index, int(row["row"])))
            if predicate is not None and predicate != int(row.get("pred", -1)):
                selected = apply_action(selected, row, predicate)
        current = dict(image); current["selected"] = selected; out.append(current)
    return out


def _metrics(records: Sequence[Mapping]) -> dict:
    value = evaluate_population(records, 56)
    return {"mR": float(value["mR@50"]), "R": float(value["R@50"])}


def _curve(records: Sequence[Mapping], rows: Sequence[Mapping], slots: np.ndarray,
           targets: Mapping[str, np.ndarray], scores: np.ndarray) -> tuple[list[dict], dict]:
    utility = np.asarray(targets["utility"], dtype=float)
    delta_r = np.asarray(targets["delta_r"], dtype=float)
    oracle = float(sum(max(0.0, float(r["signed_best_utility"])) for r in rows))
    native = _metrics(records)
    curve = []
    for q in COVERAGES:
        count = max(1, int(np.ceil(len(scores) * q / 100.0)))
        active = np.zeros(len(scores), dtype=bool); active[np.argsort(-scores, kind="stable")[:count]] = True
        sel = utility[active]; positive = float(sel[sel > 0].sum())
        repaired = _metrics(_apply(records, rows, slots, scores, q))
        curve.append({"q_percent": float(q), "applied_rows": int(active.sum()),
                      "applied_edit_rate_percent": float(100.0 * active.mean()),
                      "delta_mR_pp": 100.0 * (repaired["mR"] - native["mR"]),
                      "delta_R_pp": 100.0 * (repaired["R"] - native["R"]),
                      "sum_positive_utility": positive,
                      "sum_negative_utility": float(sel[sel < 0].sum()),
                      "positive_utility_capture": positive / oracle if oracle > 0 else 0.0,
                      "rescue_precision": float(np.mean(sel > 0)),
                      "harm_rate": float(np.mean(sel < 0))})
    return curve, {"native": native, "oracle_gain_pp": 100.0 * oracle}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrier", required=True, type=Path)
    ap.add_argument("--raw-predictions", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    document = pickle.loads(args.carrier.read_bytes())
    raw = pickle.loads(args.raw_predictions.read_bytes())
    records = compact_records(document, 50)
    raw_by_id = {str(item["img_id"]): item for item in raw}
    partitions = split_records(records)
    rows_by = {key: build_teacher(value) for key, value in partitions.items()}
    x0_by = {key: _feature_rows(rows_by[key], partitions[key], raw_by_id, "e0") for key in partitions}
    train_rows, train_x = rows_by["train"], x0_by["train"]
    oof_slots, chooser, chooser_meta = _fit_chooser_oof(train_rows, train_x, 0, max_positive_rows=50000)
    slots = {"train": oof_slots,
             "select": _predict_chooser(chooser, x0_by["select"], rows_by["select"]),
             "calib": _predict_chooser(chooser, x0_by["calib"], rows_by["calib"])}
    result = {"schema_version": 1, "contract": "fit-only E0/E1/E2 accessibility probe",
              "records": {key: len(value) for key, value in partitions.items()},
              "rows": {key: len(value) for key, value in rows_by.items()},
              "chooser": chooser_meta, "feature_sets": {}, "status": "running"}
    for level in ("e0", "e1", "e2"):
        result["feature_sets"][level] = {}
        x_by = {key: _feature_rows(rows_by[key], partitions[key], raw_by_id, level) for key in partitions}
        for kind in ("ridge", "hist_gradient"):
            train_targets = proposal_targets(train_rows, slots["train"])
            y = np.asarray(train_targets["utility"], dtype=np.float64) * 100000.0
            # Keep the registered controls reproducible and bounded.
            cap = min(len(y), 100000)
            fit_idx = np.arange(cap, dtype=np.int64)
            model_pred = {}
            for name in ("select", "calib"):
                model_pred[name] = _fit_predict(kind, x_by["train"][fit_idx], y[fit_idx], x_by[name])
            for name in ("select", "calib"):
                target = proposal_targets(rows_by[name], slots[name])
                curve, meta = _curve(partitions[name], rows_by[name], slots[name], target, model_pred[name])
                result["feature_sets"][level].setdefault(kind, {})[name] = {**meta, "curve": curve}
        result["feature_sets"][level]["dimensions"] = int(x_by["train"].shape[1])
    result["status"] = "completed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "feature_sets": list(result["feature_sets"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
