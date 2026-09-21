#!/usr/bin/env python3
"""Run the preregistered CCUV G0--G4 controls.

The process intentionally fits and selects on the grouped fit population first.
The dev carrier is opened only when the fit select/calibration gates promote a
control, preventing a negative dev result from being used to tune this line.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action
from models.relation_decision_regret_v2.run_p17_validity import compact_records
from models.selective_predicate_surgery_v2.ccuv import (
    UTILITY_SCALE,
    build_teacher,
    proposal_targets,
    row_matrix,
    select_curve_point,
    split_records,
    utility_curve,
    verifier_matrix,
    chooser_matrix,
)


COVERAGES = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


def _metric(records: Sequence[Mapping], budget: int = 50) -> dict:
    result = evaluate_population(records, 56)
    return {"mR": float(result[f"mR@{budget}"]), "R": float(result[f"R@{budget}"])}


def _sample_indices(labels: np.ndarray, cap: int, seed: int,
                    stratified: bool = False) -> np.ndarray:
    labels = np.asarray(labels)
    n = len(labels)
    if n <= int(cap):
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if not stratified:
        return np.sort(rng.choice(n, size=int(cap), replace=False))
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    chosen = []
    for group in groups:
        take = max(1, int(round(int(cap) * len(group) / n)))
        take = min(take, len(group))
        chosen.extend(rng.choice(group, size=take, replace=False).tolist())
    if len(chosen) > int(cap):
        chosen = rng.choice(np.asarray(chosen), size=int(cap), replace=False).tolist()
    elif len(chosen) < int(cap):
        remaining = np.setdiff1d(np.arange(n), np.asarray(chosen, dtype=np.int64))
        chosen.extend(rng.choice(remaining, size=int(cap) - len(chosen), replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _action_matrix(x_row: np.ndarray, rows: Sequence[Mapping], slots: np.ndarray) -> np.ndarray:
    return verifier_matrix(x_row, rows, slots)


def _predict_chooser(model: Ridge, x_row: np.ndarray, rows: Sequence[Mapping],
                     batch_size: int = 20000) -> np.ndarray:
    choices = np.empty(len(rows), dtype=np.int16)
    for start in range(0, len(rows), int(batch_size)):
        end = min(len(rows), start + int(batch_size))
        chunk_rows = rows[start:end]
        values = []
        for slot in range(len(chunk_rows[0]["candidates"])):
            slots = np.full(end - start, slot, dtype=np.int64)
            features = chooser_matrix(x_row[start:end], chunk_rows, slots)
            values.append(model.predict(features))
        choices[start:end] = np.argmax(np.stack(values, axis=1), axis=1)
    return choices


def _fit_chooser_oof(rows: Sequence[Mapping], x_row: np.ndarray, seed: int,
                     max_positive_rows: int = 100000) -> tuple[np.ndarray, Ridge, dict]:
    """Five grouped OOF chooser predictions plus final train chooser."""
    groups = np.asarray([str(row["bootstrap_group"]) for row in rows], dtype=object)
    fold_ids = np.asarray([
        int.from_bytes(__import__("hashlib").sha1(group.encode()).digest()[:4], "little") % 5
        for group in groups
    ], dtype=np.int64)
    oof = np.empty(len(rows), dtype=np.int16)
    fold_rows = []
    for fold in range(5):
        held = np.flatnonzero(fold_ids == fold)
        fit = np.flatnonzero(fold_ids != fold)
        positive = fit[np.asarray([bool(rows[i]["edit_label"]) for i in fit])]
        keep = _sample_indices(positive, max_positive_rows, seed + 100 + fold)
        positive = positive[keep]
        train_rows = [rows[int(i)] for i in positive]
        cx, cy = [], []
        for slot in range(2):
            slots = np.full(len(train_rows), slot, dtype=np.int64)
            cx.append(chooser_matrix(x_row[positive], train_rows, slots))
            cy.append(np.asarray([float(rows[int(i)]["utilities"][slot]) for i in positive]))
        model = Ridge(alpha=10.0)
        model.fit(np.concatenate(cx), np.concatenate(cy) * UTILITY_SCALE)
        oof[held] = _predict_chooser(model, x_row[held], [rows[int(i)] for i in held])
        fold_rows.append({"fold": fold, "fit_rows": int(len(positive)), "held_rows": int(len(held))})
    positive = np.flatnonzero(np.asarray([bool(row["edit_label"]) for row in rows]))
    positive = positive[_sample_indices(positive, max_positive_rows, seed + 200)]
    final_rows = [rows[int(i)] for i in positive]
    cx, cy = [], []
    for slot in range(2):
        slots = np.full(len(final_rows), slot, dtype=np.int64)
        cx.append(chooser_matrix(x_row[positive], final_rows, slots))
        cy.append(np.asarray([float(rows[int(i)]["utilities"][slot]) for i in positive]))
    final = Ridge(alpha=10.0)
    final.fit(np.concatenate(cx), np.concatenate(cy) * UTILITY_SCALE)
    return oof, final, {"folds": fold_rows, "final_positive_rows": int(len(positive))}


def _fit_controls(x_row: np.ndarray, rows: Sequence[Mapping], oof_slots: np.ndarray,
                  seed: int, linear_cap: int, gbdt_cap: int) -> dict:
    action_x = _action_matrix(x_row, rows, oof_slots)
    targets = proposal_targets(rows, oof_slots)
    utility = targets["utility"] * UTILITY_SCALE
    y_binary = np.asarray([bool(row["edit_label"]) for row in rows], dtype=np.int64)
    y_best = np.asarray([float(row["signed_best_utility"]) for row in rows]) * UTILITY_SCALE
    y_rnh = np.asarray([1 if value > 0 else 2 if value < 0 else 0 for value in targets["utility"]], dtype=np.int64)
    row_idx = _sample_indices(y_binary, linear_cap, seed + 300)
    action_idx = _sample_indices(y_rnh, linear_cap, seed + 301, stratified=True)
    gbdt_idx = _sample_indices(y_rnh, gbdt_cap, seed + 302, stratified=True)

    g0 = LogisticRegression(max_iter=100, class_weight="balanced", solver="liblinear", random_state=seed)
    g0.fit(x_row[row_idx], y_binary[row_idx])
    g1 = Ridge(alpha=10.0)
    g1.fit(x_row[row_idx], y_best[row_idx])
    g2 = Ridge(alpha=10.0)
    g2.fit(action_x[action_idx], utility[action_idx])
    g3 = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=seed,
    )
    g3.fit(action_x[gbdt_idx], utility[gbdt_idx])
    g4 = LogisticRegression(max_iter=120, class_weight="balanced", solver="liblinear",
                            multi_class="ovr", random_state=seed)
    g4.fit(action_x[action_idx], y_rnh[action_idx])
    return {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4,
            "fit_rows": {"row": int(len(row_idx)), "action": int(len(action_idx)), "gbdt": int(len(gbdt_idx))}}


def _scores(models: Mapping, x_row: np.ndarray, rows: Sequence[Mapping], slots: np.ndarray) -> dict[str, np.ndarray]:
    action_x = _action_matrix(x_row, rows, slots)
    g4 = models["G4"].predict_proba(action_x)
    columns = {int(value): index for index, value in enumerate(models["G4"].classes_)}
    rescue = g4[:, columns.get(1, -1)] if 1 in columns else np.zeros(len(rows))
    harm = g4[:, columns.get(2, -1)] if 2 in columns else np.zeros(len(rows))
    return {
        "G0": models["G0"].predict_proba(x_row)[:, 1],
        "G1": models["G1"].predict(x_row),
        "G2": models["G2"].predict(action_x),
        "G3": models["G3"].predict(action_x),
        "G4": rescue - harm,
    }


def _apply(records: Sequence[Mapping], teachers: Sequence[Mapping], slots: np.ndarray,
           scores: np.ndarray, q: float) -> list[dict]:
    active = __import__("models.selective_predicate_surgery_v2.ccuv", fromlist=["exact_top_mask"]).exact_top_mask(scores, q)
    decisions = {
        (int(row["image_index"]), int(row["row_id"])): int(row["candidates"][slot])
        for row, slot, on in zip(teachers, slots, active) if on
    }
    output = []
    for image_index, image in enumerate(records):
        selected = [dict(row) for row in image["selected"]]
        for row in image["selected"]:
            predicate = decisions.get((image_index, int(row["row"])))
            if predicate is not None and predicate != int(row.get("pred", -1)):
                selected = apply_action(selected, row, predicate)
        current = dict(image)
        current["selected"] = selected
        output.append(current)
    return output


def _evaluate_population_partition(records: Sequence[Mapping], teachers: Sequence[Mapping],
                                   slots: np.ndarray, scores: np.ndarray, q: float) -> dict:
    native = _metric(records)
    repaired = _apply(records, teachers, slots, scores, q)
    changed = _metric(repaired)
    return {"native": native, "repaired": changed,
            "delta_mR_pp": 100.0 * (changed["mR"] - native["mR"]),
            "delta_R_pp": 100.0 * (changed["R"] - native["R"])}


def _run_dev(args, fit_doc, train_rows, train_x, chooser, models, selected_name,
             selected_q, output):
    if args.dev_carrier is None:
        output["dev"] = {"status": "not_requested"}
        return output
    print("FIT GATE PASSED; loading dev carrier exactly once", flush=True)
    with Path(args.dev_carrier).open("rb") as stream:
        dev_doc = pickle.load(stream)
    dev_records = compact_records(dev_doc, args.budget)
    dev_rows = build_teacher(dev_records)
    dev_x = row_matrix(dev_rows)
    dev_slots = _predict_chooser(chooser, dev_x, dev_rows)
    dev_targets = proposal_targets(dev_rows, dev_slots)
    dev_scores = _scores(models, dev_x, dev_rows, dev_slots)[selected_name]
    dev_curve = utility_curve(dev_scores, dev_targets, dev_rows, COVERAGES)
    dev_point = next(point for point in dev_curve if point["q_percent"] == selected_q)
    dev_eval = _evaluate_population_partition(dev_records, dev_rows, dev_slots, dev_scores, selected_q)
    oracle_records = _apply(dev_records, dev_rows,
                            np.asarray([row["best_candidate"] for row in dev_rows]),
                            np.asarray([row["signed_best_utility"] for row in dev_rows]), 100.0)
    oracle = _metric(oracle_records)
    native = _metric(dev_records)
    gain = 100.0 * (dev_eval["repaired"]["mR"] - native["mR"])
    gate = {
        "delta_mR_pp": gain,
        "delta_R_pp": dev_eval["delta_R_pp"],
        "oracle_delta_mR_pp": 100.0 * (oracle["mR"] - native["mR"]),
        "minimum_delta_mR_pp": 4.4706,
        "pass": bool(gain >= 4.4706 and dev_eval["delta_R_pp"] >= -0.5 and
                     dev_point["applied_edit_rate_percent"] <= 10.0),
    }
    output["dev"] = {"status": "evaluated_once", "control": selected_name,
                      "q_percent": selected_q, "curve": dev_curve,
                      "selected_point": dev_point, "evaluator": dev_eval,
                      "oracle": oracle, "gate": gate}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-carrier", required=True, type=Path)
    parser.add_argument("--dev-carrier", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--linear-cap", type=int, default=200000)
    parser.add_argument("--gbdt-cap", type=int, default=100000)
    args = parser.parse_args()
    print("Loading fit carrier", flush=True)
    with args.fit_carrier.open("rb") as stream:
        fit_doc = pickle.load(stream)
    fit_records_all = compact_records(fit_doc, args.budget)
    partitions = split_records(fit_records_all)
    print("Building grouped fit teachers", {key: len(value) for key, value in partitions.items()}, flush=True)
    train_records = partitions["train"]
    select_records = partitions["select"]
    calib_records = partitions["calib"]
    train_rows = build_teacher(train_records)
    select_rows = build_teacher(select_records)
    calib_rows = build_teacher(calib_records)
    train_x = row_matrix(train_rows)
    select_x = row_matrix(select_rows)
    calib_x = row_matrix(calib_rows)
    print(f"Rows train/select/calib: {len(train_rows)}/{len(select_rows)}/{len(calib_rows)}", flush=True)
    oof_slots, chooser, chooser_meta = _fit_chooser_oof(train_rows, train_x, args.seed)
    print("Chooser OOF complete", chooser_meta, flush=True)
    models = _fit_controls(train_x, train_rows, oof_slots, args.seed,
                           args.linear_cap, args.gbdt_cap)
    print("Controls fit", models["fit_rows"], flush=True)
    select_slots = _predict_chooser(chooser, select_x, select_rows)
    calib_slots = _predict_chooser(chooser, calib_x, calib_rows)
    select_targets = proposal_targets(select_rows, select_slots)
    calib_targets = proposal_targets(calib_rows, calib_slots)
    select_scores = _scores(models, select_x, select_rows, select_slots)
    calib_scores = _scores(models, calib_x, calib_rows, calib_slots)
    curves, points = {}, {}
    for name in models:
        if name == "fit_rows":
            continue
        curves[name] = {
            "select": utility_curve(select_scores[name], select_targets, select_rows, COVERAGES),
            "calib": utility_curve(calib_scores[name], calib_targets, calib_rows, COVERAGES),
        }
        point = select_curve_point(curves[name]["select"])
        calib_point = next(item for item in curves[name]["calib"] if item["q_percent"] == point["q_percent"])
        points[name] = {"select": point, "calib": calib_point}
        print(name, "select", point["q_percent"], point["delta_mR_pp"],
              "calib", calib_point["delta_mR_pp"], flush=True)
    signed = max(("G1", "G2", "G3", "G4"), key=lambda name: points[name]["calib"]["delta_mR_pp"])
    conditioned = max(("G2", "G3", "G4"), key=lambda name: points[name]["calib"]["delta_mR_pp"])
    fit_gate = {
        "signed_best": signed,
        "conditioned_best": conditioned,
        "signed_select_positive": points[signed]["select"]["delta_mR_pp"] > 0.0,
        "signed_calib_positive": points[signed]["calib"]["delta_mR_pp"] > 0.0,
        "candidate_conditioned_beats_G0": points[conditioned]["calib"]["delta_mR_pp"] > points["G0"]["calib"]["delta_mR_pp"],
        "candidate_conditioned_beats_G1": points[conditioned]["calib"]["delta_mR_pp"] > points["G1"]["calib"]["delta_mR_pp"],
        "calib_R_guardrail": points[conditioned]["calib"]["delta_R_pp"] >= -0.5,
        "coverage_guardrail": points[conditioned]["calib"]["applied_edit_rate_percent"] <= 10.0,
    }
    fit_gate["pass"] = all(value for key, value in fit_gate.items()
                            if key not in {"signed_best", "conditioned_best"})
    output = {
        "schema_version": 1,
        "name": "selective_predicate_surgery_v2_ccuv_controls",
        "contract": {"fit_only_until_gate": True, "dev_tuning_forbidden": True,
                      "seed": args.seed, "coverages_percent": list(COVERAGES)},
        "fit_population": {"records": {key: len(value) for key, value in partitions.items()},
                           "rows": {"train": len(train_rows), "select": len(select_rows), "calib": len(calib_rows)},
                           "groups": {key: len({str(r.get('bootstrap_group')) for r in value}) for key, value in partitions.items()}},
        "chooser": chooser_meta,
        "controls": curves,
        "selected_points": points,
        "fit_gate": fit_gate,
        "status": "fit_promoted" if fit_gate["pass"] else "fit_killed",
    }
    if fit_gate["pass"]:
        output = _run_dev(args, fit_doc, train_rows, train_x, chooser, models,
                          conditioned, points[conditioned]["select"]["q_percent"], output)
        if output["dev"]["status"] == "evaluated_once" and not output["dev"]["gate"]["pass"]:
            output["status"] = "dev_killed"
        elif output["dev"]["status"] == "evaluated_once":
            output["status"] = "dev_promoted_seed0"
    else:
        output["dev"] = {"status": "not_loaded_fit_gate_failed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "fit_gate": fit_gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
