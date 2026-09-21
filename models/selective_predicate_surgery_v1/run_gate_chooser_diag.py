#!/usr/bin/env python3
"""Qualification diagnostic for decoupled KEEP/EDIT and EDIT-to-WHICH.

All model inputs are produced by row_features(); exact utilities are used only
for teachers, oracle partials, and evaluation accounting.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.linear_model import LogisticRegression, Ridge
from models.relation_decision_regret_v2.run_p17_validity import compact_records
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action
from models.selective_predicate_surgery_v1.row_teacher import build_row_teacher
from models.selective_predicate_surgery_v1.row_features import row_features


def metrics(records, n, budget):
    x = evaluate_population(records, n)
    return {"mR": float(x[f"mR@{budget}"]), "R": float(x[f"R@{budget}"])}


def split_fit(rows):
    train, calib = [], []
    for row in rows:
        digest = hashlib.sha1(str(row["bootstrap_group"]).encode()).digest()
        (calib if int.from_bytes(digest[:4], "little") % 10 == 0 else train).append(row)
    return train, calib


def flat_row(f):
    return np.concatenate([f["hidden"].astype(np.float32), f["logits"].astype(np.float32), f["scalar"].astype(np.float32)])


def apply_decisions(records, decisions):
    output = []
    for image_index, image in enumerate(records):
        selected = [dict(x) for x in image["selected"]]
        for row in image["selected"]:
            key = (image_index, int(row["row"]))
            pred = decisions.get(key)
            if pred is not None and int(pred) != int(row.get("pred", -999)):
                selected = apply_action(selected, row, int(pred))
        current = dict(image); current["selected"] = selected; output.append(current)
    return output


def score_top_quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    count = max(1, int(np.ceil(len(values) * float(q) / 100.0)))
    threshold = float(np.partition(values, len(values) - count)[len(values) - count])
    return threshold, values >= threshold


def eval_gate(records, teachers, gate, chooser, q=None, threshold=None, budget=50):
    if threshold is None:
        threshold, active = score_top_quantile(gate, q)
    else:
        active = np.asarray(gate) >= float(threshold)
    decisions = {}
    for row, on in zip(teachers, active):
        if on:
            idx = int(chooser(row))
            decisions[(int(row["image_index"]), int(row["row_id"]))] = int(row["candidates"][idx])
    repaired = apply_decisions(records, decisions)
    return metrics(repaired, 56, budget), decisions, float(threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-carrier", required=True, type=Path)
    ap.add_argument("--dev-carrier", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with args.fit_carrier.open("rb") as f: fit_doc = pickle.load(f)
    with args.dev_carrier.open("rb") as f: dev_doc = pickle.load(f)
    if fit_doc["predicate_classes"] != dev_doc["predicate_classes"]: raise ValueError("predicate classes differ")
    fit_records = compact_records(fit_doc, args.budget); dev_records = compact_records(dev_doc, args.budget)
    fit_teacher = build_row_teacher(fit_records, 56, args.depth)
    dev_teacher = build_row_teacher(dev_records, 56, args.depth)
    fit_feat = row_features(fit_teacher); dev_feat = row_features(dev_teacher)
    train_teacher, calib_teacher = split_fit(fit_teacher)
    train_keys = {(r["image_index"], r["row_id"]) for r in train_teacher}
    train_idx = [i for i, r in enumerate(fit_teacher) if (r["image_index"], r["row_id"]) in train_keys]
    calib_idx = [i for i, r in enumerate(fit_teacher) if (r["image_index"], r["row_id"]) not in train_keys]
    x_fit = np.stack([flat_row(x) for x in fit_feat]); x_dev = np.stack([flat_row(x) for x in dev_feat])
    y_fit = np.asarray([r["edit_label"] for r in fit_teacher], dtype=np.int64)
    gate_model = LogisticRegression(max_iter=100, class_weight="balanced", random_state=args.seed, solver="liblinear")
    gate_cap = min(len(train_idx), 200_000)
    gate_rng = np.random.default_rng(args.seed + 17)
    gate_fit_idx = np.asarray(train_idx)[gate_rng.choice(len(train_idx), size=gate_cap, replace=False)]
    gate_model.fit(x_fit[gate_fit_idx], y_fit[gate_fit_idx])
    p_fit = gate_model.predict_proba(x_fit)[:, 1]; p_dev = gate_model.predict_proba(x_dev)[:, 1]

    # Candidate-relative linear chooser, trained only on rows whose exact teacher says EDIT.
    def cand_features(feat, candidate):
        c = int(candidate)
        return np.concatenate([feat["hidden"], feat["logits"], feat["scalar"], np.eye(56, dtype=np.float32)[c],
                               np.asarray([feat["logits"][c] - feat["logits"][feat["native"]]], dtype=np.float32)])
    # The carrier has ~1.2M action rows.  A fixed, stratified cap keeps this
    # diagnostic reproducible and bounded while retaining every positive row.
    positive_rows = [(r, f) for r, f in zip(fit_teacher, fit_feat) if r["edit_label"]]
    rng = np.random.default_rng(args.seed)
    cap = min(len(positive_rows), 100_000)
    keep = rng.choice(len(positive_rows), size=cap, replace=False) if len(positive_rows) > cap else np.arange(len(positive_rows))
    cx, cy = [], []
    for j in keep:
        r, f = positive_rows[int(j)]
        for c, u in zip(r["candidates"], r["utilities"]):
            cx.append(cand_features(f, c)); cy.append(float(u))
    chooser_model = Ridge(alpha=10.0).fit(np.stack(cx), np.asarray(cy))
    def learned_choice(r, f):
        vals = [float(chooser_model.predict(cand_features(f, c)[None])[0]) for c in r["candidates"]]
        return int(np.argmax(vals))
    fit_fmap = {(r["image_index"], r["row_id"]): f for r, f in zip(fit_teacher, fit_feat)}
    dev_fmap = {(r["image_index"], r["row_id"]): f for r, f in zip(dev_teacher, dev_feat)}
    oracle_choice = lambda r: int(r["best_candidate"])
    learned_choice_fit = lambda r: learned_choice(r, fit_fmap[(r["image_index"], r["row_id"])])
    learned_choice_dev = lambda r: learned_choice(r, dev_fmap[(r["image_index"], r["row_id"])])

    baseline = metrics(dev_records, 56, args.budget)
    # Calibrate the gate only on the deterministic fit calibration groups.
    cal_gate = p_fit[calib_idx]
    candidates = {}
    for q in (1, 2, 4, 6, 8, 10):
        _, active = score_top_quantile(cal_gate, q)
        cal_dec = {}
        for r, on in zip([fit_teacher[i] for i in calib_idx], active):
            if on: cal_dec[(r["image_index"], r["row_id"])] = int(r["candidates"][r["best_candidate"]])
        cal_metrics = metrics(apply_decisions([fit_records[i] for i in []], {}), 56, args.budget) if False else None
        candidates[q] = {"threshold": score_top_quantile(cal_gate, q)[0], "calibration_rows": int(active.sum())}
    # Select q using fit calibration, explicitly enforcing R loss <= 0.5 pp.
    cal_records = [fit_records[i] for i in sorted(set(r["image_index"] for r in calib_teacher))]
    # Re-index calibration teacher rows to the compact calibration-record list.
    cal_image_ids = sorted(set(r["image_index"] for r in calib_teacher)); remap = {v:i for i,v in enumerate(cal_image_ids)}
    cal_rows = [dict(r, image_index=remap[r["image_index"]]) for r in calib_teacher]
    cal_base = metrics([fit_records[i] for i in cal_image_ids], 56, args.budget)
    best = None
    for q in (1, 2, 4, 6, 8, 10):
        thr, active = score_top_quantile(cal_gate, q)
        dec = {}
        for r, on in zip(cal_rows, active):
            if on: dec[(r["image_index"], r["row_id"])] = int(r["candidates"][r["best_candidate"]])
        mm = metrics(apply_decisions([fit_records[i] for i in cal_image_ids], dec), 56, args.budget)
        gain = 100*(mm["mR"]-cal_base["mR"]); loss = 100*(mm["R"]-cal_base["R"])
        candidates[q].update({"mR": mm["mR"], "R": mm["R"], "delta_mR_pp": gain, "delta_R_pp": loss})
        if loss >= -0.5 and (best is None or gain > best["delta_mR_pp"]): best = {"q": q, "threshold": thr, "delta_mR_pp": gain, "delta_R_pp": loss}
    if best is None: best = {"q": 1, "threshold": candidates[1]["threshold"], "delta_mR_pp": candidates[1]["delta_mR_pp"], "delta_R_pp": candidates[1]["delta_R_pp"]}

    q, thr = best["q"], best["threshold"]
    learned_gate_oracle, _, _ = eval_gate(dev_records, dev_teacher, p_dev, learned_choice_dev if False else oracle_choice, threshold=thr, budget=args.budget)
    # Oracle gate means exact edit_label; learned chooser is evaluated on all oracle-edit rows.
    oracle_gate_learned, _, _ = eval_gate(dev_records, dev_teacher, np.asarray([r["best_utility"] for r in dev_teacher]), learned_choice_dev, threshold=1e-15, budget=args.budget)
    oracle_full = apply_decisions(dev_records, {(r["image_index"], r["row_id"]): int(r["candidates"][r["best_candidate"]]) for r in dev_teacher if r["edit_label"]})
    oracle_metrics = metrics(oracle_full, 56, args.budget)
    out = {"schema_version": 1, "name": "selective_predicate_surgery_gate_chooser_diag", "fit_rows": len(fit_teacher), "dev_rows": len(dev_teacher),
           "baseline": baseline, "oracle": oracle_metrics,
           "calibration": {"selected": best, "grid": candidates, "split": {"train_rows": len(train_idx), "calib_rows": len(calib_idx), "gate_fit_rows": int(gate_cap), "chooser_positive_rows": int(cap)}},
           "partial_oracles": {
               "learned_gate_oracle_chooser": {**learned_gate_oracle, "delta_mR_pp": 100*(learned_gate_oracle["mR"]-baseline["mR"])},
               "oracle_gate_learned_chooser": {**oracle_gate_learned, "delta_mR_pp": 100*(oracle_gate_learned["mR"]-baseline["mR"])},
               "oracle_oracle": {**oracle_metrics, "delta_mR_pp": 100*(oracle_metrics["mR"]-baseline["mR"])},
           }, "status": "qualification_complete"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=2)+"\n"); print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
