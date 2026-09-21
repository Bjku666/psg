#!/usr/bin/env python3
"""Run the fit-only catastrophic-harm audit and gated G5/G6 controls.

The p1.9 chooser and G4 control are replayed with the exact locked split.  No
new representation, carrier, dev tuning, GPU job, or visual feature is used.
G5 changes only the G4 decision cost; G6 is fitted only if the P0 audit passes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v2.run_p17_validity import compact_records
from models.selective_predicate_surgery_v2.ccuv import (
    build_teacher,
    chooser_matrix,
    row_matrix,
    split_records,
)
from models.selective_predicate_surgery_v2.run_ccuv_controls import (
    COVERAGES,
    _fit_chooser_oof,
    _fit_controls,
    _predict_chooser,
    _scores,
)


NUM_PREDICATES = 56
MAX_EDIT_RATE = 5.0
MIN_DELTA_R = -0.5
G6_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _action_matrix(x_row: np.ndarray, rows: Sequence[Mapping], slots: np.ndarray) -> np.ndarray:
    from models.selective_predicate_surgery_v2.ccuv import verifier_matrix
    return verifier_matrix(x_row, rows, slots)


def _class_cost(records: Sequence[Mapping], num_predicates: int = NUM_PREDICATES) -> np.ndarray:
    occurrences = np.zeros(num_predicates, dtype=np.int64)
    for record in records:
        present = {int(predicate) for _, _, predicate in record.get("relations", ())}
        for predicate in present:
            if 0 <= predicate < num_predicates:
                occurrences[predicate] += 1
    active = int(np.count_nonzero(occurrences))
    if active == 0:
        raise ValueError("no active predicates in training population")
    cost = np.zeros(num_predicates, dtype=np.float64)
    cost[occurrences > 0] = 1.0 / (active * occurrences[occurrences > 0])
    return cost


def _transition_counts(transitions: Sequence[str], active: np.ndarray) -> dict[str, int]:
    return dict(Counter(str(value) for value, on in zip(transitions, active) if on))


def _top_mask(scores: np.ndarray, q: float, eligible: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if eligible is None:
        eligible = np.ones(len(scores), dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    count = min(int(eligible.sum()), max(1, int(np.ceil(len(scores) * float(q) / 100.0))))
    active = np.zeros(len(scores), dtype=bool)
    if count:
        candidates = np.flatnonzero(eligible)
        chosen = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
        active[chosen] = True
    return active


def _point(scores: np.ndarray, targets: Mapping[str, np.ndarray], rows: Sequence[Mapping],
          q: float, oracle_gain: float, eligible: np.ndarray | None = None) -> dict:
    active = _top_mask(scores, q, eligible)
    utility = np.asarray(targets["utility"], dtype=np.float64)
    delta_r = np.asarray(targets["delta_r"], dtype=np.float64)
    selected = utility[active]
    positive = float(selected[selected > 0].sum())
    negative = float(selected[selected < 0].sum())
    return {
        "q_percent": float(q),
        "applied_rows": int(active.sum()),
        "applied_edit_rate_percent": float(100.0 * active.mean()),
        "threshold": float(np.min(np.asarray(scores)[active])),
        "delta_mR_pp": float(100.0 * selected.sum()),
        "delta_R_pp": float(100.0 * delta_r[active].sum()),
        "sum_positive_utility": positive,
        "sum_negative_utility": negative,
        "rescue_precision": float(np.mean(selected > 0)) if len(selected) else 0.0,
        "harm_rate": float(np.mean(selected < 0)) if len(selected) else 0.0,
        "oracle_utility_capture": float(positive / oracle_gain) if oracle_gain > 0 else 0.0,
        "transitions": _transition_counts(targets["transition"], active),
        "active_mask": active,
    }


def _select(curve: Sequence[Mapping], require_positive: bool = False) -> dict:
    eligible = [p for p in curve if p["applied_edit_rate_percent"] <= MAX_EDIT_RATE
                and p["delta_R_pp"] >= MIN_DELTA_R]
    if require_positive:
        positive = [p for p in eligible if p["delta_mR_pp"] > 0]
        if positive:
            eligible = positive
    if not eligible:
        eligible = list(curve)
    return dict(max(eligible, key=lambda p: (p["delta_mR_pp"], -p["q_percent"])))


def _positive_ceiling(scores: np.ndarray, targets: Mapping[str, np.ndarray],
                     rows: Sequence[Mapping], q: float, oracle_gain: float) -> dict:
    utility = np.asarray(targets["utility"], dtype=np.float64)
    count = max(1, int(np.ceil(len(utility) * q / 100.0)))
    positive = np.flatnonzero(utility > 0)
    order = positive[np.argsort(-utility[positive], kind="stable")[:count]]
    active = np.zeros(len(utility), dtype=bool)
    active[order] = True
    # Use the shared point formatter; scores only provide an auditable threshold.
    proxy = np.full(len(utility), -np.inf, dtype=np.float64)
    proxy[active] = utility[active]
    point = _point(proxy, targets, rows, q, oracle_gain)
    point.pop("active_mask", None)
    point["ceiling_definition"] = "top positive proposal utilities within the existing proposal support"
    return point


def _g4_probabilities(models: Mapping, x_row: np.ndarray, rows: Sequence[Mapping],
                      slots: np.ndarray) -> dict[str, np.ndarray]:
    action_x = _action_matrix(x_row, rows, slots)
    g4 = models["G4"].predict_proba(action_x)
    columns = {int(value): index for index, value in enumerate(models["G4"].classes_)}
    return {
        "rescue": g4[:, columns.get(1, -1)] if 1 in columns else np.zeros(len(rows)),
        "harm": g4[:, columns.get(2, -1)] if 2 in columns else np.zeros(len(rows)),
    }


def _risk_scores(prob: Mapping[str, np.ndarray], rows: Sequence[Mapping], slots: np.ndarray,
                 costs: np.ndarray, lam: float) -> np.ndarray:
    candidates = np.asarray([int(row["candidates"][slot]) for row, slot in zip(rows, slots)])
    natives = np.asarray([int(row["native"]) for row in rows])
    return prob["rescue"] * costs[candidates] - float(lam) * prob["harm"] * costs[natives]


def _g6_fit(train_x: np.ndarray, train_rows: Sequence[Mapping], train_slots: np.ndarray,
            seed: int) -> tuple[LogisticRegression, LogisticRegression, dict]:
    action_x = _action_matrix(train_x, train_rows, train_slots)
    labels = np.asarray([str(row["transitions"][slot]) for row, slot in zip(train_rows, train_slots)], dtype=object)
    y_rescue = (labels == "wrong_to_correct").astype(np.int64)
    y_harm = (labels == "correct_to_wrong").astype(np.int64)
    rescue = LogisticRegression(max_iter=160, class_weight="balanced", solver="liblinear", random_state=seed)
    harm = LogisticRegression(max_iter=160, class_weight="balanced", solver="liblinear", random_state=seed + 1)
    rescue.fit(action_x, y_rescue)
    harm.fit(action_x, y_harm)
    return rescue, harm, {"rows": int(len(action_x)), "rescue_positive": int(y_rescue.sum()), "harm_positive": int(y_harm.sum())}


def _g6_probabilities(models: tuple[LogisticRegression, LogisticRegression],
                      x_row: np.ndarray, rows: Sequence[Mapping], slots: np.ndarray) -> dict[str, np.ndarray]:
    action_x = _action_matrix(x_row, rows, slots)
    return {"rescue": models[0].predict_proba(action_x)[:, 1],
            "harm": models[1].predict_proba(action_x)[:, 1]}


def _strip_masks(obj):
    if isinstance(obj, dict):
        return {key: _strip_masks(value) for key, value in obj.items() if key != "active_mask"}
    if isinstance(obj, list):
        return [_strip_masks(value) for value in obj]
    return obj


def _partition_audit(name: str, rows: Sequence[Mapping], slots: np.ndarray,
                    targets: Mapping[str, np.ndarray], g4_scores: np.ndarray,
                    oracle_gain: float, costs: np.ndarray) -> dict:
    learned = [_point(g4_scores, targets, rows, q, oracle_gain) for q in COVERAGES]
    harm = []
    native = []
    for q in COVERAGES:
        utility = np.asarray(targets["utility"])
        transition = np.asarray(targets["transition"], dtype=object)
        harm_mask = utility >= 0.0
        native_mask = np.asarray([not str(v).startswith("correct_to_") for v in transition])
        harm.append(_point(g4_scores, targets, rows, q, oracle_gain, harm_mask))
        native.append(_point(g4_scores, targets, rows, q, oracle_gain, native_mask))
    ceiling = [_positive_ceiling(g4_scores, targets, rows, q, oracle_gain) for q in COVERAGES]
    for curve in (learned, harm, native):
        for point in curve:
            point.pop("active_mask", None)
    return {
        "population": name,
        "oracle_gain": float(oracle_gain),
        "oracle_gain_pp": float(100.0 * oracle_gain),
        "curves": {
            "learned": learned,
            "oracle_harm_veto": harm,
            "oracle_native_correct_veto": native,
            "positive_only_proposal_ceiling": ceiling,
        },
        "selected": {
            "learned": _select(learned),
            "oracle_harm_veto": _select(harm, require_positive=True),
            "oracle_native_correct_veto": _select(native, require_positive=True),
            "positive_only_proposal_ceiling": _select(ceiling, require_positive=True),
        },
    }


def _control_curve(scores: np.ndarray, targets: Mapping[str, np.ndarray], rows: Sequence[Mapping],
                   oracle_gain: float) -> list[dict]:
    curve = [_point(scores, targets, rows, q, oracle_gain) for q in COVERAGES]
    for point in curve:
        point.pop("active_mask", None)
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-carrier", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--linear-cap", type=int, default=200000)
    parser.add_argument("--gbdt-cap", type=int, default=100000)
    args = parser.parse_args()

    with args.fit_carrier.open("rb") as stream:
        document = pickle.load(stream)
    records = compact_records(document, args.budget)
    partitions = split_records(records)
    train_rows = build_teacher(partitions["train"])
    select_rows = build_teacher(partitions["select"])
    calib_rows = build_teacher(partitions["calib"])
    train_x = row_matrix(train_rows)
    select_x = row_matrix(select_rows)
    calib_x = row_matrix(calib_rows)
    oof_slots, chooser, chooser_meta = _fit_chooser_oof(train_rows, train_x, args.seed)
    controls = _fit_controls(train_x, train_rows, oof_slots, args.seed, args.linear_cap, args.gbdt_cap)
    slots = {
        "train": oof_slots,
        "select": _predict_chooser(chooser, select_x, select_rows),
        "calib": _predict_chooser(chooser, calib_x, calib_rows),
    }
    matrices = {"train": (train_x, train_rows), "select": (select_x, select_rows), "calib": (calib_x, calib_rows)}
    costs = _class_cost(partitions["train"])

    partitions_out = {}
    for name in ("select", "calib"):
        x_row, rows = matrices[name]
        targets = __import__("models.selective_predicate_surgery_v2.ccuv", fromlist=["proposal_targets"]).proposal_targets(rows, slots[name])
        scores = _scores(controls, x_row, rows, slots[name])["G4"]
        oracle_gain = float(sum(max(0.0, float(row["signed_best_utility"])) for row in rows))
        partitions_out[name] = _partition_audit(name, rows, slots[name], targets, scores, oracle_gain, costs)

    audit_select = partitions_out["select"]["selected"]["oracle_harm_veto"]
    audit_calib = partitions_out["calib"]["selected"]["oracle_harm_veto"]
    audit_gate = {
        "select_capture_ge_20_percent": audit_select["oracle_utility_capture"] >= 0.20,
        "calib_capture_ge_20_percent": audit_calib["oracle_utility_capture"] >= 0.20,
        "select_delta_mR_positive": audit_select["delta_mR_pp"] > 0.0,
        "calib_delta_mR_positive": audit_calib["delta_mR_pp"] > 0.0,
    }
    audit_gate["pass"] = all(audit_gate.values())

    output = {
        "schema_version": 1,
        "name": "selective_predicate_surgery_v3_catastrophic_harm_audit",
        "contract": {"fit_only_until_gate": True, "dev_tuning_forbidden": True,
                      "seed": args.seed, "coverages_percent": list(COVERAGES),
                      "g5_lambda": 1.0, "g6_lambda_grid": list(G6_LAMBDAS)},
        "fit_population": {"records": {key: len(value) for key, value in partitions.items()},
                           "rows": {"train": len(train_rows), "select": len(select_rows), "calib": len(calib_rows)},
                           "groups": {key: len({str(r.get('bootstrap_group')) for r in value}) for key, value in partitions.items()}},
        "chooser": chooser_meta,
        "controls_replayed": controls["fit_rows"],
        "class_cost": {"active_predicates": int(np.count_nonzero(costs)),
                        "min": float(costs[costs > 0].min()), "max": float(costs.max())},
        "audit": {"select": partitions_out["select"], "calib": partitions_out["calib"], "gate": audit_gate},
        "status": "audit_promoted" if audit_gate["pass"] else "audit_killed",
    }

    if audit_gate["pass"]:
        # P1/G5 uses the frozen G4 probability model; only the decision score changes.
        g5 = {}
        for name in ("select", "calib"):
            x_row, rows = matrices[name]
            prob = _g4_probabilities(controls, x_row, rows, slots[name])
            targets = __import__("models.selective_predicate_surgery_v2.ccuv", fromlist=["proposal_targets"]).proposal_targets(rows, slots[name])
            oracle_gain = float(sum(max(0.0, float(row["signed_best_utility"])) for row in rows))
            score = _risk_scores(prob, rows, slots[name], costs, 1.0)
            curve = _control_curve(score, targets, rows, oracle_gain)
            g5[name] = {"curve": curve, "selected": _select(curve, require_positive=True)}
        # P2/G6 is authorized only after the P0 audit.  It remains fit/select/calib only.
        g6_models = _g6_fit(train_x, train_rows, slots["train"], args.seed)
        g6 = {"fit": g6_models[2]}
        for name in ("select", "calib"):
            x_row, rows = matrices[name]
            prob = _g6_probabilities(g6_models[:2], x_row, rows, slots[name])
            targets = __import__("models.selective_predicate_surgery_v2.ccuv", fromlist=["proposal_targets"]).proposal_targets(rows, slots[name])
            oracle_gain = float(sum(max(0.0, float(row["signed_best_utility"])) for row in rows))
            by_lambda = {}
            for lam in G6_LAMBDAS:
                score = _risk_scores(prob, rows, slots[name], costs, lam)
                curve = _control_curve(score, targets, rows, oracle_gain)
                by_lambda[str(lam)] = {"curve": curve, "selected": _select(curve, require_positive=True)}
            g6[name] = by_lambda
        # Lambda is selected on select and frozen for calib.
        choices = [(float(lam), g6["select"][str(lam)]["selected"]) for lam in G6_LAMBDAS]
        chosen_lam, chosen_select = max(choices, key=lambda pair: (pair[1]["delta_mR_pp"], -pair[0]))
        chosen_calib = g6["calib"][str(chosen_lam)]["selected"]
        g6["selected_lambda"] = chosen_lam
        g6["selected"] = {"select": chosen_select, "calib": chosen_calib}

        g5_select = g5["select"]["selected"]
        g5_calib = g5["calib"]["selected"]
        candidates = [("G5", g5_select, g5_calib), ("G6", chosen_select, chosen_calib)]
        best_name, best_select, best_calib = max(candidates, key=lambda item: item[2]["delta_mR_pp"])
        p1_gate = {
            "selected_control": best_name,
            "select_delta_mR_positive": best_select["delta_mR_pp"] > 0,
            "calib_delta_mR_positive": best_calib["delta_mR_pp"] > 0,
            "calib_delta_R_guardrail": best_calib["delta_R_pp"] >= MIN_DELTA_R,
            "select_edit_rate_guardrail": best_select["applied_edit_rate_percent"] <= MAX_EDIT_RATE,
            "calib_edit_rate_guardrail": best_calib["applied_edit_rate_percent"] <= MAX_EDIT_RATE,
            "calib_capture_ge_20_percent": best_calib["oracle_utility_capture"] >= 0.20,
            "calib_gain_vs_g4": best_calib["delta_mR_pp"] > partitions_out["calib"]["selected"]["learned"]["delta_mR_pp"],
        }
        p1_gate["pass"] = all(value for key, value in p1_gate.items() if key != "selected_control")
        output["G5"] = g5
        output["G6"] = g6
        output["fit_promote_gate"] = p1_gate
        output["status"] = "fit_promoted" if p1_gate["pass"] else "fit_killed"
    output = _strip_masks(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "audit_gate": audit_gate,
                      "fit_promote_gate": output.get("fit_promote_gate")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
