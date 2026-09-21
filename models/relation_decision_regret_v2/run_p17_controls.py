#!/usr/bin/env python3
"""Fit S0--S7 on fit and evaluate once on the clean development split."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action
from models.relation_decision_regret_v2.repair_accounting import (
    account_repairs,
    merge_repair_accounts,
)
from models.relation_decision_regret_v2.run_p17_validity import compact_records, exact_oracle
from models.relation_decision_regret_v2.simple_action_controls import (
    build_groups,
    fit_controls,
    predicate_frequencies,
)


def apply_control(records: list[dict], groups: list[dict], control: object) -> tuple[list[dict], dict]:
    selected = [[dict(row) for row in record["selected"]] for record in records]
    for group in groups:
        choice = control.choose(group)
        if choice is None:
            continue
        action = group["actions"][int(choice)]
        image_index = int(group["image_index"])
        selected[image_index] = apply_action(
            selected[image_index], group["row"], int(action["predicate"])
        )
    output, accounts = [], []
    for record, repaired in zip(records, selected):
        current = dict(record)
        current["selected"] = repaired
        output.append(current)
        accounts.append(account_repairs(record["relations"], record["selected"], repaired))
    return output, merge_repair_accounts(accounts)


def metrics(records: list[dict], num_predicates: int, budget: int) -> dict:
    result = evaluate_population(records, num_predicates)
    return {"mR": float(result[f"mR@{budget}"]), "R": float(result[f"R@{budget}"])}


def describe(control: object) -> dict:
    output = {"type": type(control).__name__}
    if hasattr(control, "threshold"):
        value = control.threshold
        output["threshold"] = (
            {str(key): float(item) for key, item in value.items()}
            if isinstance(value, dict) else float(value)
        )
    if hasattr(control, "alpha"):
        output["alpha"] = float(control.alpha)
    if hasattr(control, "threshold") and "threshold" not in output:
        output["threshold"] = float(control.threshold)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-carrier", required=True, type=Path)
    parser.add_argument("--dev-carrier", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learned-group-cap", type=int, default=None)
    parser.add_argument("--decisions-output", type=Path, default=None)
    args = parser.parse_args()
    with args.fit_carrier.open("rb") as stream:
        fit_document = pickle.load(stream)
    with args.dev_carrier.open("rb") as stream:
        dev_document = pickle.load(stream)
    if fit_document["predicate_classes"] != dev_document["predicate_classes"]:
        raise ValueError("fit/dev predicate classes differ")
    num_predicates = len(fit_document["predicate_classes"])
    fit_records = compact_records(fit_document, args.budget)
    dev_records = compact_records(dev_document, args.budget)
    frequencies = predicate_frequencies(fit_records, num_predicates)
    fit_groups = build_groups(fit_records, num_predicates, args.depth, frequencies)
    controls = fit_controls(fit_groups, args.seed, args.learned_group_cap)
    dev_groups = build_groups(dev_records, num_predicates, args.depth, frequencies)
    native = metrics(dev_records, num_predicates, args.budget)
    oracle_records, oracle_account = exact_oracle(dev_records, num_predicates, args.depth)
    oracle = metrics(oracle_records, num_predicates, args.budget)
    results = {}
    for mode, control in controls.items():
        repaired, accounting = apply_control(dev_records, dev_groups, control)
        current = metrics(repaired, num_predicates, args.budget)
        current["delta_mR_pp"] = 100.0 * (current["mR"] - native["mR"])
        current["delta_R_pp"] = 100.0 * (current["R"] - native["R"])
        current["accounting"] = accounting
        current["fit"] = describe(control)
        results[mode] = current
    strongest_mode = max(results, key=lambda mode: results[mode]["mR"])
    headroom = oracle["mR"] - native["mR"]
    captured = (
        (results[strongest_mode]["mR"] - native["mR"]) / headroom
        if headroom > 0 else 0.0
    )
    if captured >= 0.70:
        decision = "kill complex network"
    elif captured >= 0.50:
        decision = "lightweight MEAR only"
    else:
        decision = "full MEAR authorized"
    output = {
        "schema_version": 1,
        "name": "p1.7_s0_s7_fit500_dev500",
        "contract": {
            "fit_only_selection": True,
            "dev_evaluated_once": True,
            "confirm_used": False,
            "support": "frozen top-50 physical pairs",
        },
        "fit_images": len(fit_records),
        "dev_images": len(dev_records),
        "fit_action_groups": len(fit_groups),
        "learned_control_group_cap": args.learned_group_cap,
        "dev_action_groups": len(dev_groups),
        "native": native,
        "oracle": {**oracle, "accounting": oracle_account},
        "controls": results,
        "gate2": {
            "strongest": strongest_mode,
            "oracle_headroom_pp": 100.0 * headroom,
            "simple_gain_pp": results[strongest_mode]["delta_mR_pp"],
            "captured_fraction": float(captured),
            "decision": decision,
        },
    }
    if args.decisions_output is not None:
        strongest_records, _ = apply_control(dev_records, dev_groups, controls[strongest_mode])
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.decisions_output.open("wb") as stream:
            pickle.dump(
                {"native": dev_records, "strongest": strongest_records,
                 "strongest_mode": strongest_mode},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
