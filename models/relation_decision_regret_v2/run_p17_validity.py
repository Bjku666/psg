#!/usr/bin/env python3
"""Run the P1.7 Gate-0 population-exact surgery validity audit."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.legal_oracle import baseline_selection
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.additivity_audit_v2 import (
    distinct_row_additivity,
    single_action_parity,
)
from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators,
    population_substitution_teacher,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action
from models.relation_decision_regret_v2.repair_accounting import (
    account_repairs,
    merge_repair_accounts,
)


def compact_records(document: dict, budget: int) -> list[dict]:
    return [
        {
            "image_id": str(image["image_id"]),
            "file_name": str(image["file_name"]),
            "bootstrap_group": str(image.get("bootstrap_group", image["file_name"])),
            "relations": image.get("relations", ()),
            "selected": baseline_selection(image.get("candidates", ()), budget),
            "budget": int(budget),
        }
        for image in document["images"]
        if image.get("relations")
    ]


def exact_oracle(records: list[dict], num_predicates: int, depth: int) -> tuple[list[dict], dict]:
    denominators = PopulationDenominators.from_records(records, num_predicates)
    repaired, accounts = [], []
    for record in records:
        selected = [dict(row) for row in record["selected"]]
        labels = population_substitution_teacher(
            record["relations"], selected, denominators, num_predicates, depth
        )
        for row in record["selected"]:
            choices = [action for action in labels if int(action["row"]) == int(row["row"])]
            best = max(
                choices,
                key=lambda action: (
                    float(action["delta_mr"]),
                    float(action["delta_r"]),
                    -int(action["predicate"]),
                ),
            )
            if float(best["delta_mr"]) > 0.0:
                selected = apply_action(selected, row, int(best["predicate"]))
        current = dict(record)
        current["selected"] = selected
        repaired.append(current)
        accounts.append(account_repairs(record["relations"], record["selected"], selected))
    return repaired, merge_repair_accounts(accounts)


def metric(records: list[dict], num_predicates: int, budget: int) -> dict:
    result = evaluate_population(records, num_predicates)
    return {"mR": float(result[f"mR@{budget}"]), "R": float(result[f"R@{budget}"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--parity-actions", type=int, default=256)
    parser.add_argument("--trials-per-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with args.carrier.open("rb") as stream:
        document = pickle.load(stream)
    num_predicates = len(document["predicate_classes"])
    records = compact_records(document, args.budget)
    repaired, accounting = exact_oracle(records, num_predicates, args.depth)
    baseline_metrics = metric(records, num_predicates, args.budget)
    oracle_metrics = metric(repaired, num_predicates, args.budget)
    parity = single_action_parity(
        records, num_predicates, args.depth, args.parity_actions, args.seed
    )
    additivity = distinct_row_additivity(
        records,
        num_predicates,
        args.depth,
        (2, 3, 5, 10),
        args.trials_per_size,
        args.seed,
    )
    tolerance = 1e-10
    gate_pass = all(
        value <= tolerance
        for value in (
            parity["max_abs_mr_residual"],
            parity["max_abs_r_residual"],
            additivity["max_abs_mr_residual"],
            additivity["max_abs_r_residual"],
        )
    )
    output = {
        "schema_version": 1,
        "name": "p1.7_metric_exact_surgery_gate0",
        "carrier": str(args.carrier),
        "images": len(records),
        "num_predicates": num_predicates,
        "budget": int(args.budget),
        "depth": int(args.depth),
        "results": {
            "baseline": baseline_metrics,
            "population_exact_oracle": oracle_metrics,
            "delta_mR_pp": 100.0 * (oracle_metrics["mR"] - baseline_metrics["mR"]),
            "delta_R_pp": 100.0 * (oracle_metrics["R"] - baseline_metrics["R"]),
        },
        "repair_accounting": accounting,
        "single_action_parity": parity,
        "distinct_row_additivity": additivity,
        "gate0": {
            "tolerance": tolerance,
            "pass": bool(gate_pass),
            "next": "S0-S7 action controls" if gate_pass else "repair evaluator/teacher",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

