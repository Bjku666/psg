#!/usr/bin/env python3
"""Train MEAR on full fit and evaluate its preregistered seed-0 gate on dev."""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.metric_exact_action_ranker import (
    pack_records,
    predict_scores,
    train_ranker,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action
from models.relation_decision_regret_v2.repair_accounting import (
    account_repairs,
    merge_repair_accounts,
)
from models.relation_decision_regret_v2.run_p17_validity import compact_records, exact_oracle
from models.relation_decision_regret_v2.simple_action_controls import predicate_frequencies


def metrics(records: list[dict], num_predicates: int, budget: int) -> dict:
    result = evaluate_population(records, num_predicates)
    return {"mR": float(result[f"mR@{budget}"]), "R": float(result[f"R@{budget}"])}


def bootstrap_delta(left: list[dict], right: list[dict], num_predicates: int,
                    repeats: int = 500, seed: int = 0) -> dict:
    """Grouped physical-file bootstrap for right-minus-left mR."""
    left_rows = evaluate_population(left, num_predicates)["per_image"]
    right_rows = evaluate_population(right, num_predicates)["per_image"]
    groups = sorted({str(row["bootstrap_group"]) for row in left})
    by_group = {
        group: [index for index, row in enumerate(left)
                if str(row["bootstrap_group"]) == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)

    def value(rows, indices):
        array = np.stack([rows[index]["per_predicate_recall"] for index in indices])
        means = [float(np.mean(column[np.isfinite(column)]))
                 for column in array.T if np.isfinite(column).any()]
        return float(np.mean(means))

    all_indices = list(range(len(left)))
    observed = (value(right_rows, all_indices) - value(left_rows, all_indices)) * 100.0
    samples = []
    for _ in range(int(repeats)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = [index for group in sampled for index in by_group[str(group)]]
        samples.append((value(right_rows, indices) - value(left_rows, indices)) * 100.0)
    return {
        "estimate_pp": float(observed),
        "ci95_pp": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "groups": len(groups),
        "replicates": int(repeats),
    }


def decode(records: list[dict], packed, scores: np.ndarray) -> tuple[list[dict], dict]:
    selected = [[dict(row) for row in record["selected"]] for record in records]
    by_row = [
        {int(row["row"]): row for row in image_rows}
        for image_rows in selected
    ]
    best = np.argmax(scores, axis=1)
    best_score = scores[np.arange(len(scores)), best]
    for index in np.flatnonzero(best_score > 0.0):
        image_index = int(packed.image_index[index])
        row = by_row[image_index][int(packed.row_id[index])]
        predicate = int(packed.candidates[index, best[index]])
        row["pred"] = predicate
        raw_scores = np.asarray(row.get("pred_scores", ()))
        if raw_scores.size > predicate + 1:
            row["pred_score"] = float(raw_scores[predicate + 1])
    output, accounts = [], []
    for record, repaired in zip(records, selected):
        current = dict(record)
        current["selected"] = repaired
        output.append(current)
        accounts.append(account_repairs(record["relations"], record["selected"], repaired))
    return output, merge_repair_accounts(accounts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-carrier", required=True, type=Path)
    parser.add_argument("--dev-carrier", required=True, type=Path)
    parser.add_argument("--simple-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decisions-output", type=Path, default=None)
    parser.add_argument("--simple-decisions", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
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
    fit_data = pack_records(fit_records, num_predicates, args.depth, frequencies)
    fit_group_count = len(fit_data)
    del fit_document, fit_records
    gc.collect()
    model, history = train_ranker(
        fit_data,
        num_predicates,
        args.device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_predicates": num_predicates,
            "depth": int(args.depth),
            "frequencies": frequencies,
            "seed": int(args.seed),
        },
        args.checkpoint,
    )
    del fit_data
    gc.collect()
    dev_data = pack_records(dev_records, num_predicates, args.depth, frequencies)
    scores = predict_scores(model, dev_data, args.device, args.batch_size * 2)
    repaired, accounting = decode(dev_records, dev_data, scores)
    oracle_records, oracle_accounting = exact_oracle(dev_records, num_predicates, args.depth)
    native = metrics(dev_records, num_predicates, args.budget)
    mear = metrics(repaired, num_predicates, args.budget)
    oracle = metrics(oracle_records, num_predicates, args.budget)
    simple_document = json.loads(args.simple_results.read_text())
    strongest = str(simple_document["gate2"]["strongest"])
    simple = simple_document["controls"][strongest]
    gain_vs_simple = 100.0 * (mear["mR"] - float(simple["mR"]))
    residual = oracle["mR"] - float(simple["mR"])
    residual_capture = (mear["mR"] - float(simple["mR"])) / residual if residual > 0 else 0.0
    r_guardrail = 100.0 * (mear["R"] - native["R"])
    gate_pass = gain_vs_simple >= 1.5 and residual_capture >= 0.20 and r_guardrail >= -0.5
    bootstrap = None
    if args.simple_decisions is not None:
        with args.simple_decisions.open("rb") as stream:
            simple_decisions = pickle.load(stream)
        bootstrap = bootstrap_delta(
            simple_decisions["strongest"], repaired, num_predicates,
            args.bootstrap_replicates, args.seed,
        )
    output = {
        "schema_version": 1,
        "name": "p1.7_mear_seed0",
        "contract": {
            "fit_only_training": True,
            "dev_evaluated_once": True,
            "confirm_used": False,
            "keep_score": 0.0,
            "loss": "metric-weighted action ranking + sign",
        },
        "fit_groups": int(fit_group_count),
        "dev_groups": len(dev_data),
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "history": history,
            "checkpoint": str(args.checkpoint),
        },
        "native": native,
        "strongest_simple": {"name": strongest, "mR": float(simple["mR"]), "R": float(simple["R"])},
        "mear": {**mear, "accounting": accounting},
        "oracle": {**oracle, "accounting": oracle_accounting},
        "gate3": {
            "gain_vs_simple_mR_pp": float(gain_vs_simple),
            "residual_oracle_capture": float(residual_capture),
            "delta_R_vs_native_pp": float(r_guardrail),
            "pass": bool(gate_pass),
            "decision": "inspect DARS trigger" if gate_pass else "kill MEAR",
        },
        "bootstrap_vs_simple": bootstrap,
    }
    if args.decisions_output is not None:
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.decisions_output.open("wb") as stream:
            pickle.dump({"native": dev_records, "mear": repaired, "oracle": oracle_records},
                        stream, protocol=pickle.HIGHEST_PROTOCOL)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
