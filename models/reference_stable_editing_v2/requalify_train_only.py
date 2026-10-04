"""Train-only, physical-group cross-fitted reference-stability diagnostics.

This runner is diagnostic only. It never trains a new method and never reads
the official-test groups into a fit, threshold, or reported qualification set.
The cached frozen carrier is evaluated on the 222 logical-train paired groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.counterfactual_relation_verification_v1 import run_robust_simple_controls as controls


def fold_for(name: str, k: int = 5) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16) % k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psg", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--panoptic-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    psg = json.loads(args.psg.read_text())
    with args.predictions.open("rb") as stream:
        predictions = {str(x["img_id"]): x for x in pickle.load(stream)}
    test_ids = {str(x) for x in psg.get("test_image_ids", [])}
    groups_all = controls._groups(psg, predictions, args.panoptic_root,
                                  len(psg["predicate_classes"]))
    source_rows = {}
    for row in psg["data"]:
        source_rows.setdefault(row["file_name"], []).append(row)
    train_groups, test_groups = [], []
    for group in groups_all:
        rows = source_rows[group["file_name"]]
        (test_groups if any(str(row["image_id"]) in test_ids for row in rows)
         else train_groups).append(group)
    train_groups.sort(key=lambda x: x["file_name"])
    n = len(psg["predicate_classes"])

    # Direct A4 diagnostic: compare signed utility for exactly matched actions.
    counts = {"nonneutral_actions": 0, "sign_agree": 0, "sign_unstable": 0,
              "one_neutral": 0, "actions_total": 0}
    for group in train_groups:
        for action in group["actions"]:
            ua, ub = float(action["u_a"]), float(action["u_b"])
            sa = 0 if abs(ua) < 1e-12 else (1 if ua > 0 else -1)
            sb = 0 if abs(ub) < 1e-12 else (1 if ub > 0 else -1)
            counts["actions_total"] += 1
            if sa and sb:
                counts["nonneutral_actions"] += 1
                counts["sign_agree"] += int(sa == sb)
                counts["sign_unstable"] += int(sa != sb)
            elif sa or sb:
                counts["one_neutral"] += 1
    instability = (counts["sign_unstable"] / counts["nonneutral_actions"]
                   if counts["nonneutral_actions"] else None)

    # Oracle is evaluated OOF by physical group. The oracle is non-deployable;
    # it is a ceiling only, and every accepted action must help both records.
    robust_decisions = []
    for group in train_groups:
        by_key = {}
        for action in group["actions"]:
            if action["u_a"] > 1e-12 and action["u_b"] > 1e-12:
                by_key.setdefault(action["key"], []).append(action)
        robust_decisions.append([
            max(values, key=lambda a: (min(a["u_a"], a["u_b"]),
                                       a["u_a"] + a["u_b"], -a["predicate"]))
            for values in by_key.values()
        ])
    baseline = controls._evaluate("native", train_groups,
                                  [None] * len(train_groups), n)
    oracle = controls._evaluate("robust_oracle", train_groups, robust_decisions,
                                n, baseline)

    folds = {i: [g for g in train_groups if fold_for(g["file_name"]) == i]
             for i in range(5)}
    oof_decisions = {name: [None] * len(train_groups) for name in
                     ("margin", "entropy", "frequency", "ridge", "histgbdt", "logistic")}
    index = {g["file_name"]: i for i, g in enumerate(train_groups)}
    for fold in range(5):
        fit = [g for i, rows in folds.items() if i != fold for g in rows]
        evaluation = folds[fold]
        if not evaluation or not fit:
            continue
        frequencies = controls._frequencies(fit, n)
        scorers = {
            "margin": lambda a: -max(controls._stats(a["a"])[0], controls._stats(a["b"])[0]),
            "entropy": lambda a: min(controls._stats(a["a"])[1], controls._stats(a["b"])[1]),
            "frequency": lambda a: -float(np.log1p(frequencies[int(a["predicate"])])),
        }
        for name, scorer in scorers.items():
            fit_actions = [a for g in fit for a in g["actions"]]
            scores = np.asarray([float(scorer(a)) for a in fit_actions])
            targets = np.asarray([float(a["target"]) for a in fit_actions])
            grid = np.unique(np.quantile(scores, np.linspace(0, 1, 81))) if len(scores) else []
            threshold = max(grid, key=lambda t: targets[scores > t].sum()) if len(grid) else float("inf")
            for group in evaluation:
                selected = controls._best_by_score(group, scorer)
                oof_decisions[name][index[group["file_name"]]] = [
                    a for a in selected if scorer(a) > threshold]

        # Fit conventional utility regressors on fit-fold action labels. The
        # threshold is selected only on fit folds, then applied to the held fold.
        for name in ("ridge", "histgbdt"):
            fitted = controls._fit_model(fit, frequencies, name, args.seed)
            if fitted:
                dec = controls._model_decisions(evaluation, frequencies, fitted)
                for group, selected in zip(evaluation, dec):
                    oof_decisions[name][index[group["file_name"]]] = selected

        # Logistic control predicts positive-vs-nonpositive conservative utility.
        fit_actions = [a for g in fit for a in g["actions"]]
        x_fit = np.stack([controls._features(a, a["predicate"], frequencies)
                          for a in fit_actions]) if fit_actions else np.empty((0, 0))
        y_fit = np.asarray([int(a["target"] > 1e-12) for a in fit_actions])
        if len(np.unique(y_fit)) > 1:
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
            model.fit(x_fit, y_fit)
            fit_prob = model.predict_proba(x_fit)[:, 1]
            best = (0.0, 1.0)
            for threshold in np.unique(np.quantile(fit_prob, np.linspace(0, 1, 81))):
                gain = sum(float(a["target"]) for a, prob in zip(fit_actions, fit_prob)
                           if prob > threshold)
                if gain > best[0]:
                    best = gain, float(threshold)
            for group in evaluation:
                by_key = {}
                for action in group["actions"]:
                    feat = controls._features(action, action["predicate"], frequencies)[None]
                    score = float(model.predict_proba(feat)[0, 1])
                    by_key.setdefault(action["key"], []).append((score, action))
                selected = [max(rows, key=lambda item: (item[0], -item[1]["predicate"]))[1]
                            for rows in by_key.values()
                            if max(rows, key=lambda item: (item[0], -item[1]["predicate"]))[0] > best[1]]
                oof_decisions["logistic"][index[group["file_name"]]] = selected

    simple = {}
    for name, decisions in oof_decisions.items():
        # A compact pair-hidden representation was not retained in this cache;
        # do not substitute logits and call them hidden features.
        if any(x is None for x in decisions):
            simple[name] = {"status": "incomplete_oof"}
            continue
        simple[name] = controls._evaluate(name, train_groups, decisions, n, baseline)

    deployable = {k: v for k, v in simple.items()
                  if k in {"margin", "entropy", "frequency", "ridge", "histgbdt", "logistic"}
                  and v.get("delta_mR_pp") is not None}
    strongest = max(deployable, key=lambda k: deployable[k]["delta_mR_pp"]) if deployable else None
    fraction = (100.0 * deployable[strongest]["delta_mR_pp"] / oracle["delta_mR_pp"]
                if strongest and oracle["delta_mR_pp"] else None)
    report = {
        "schema_version": 1,
        "name": "reference_stable_editing_v2_train_only_requalification",
        "status": "complete_diagnostic",
        "source": {"psg": str(args.psg), "predictions": str(args.predictions),
                   "checkpoint": str(args.checkpoint)},
        "population": {"paired_groups_available": len(groups_all),
                       "official_test_quarantined": len(test_groups),
                       "train_groups": len(train_groups), "folds": 5,
                       "fold_rule": "int(sha256(file_name),16) % 5"},
        "a4_train_oof": {"counts": counts, "sign_instability": instability,
                         "gate": "KILL" if instability is not None and instability < .02 else
                                 "STRONG_PROMOTE" if instability is not None and instability >= .10 else "EXPAND"},
        "robust_oracle_train_oof": {"baseline": baseline, "robust_oracle": oracle,
                                    "delta_mR_pp": oracle["delta_mR_pp"],
                                    "gate": "PASS" if oracle["delta_mR_pp"] >= 5 else
                                            "KILL" if oracle["delta_mR_pp"] < 2 else "WEAKEN"},
        "simple_controls_train_oof": {"controls": simple, "strongest_deployable": strongest,
                                      "captured_fraction_pct": fraction,
                                      "decision": "AUTHORIZE_METHOD_DESIGN" if fraction is not None and fraction < 30 else
                                                  "KEEP_SIMPLE_PRIMARY" if fraction is not None and fraction >= 70 else
                                                  "CONDITIONAL_METHOD" if fraction is not None else "UNDEFINED",
                                      "unavailable": ["pair_hidden: absent from immutable paired_final.pkl"]},
        "method_training_authorized": False,
        "note": "Diagnostics are train-only requalification, not pristine confirmation; source provenance is a separate hard gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "population": report["population"],
                      "a4": report["a4_train_oof"],
                      "oracle_delta_mR_pp": oracle["delta_mR_pp"],
                      "simple_decision": report["simple_controls_train_oof"]["decision"]}, indent=2))


if __name__ == "__main__":
    main()
