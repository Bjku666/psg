"""Same-target simple controls for the paired-reference robust utility gate.

This runner deliberately keeps the carrier support fixed.  A physical image
group contributes two mapped references; an action is legal only when the
same directed pair exists in both references.  Fit labels are the conservative
paired utility ``min(u_A, u_B)`` and evaluation applies a selected action to
both references before recomputing the exact SingleMPO metric.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.counterfactual_relation_verification_v1.reference_ambiguity_audit import map_panoptic_entities
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_image, evaluate_population, mapped_candidates
from models.relation_decision_regret_v1.legal_oracle import baseline_selection, _predicate_options
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action, _native_pred


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mr(metrics: Mapping) -> float:
    values = np.asarray(metrics["per_predicate_recall"], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


def _logits(row: Mapping) -> np.ndarray:
    values = np.asarray(row["pred_scores"], dtype=float)[1:]
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _stats(row: Mapping) -> tuple[float, float]:
    values = _logits(row)
    order = np.argsort(-values, kind="stable")
    margin = float(values[order[0]] - values[order[1]]) if len(order) > 1 else 0.0
    shifted = values - float(values.max())
    prob = np.exp(shifted)
    prob /= max(float(prob.sum()), 1e-12)
    entropy = float(-(prob * np.log(np.maximum(prob, 1e-12))).sum())
    return margin, entropy


def _features(pair: Mapping, predicate: int, frequencies: np.ndarray) -> np.ndarray:
    """Compact, pair-symmetric evidence; no GT-derived feature is included."""
    a, b = pair["a"], pair["b"]
    la, lb = _logits(a), _logits(b)
    mean, diff = (la + lb) / 2.0, np.abs(la - lb)
    ma, ea = _stats(a); mb, eb = _stats(b)
    na, nb = _native_pred(a), _native_pred(b)
    onehot = np.zeros(2 * len(mean), dtype=float)
    onehot[na] = 1.0; onehot[len(mean) + nb] = 1.0
    p = int(predicate)
    extras = np.asarray([
        float(a.get("score", 0.0)), float(b.get("score", 0.0)),
        min(float(a.get("score", 0.0)), float(b.get("score", 0.0))),
        float(mean[p] - mean[na]), float(mean[p] - mean[nb]),
        ma, mb, min(ma, mb), ea, eb, max(ea, eb),
        np.log1p(float(frequencies[p])), np.log1p(float(frequencies[na])),
        np.log1p(float(frequencies[nb])),
    ])
    candidate = np.zeros(len(mean), dtype=float); candidate[p] = 1.0
    return np.concatenate([mean, diff, onehot, candidate, extras]).astype(np.float32)


def _utility(relations, baseline, row, predicate, n):
    before = evaluate_image(relations, baseline, n)
    after = evaluate_image(relations, apply_action(baseline, row, predicate), n)
    return mr(after) - mr(before), float(after["r"]) - float(before["r"])


def _groups(psg: dict, predictions: dict[str, Mapping], panoptic_root: Path, n: int) -> list[dict]:
    by_file = collections.defaultdict(list)
    for entry in psg["data"]:
        by_file[entry["file_name"]].append(entry)
    groups = []
    for file_name, entries in sorted(by_file.items()):
        if len(entries) != 2:
            continue
        ea, eb = entries
        mapping, _, mode = map_panoptic_entities(ea, eb, panoptic_root)
        if mode != "mask" or len(mapping) != len(ea.get("annotations", [])):
            continue
        inv = {v: k for k, v in mapping.items()}
        pa, pb = predictions.get(str(ea["image_id"])), predictions.get(str(eb["image_id"]))
        if pa is None or pb is None:
            continue
        ca, _ = mapped_candidates(ea, pa, panoptic_root)
        cb, _ = mapped_candidates(eb, pb, panoptic_root)
        for row in cb:
            if row.get("gt_pair") is not None:
                row["gt_pair"] = tuple(inv[int(v)] for v in row["gt_pair"])
        ba, bb = baseline_selection(ca, 50), baseline_selection(cb, 50)
        aa = {tuple(x["gt_pair"]): x for x in ba if x.get("gt_pair") is not None}
        ab = {tuple(x["gt_pair"]): x for x in bb if x.get("gt_pair") is not None}
        rel_a = [tuple(map(int, x)) for x in ea.get("relations", [])]
        rel_b = [(inv[int(s)], inv[int(o)], int(p)) for s, o, p in eb.get("relations", [])
                 if int(s) in inv and int(o) in inv]
        actions = []
        for key in sorted(set(aa) & set(ab)):
            xa, xb = aa[key], ab[key]
            options = sorted(set(_predicate_options(xa, 3)) | set(_predicate_options(xb, 3)))
            for predicate in options:
                if int(predicate) in {_native_pred(xa), _native_pred(xb)}:
                    continue
                ua, ra = _utility(rel_a, ba, xa, predicate, n)
                ub, rb = _utility(rel_b, bb, xb, predicate, n)
                actions.append({"key": key, "a": xa, "b": xb, "predicate": int(predicate),
                                "u_a": ua, "u_b": ub, "r_a": ra, "r_b": rb,
                                "target": min(ua, ub)})
        groups.append({"file_name": file_name, "relations_a": rel_a, "relations_b": rel_b,
                       "baseline_a": ba, "baseline_b": bb, "actions": actions})
    return groups


def _frequencies(groups: Sequence[Mapping], n: int) -> np.ndarray:
    counts = np.zeros(n, dtype=float)
    for group in groups:
        for relations in (group["relations_a"], group["relations_b"]):
            for _, _, p in relations:
                if 0 <= int(p) < n:
                    counts[int(p)] += 1
    return counts


def _best_by_score(group, scorer):
    choices = collections.defaultdict(list)
    for action in group["actions"]:
        choices[action["key"]].append((float(scorer(action)), action))
    selected = []
    for values in choices.values():
        score, action = max(values, key=lambda x: (x[0], -int(x[1]["predicate"])))
        if score > 0.0:
            selected.append(action)
    return selected


def _apply(groups, decisions):
    out = []
    for group, decisions_for_group in zip(groups, decisions):
        sa, sb = list(group["baseline_a"]), list(group["baseline_b"])
        for action in decisions_for_group or ():
            sa = apply_action(sa, action["a"], action["predicate"])
            sb = apply_action(sb, action["b"], action["predicate"])
        out.extend([
            {"relations": group["relations_a"], "selected": sa, "file_name": group["file_name"],
             "bootstrap_group": group["file_name"]},
            {"relations": group["relations_b"], "selected": sb, "file_name": group["file_name"],
             "bootstrap_group": group["file_name"]},
        ])
    return out


def _fit_model(groups, frequencies, kind, seed=0):
    rows, targets = [], []
    for group in groups:
        for action in group["actions"]:
            rows.append(_features(action, action["predicate"], frequencies))
            targets.append(float(action["target"]) * 100000.0)
    if not rows:
        return None
    x, y = np.stack(rows), np.asarray(targets)
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    else:
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06,
            max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
            random_state=int(seed))
    model.fit(x, y)
    # Fit-only threshold: retain only predicted-positive actions, with a
    # deterministic no-action option.  This is deliberately conservative.
    action_list = [action for group in groups for action in group["actions"]]
    score_array = model.predict(np.stack([_features(a, a["predicate"], frequencies) for a in action_list]))
    scores = [float(x) for x in score_array]
    threshold = float(np.quantile(scores, 0.90)) if scores else float("inf")
    best = (0.0, threshold)
    targets = np.asarray([float(a["target"]) for a in action_list])
    for q in np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 41))) if scores else [threshold]:
        gain = float(targets[np.asarray(scores) > q].sum())
        if gain > best[0]:
            best = (gain, float(q))
    return model, best[1]


def _model_decisions(groups, frequencies, fitted):
    model, threshold = fitted
    decisions = []
    for group in groups:
        by_key = collections.defaultdict(list)
        for action in group["actions"]:
            score = float(model.predict(_features(action, action["predicate"], frequencies)[None])[0])
            by_key[action["key"]].append((score, action))
        selected = []
        for values in by_key.values():
            score, action = max(values, key=lambda x: (x[0], -int(x[1]["predicate"])))
            if score > threshold:
                selected.append(action)
        decisions.append(selected)
    return decisions


def _evaluate(name, groups, decisions, n, baseline_metrics=None):
    rows = _apply(groups, decisions)
    metrics = evaluate_population(rows, n)
    return {"mR": float(metrics["mR"]), "R": float(metrics["R"]),
            "delta_mR_pp": 100.0 * (float(metrics["mR"]) - baseline_metrics["mR"]) if baseline_metrics else None,
            "delta_R_pp": 100.0 * (float(metrics["R"]) - baseline_metrics["R"]) if baseline_metrics else None,
            "applied_actions": int(sum(len(x or ()) for x in decisions)),
            "intervention_rate": float(sum(len(x or ()) for x in decisions) / max(1, sum(len(g["actions"]) for g in groups)))}


def main():
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
    groups = _groups(psg, predictions, args.panoptic_root, len(psg["predicate_classes"]))
    n = len(psg["predicate_classes"])
    # Group-level split prevents the two references of one physical image from
    # crossing fit and evaluation.  The final gate is dev-only and one-shot.
    cut = len(groups) // 2
    fit, dev = groups[:cut], groups[cut:]
    frequencies = _frequencies(fit, n)
    base = _evaluate("native", dev, [None] * len(dev), n)
    robust = []
    for g in dev:
        by_key = collections.defaultdict(list)
        for a in g["actions"]:
            if a["u_a"] > 1e-12 and a["u_b"] > 1e-12:
                by_key[a["key"]].append(a)
        robust.append([max(values, key=lambda a: (min(a["u_a"], a["u_b"]),
                                                   a["u_a"] + a["u_b"], -a["predicate"]))
                       for values in by_key.values()])
    oracle = _evaluate("robust_oracle", dev, robust, n, base)
    controls = {}
    # Thresholds are fit on conservative paired labels, then evaluated once.
    for name, scorer in {
        "margin": lambda a: -max(_stats(a["a"])[0], _stats(a["b"])[0]),
        "entropy": lambda a: min(_stats(a["a"])[1], _stats(a["b"])[1]),
        "robust_target": lambda a: a["target"],
    }.items():
        values = [float(scorer(a)) for g in fit for a in g["actions"]]
        grid = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 81))) if values else [0.0]
        best = (0.0, float("inf"))
        for threshold in grid:
            gain = sum(float(a["target"]) for g in fit for a in g["actions"] if scorer(a) > threshold)
            if gain > best[0]: best = (gain, float(threshold))
        decisions = [_best_by_score(g, scorer) for g in dev]
        # Enforce the fit threshold while retaining one action per group.
        decisions = [[a for a in (values or ()) if scorer(a) > best[1]] for values in decisions]
        controls[name] = _evaluate(name, dev, decisions, n, base) | {
            "fit_threshold": best[1], "deployable": name != "robust_target",
            "role": "gt_target_sanity_check" if name == "robust_target" else "heuristic_control",
        }
    for name in ("ridge", "histgbdt"):
        fitted = _fit_model(fit, frequencies, name, args.seed)
        controls[name] = (_evaluate(name, dev, _model_decisions(dev, frequencies, fitted), n, base) | {
            "deployable": True, "role": "learned_simple_control"}) if fitted else {"status": "no_actions"}
    report = {
        "schema_version": 1,
        "name": "reference_ambiguity_audit_v1_robust_simple_controls",
        "status": "complete",
        "contract": {"unit": "physical file_name group", "fit_groups": len(fit), "dev_groups": len(dev),
                      "target": "min(u_A,u_B)", "support": "common mapped top-50 directed pairs",
                      "selection": "fit-only thresholds/models; dev evaluated once", "budget": 50},
        "dataset_sha256": sha256(args.psg),
        "checkpoint_sha256": sha256(args.checkpoint),
        "prediction_sha256": sha256(args.predictions),
        "groups": {"mapped": len(groups), "fit": len(fit), "dev": len(dev)},
        "baseline": base, "robust_oracle": oracle,
        "controls": controls,
        "gate": {},
    }
    deployable = {k: v for k, v in controls.items() if v.get("deployable", False)}
    strongest = max(deployable, key=lambda k: deployable[k].get("delta_mR_pp", -1e9))
    fraction = 100.0 * float(deployable[strongest]["delta_mR_pp"]) / float(oracle["delta_mR_pp"])
    report["gate"] = {"robust_oracle_delta_mR_pp": oracle["delta_mR_pp"],
        "strongest_deployable_simple": strongest,
        "strongest_deployable_delta_mR_pp": deployable[strongest]["delta_mR_pp"],
        "headroom_pp": oracle["delta_mR_pp"], "captured_fraction_pct": fraction,
        "thresholds": {"no_complex_if_fraction_pct_gte": 70.0,
                        "authorize_complex_if_fraction_pct_lt": 30.0},
        "decision": "AUTHORIZE_METHOD_DESIGN" if fraction < 30.0 else
                    "KEEP_SIMPLE_PRIMARY" if fraction >= 70.0 else "CONDITIONAL_METHOD"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
