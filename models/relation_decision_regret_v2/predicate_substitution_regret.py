"""Exact counterfactual labels for legal same-pair predicate substitutions."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_image
from models.relation_decision_regret_v1.legal_oracle import _predicate_options


def mean_recall(metrics: Mapping) -> float:
    values = np.asarray(metrics["per_predicate_recall"], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


def _native_pred(row: Mapping) -> int:
    if "pred" in row:
        return int(row["pred"])
    scores = np.asarray(row["pred_scores"], dtype=float)
    order = [int(i) for i in np.argsort(-scores, kind="stable") if int(i) > 0]
    if not order:
        return 0
    return order[0] - 1


def _with_pred(row: Mapping, predicate: int) -> dict:
    out = dict(row)
    scores = np.asarray(row.get("pred_scores", []), dtype=float)
    out["pred"] = int(predicate)
    if scores.size > int(predicate) + 1:
        out["pred_score"] = float(scores[int(predicate) + 1])
    return out


def apply_action(baseline: Sequence[Mapping], row: Mapping, predicate: int) -> list[dict]:
    """Return baseline with exactly one selected row's predicate changed.

    The physical pair, pair score, row order, support and budget are copied
    verbatim.  A row is identified by its stable carrier ``row`` index.
    """
    target = int(row["row"])
    output = []
    found = False
    for item in baseline:
        current = dict(item)
        if int(item["row"]) == target:
            current = _with_pred(current, int(predicate))
            found = True
        output.append(current)
    if not found:
        raise ValueError("substitution row is not in the fixed baseline support")
    return output


def substitution_teacher(relations: Sequence[Sequence[int]], baseline: Sequence[Mapping],
                          num_predicates: int, depth: int | str = 3) -> list[dict]:
    """Enumerate KEEP and legal SWAP actions for every selected pair.

    Utility is exact image-wise Fair-PSG utility.  KEEP is explicitly emitted
    with zero deltas; alternatives only alter ``pred``/``pred_score``.
    """
    base = [dict(x) for x in baseline]
    base_metrics = evaluate_image(relations, base, num_predicates)
    base_mr, base_r = mean_recall(base_metrics), float(base_metrics["r"])
    labels: list[dict] = []
    for row in base:
        native = _native_pred(row)
        options = _predicate_options(row, depth)
        # Native is always legal even if an unusual depth excludes it.
        ordered = [native] + [int(p) for p in options if int(p) != native]
        for predicate in ordered:
            if predicate == native:
                metrics = base_metrics
            else:
                metrics = evaluate_image(relations, apply_action(base, row, predicate), num_predicates)
            labels.append({
                "row": int(row["row"]), "pair": list(map(int, row.get("gt_pair", row["pair"]))),
                "native_pred": int(native), "predicate": int(predicate),
                "action": "KEEP" if predicate == native else "SWAP",
                "depth": depth,
                "delta_mr": float(mean_recall(metrics) - base_mr),
                "delta_r": float(float(metrics["r"]) - base_r),
                "is_positive": bool(mean_recall(metrics) > base_mr + 1e-12),
            })
    return labels


def substitution_additivity(relations: Sequence[Sequence[int]], baseline: Sequence[Mapping],
                             actions: Sequence[Mapping], num_predicates: int,
                             max_actions: int = 3) -> dict:
    """Check pairwise/triple additivity of fixed-support edits."""
    base = [dict(x) for x in baseline]
    base_mr = mean_recall(evaluate_image(relations, base, num_predicates))
    positives = [a for a in actions if a.get("action") == "SWAP"][: int(max_actions)]
    residuals = []
    from itertools import combinations
    for size in (2, 3):
        for combo in combinations(positives, size):
            selected = base
            for action in combo:
                row = next(x for x in base if int(x["row"]) == int(action["row"]))
                selected = apply_action(selected, row, int(action["predicate"]))
            joint = mean_recall(evaluate_image(relations, selected, num_predicates)) - base_mr
            additive = sum(float(action["delta_mr"]) for action in combo)
            residuals.append({"size": size, "rows": [int(a["row"]) for a in combo],
                              "residual": float(joint - additive)})
    values = [abs(float(x["residual"])) for x in residuals]
    return {"tested": len(residuals), "max_abs_residual": max(values) if values else 0.0,
            "mean_abs_residual": float(np.mean(values)) if values else 0.0,
            "residuals": residuals}
