"""KEEP/SWAP selective ceilings on frozen pair support."""
from __future__ import annotations
from typing import Mapping, Sequence
from .predicate_substitution_regret import substitution_teacher, mean_recall, apply_action
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_image


def keep_vs_any_topl(relations: Sequence[Sequence[int]], baseline: Sequence[Mapping],
                     num_predicates: int, depth: int | str = 3) -> tuple[list[dict], dict]:
    """Choose KEEP or the best independent same-pair SWAP for each row.

    Under fixed pair support and fixed denominators, the image-wise utility is
    additive in hit counts, so selecting every positive-delta action is the
    exact selective oracle.  The returned diagnostics retain all labels.
    """
    labels = substitution_teacher(relations, baseline, num_predicates, depth)
    chosen = [dict(x) for x in baseline]
    decisions = []
    for row in baseline:
        candidates = [x for x in labels if int(x["row"]) == int(row["row"])]
        best = max(candidates, key=lambda x: (float(x["delta_mr"]), float(x["delta_r"]), -int(x["predicate"])))
        if float(best["delta_mr"]) > 0.0:
            chosen = apply_action(chosen, row, int(best["predicate"]))
        decisions.append({"row": int(row["row"]), "chosen": best["action"],
                         "predicate": int(best["predicate"]), "delta_mr": float(best["delta_mr"]),
                         "delta_r": float(best["delta_r"])})
    base_metrics = evaluate_image(relations, baseline, num_predicates)
    oracle_metrics = evaluate_image(relations, chosen, num_predicates)
    return chosen, {"labels": labels, "decisions": decisions,
                    "baseline_mr": mean_recall(base_metrics), "oracle_mr": mean_recall(oracle_metrics),
                    "baseline_r": float(base_metrics["r"]), "oracle_r": float(oracle_metrics["r"])}


def keep_vs_method(relations: Sequence[Sequence[int]], baseline: Sequence[Mapping],
                   method: Sequence[Mapping], num_predicates: int) -> list[dict]:
    """Oracle selector between two legal fixed-support selections."""
    base = evaluate_image(relations, baseline, num_predicates)
    alt = evaluate_image(relations, method, num_predicates)
    return [{"baseline_mr": mean_recall(base), "method_mr": mean_recall(alt),
             "delta_mr": mean_recall(alt) - mean_recall(base),
             "baseline_r": float(base["r"]), "method_r": float(alt["r"])}]
