"""Exact per-candidate marginal slot utilities on a frozen image.

This is an offline teacher generator for the future CMUD line.  It is kept in
the qualification package so the teacher definition is fixed before a model
is trained.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .legal_oracle import evidence_candidates
from .official_metric_adapter import evaluate_image


def _score(metrics: Mapping, key: str) -> float:
    value = float(metrics[key])
    return 0.0 if value != value else value  # NaN means no class/relations in a smoke image.


def marginal_slot_utilities(rows: Sequence[Mapping], relations: Sequence[Sequence[int]],
                            baseline: Sequence[Mapping], budget: int, depth: int | str,
                            num_predicates: int) -> list[dict]:
    """Return deterministic insertion/removal utility labels for each row.

    The insertion teacher keeps ``<=K`` and unique directed pairs.  At full
    occupancy it tries every legal removal and records the best exact
    image-wise mR/R change.  This is intentionally a small exhaustive oracle;
    it is used offline on fit artifacts, never in the training forward pass.
    """
    candidates = [row for row in evidence_candidates(rows, depth) if row.get("gt_pair") is not None]
    base = [dict(row) for row in baseline]
    base_metrics = evaluate_image(relations, base, num_predicates)
    selected_pairs = {tuple(map(int, row["gt_pair"])) for row in base}
    output = []
    for candidate in candidates:
        pair = tuple(map(int, candidate["gt_pair"]))
        if pair in selected_pairs:
            continue
        alternatives = []
        if len(base) < int(budget):
            alternatives.append((None, base + [dict(candidate)]))
        else:
            for index in range(len(base)):
                alternatives.append((index, base[:index] + [dict(candidate)] + base[index + 1:]))
        best = None
        for removal, selected in alternatives:
            metrics = evaluate_image(relations, selected, num_predicates)
            # Mean recall is primary; image R is a secondary guardrail.  The
            # replacement index is a stable final tie-break.
            rank_key = (_mean_recall(metrics), _score(metrics, "r"),
                        -(len(base) if removal is None else int(removal)))
            if best is None or rank_key > best[0]:
                best = (rank_key, removal, metrics)
        assert best is not None
        _, removal, metrics = best
        output.append({
            "row": int(candidate["row"]),
            "gt_pair": list(pair),
            "pred": int(candidate["pred"]),
            "evidence_depth": depth,
            "insertion_delta_r": _score(metrics, "r") - _score(base_metrics, "r"),
            "insertion_delta_mr": _mean_recall(metrics) - _mean_recall(base_metrics),
            "best_removal_row": None if removal is None else int(base[removal]["row"]),
        })
    return output


def _mean_recall(metrics: Mapping) -> float:
    values = metrics["per_predicate_recall"]
    finite = [float(value) for value in values if value == value]
    return float(sum(finite) / len(finite)) if finite else 0.0


def removal_utilities(relations: Sequence[Sequence[int]], baseline: Sequence[Mapping],
                      num_predicates: int) -> list[dict]:
    """Return exact utility lost by removing each currently selected row."""
    base = [dict(row) for row in baseline]
    base_metrics = evaluate_image(relations, base, num_predicates)
    base_mr = _mean_recall(base_metrics)
    result = []
    for index, candidate in enumerate(base):
        selected = base[:index] + base[index + 1:]
        metrics = evaluate_image(relations, selected, num_predicates)
        result.append({"row": int(candidate["row"]),
                       "removal_delta_r": _score(base_metrics, "r") - _score(metrics, "r"),
                       "removal_delta_mr": base_mr - _mean_recall(metrics)})
    return result
