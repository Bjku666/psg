"""Closed-form action utility for the pinned population Fair-PSG metric.

The official adapter first averages each predicate recall across images that
contain that predicate, then averages the supported predicates.  With pair
support, rank and K frozen, a predicate substitution therefore has an exact
local contribution whose denominators can be computed once per population.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from models.relation_decision_regret_v2.predicate_substitution_regret import _native_pred


def _relation_counts(relations: Sequence[Sequence[int]], num_predicates: int) -> np.ndarray:
    counts = np.zeros(int(num_predicates), dtype=np.int64)
    for _, _, predicate in relations:
        predicate = int(predicate)
        if 0 <= predicate < int(num_predicates):
            counts[predicate] += 1
    return counts


def _pair_hits(relations: Sequence[Sequence[int]], row: Mapping, predicate: int) -> int:
    mapped = row.get("gt_pair")
    if mapped is None:
        return 0
    pair = tuple(map(int, mapped))
    return sum(
        1
        for subject, obj, truth_predicate in relations
        if (int(subject), int(obj)) == pair and int(truth_predicate) == int(predicate)
    )


@dataclass(frozen=True)
class PopulationDenominators:
    """Frozen denominators defining one evaluation population."""

    image_occurrences: np.ndarray
    active_predicates: int
    images: int

    @classmethod
    def from_records(
        cls, records: Sequence[Mapping], num_predicates: int
    ) -> "PopulationDenominators":
        occurrences = np.zeros(int(num_predicates), dtype=np.int64)
        for record in records:
            occurrences += (_relation_counts(record.get("relations", ()), num_predicates) > 0)
        return cls(
            image_occurrences=occurrences,
            active_predicates=int(np.count_nonzero(occurrences)),
            images=int(len(records)),
        )


def exact_action_utility(
    relations: Sequence[Sequence[int]],
    row: Mapping,
    predicate: int,
    denominators: PopulationDenominators,
    num_predicates: int,
) -> dict:
    """Return exact population mR/R deltas for one legal same-row edit."""
    native = _native_pred(row)
    predicate = int(predicate)
    if predicate == native:
        return {"delta_mr": 0.0, "delta_r": 0.0, "delta_hits": 0}

    gt_counts = _relation_counts(relations, num_predicates)
    native_hits = _pair_hits(relations, row, native)
    candidate_hits = _pair_hits(relations, row, predicate)
    delta_mr = 0.0
    active = int(denominators.active_predicates)
    if active:
        if native_hits and gt_counts[native] and denominators.image_occurrences[native]:
            delta_mr -= native_hits / (
                active * int(denominators.image_occurrences[native]) * int(gt_counts[native])
            )
        if candidate_hits and gt_counts[predicate] and denominators.image_occurrences[predicate]:
            delta_mr += candidate_hits / (
                active
                * int(denominators.image_occurrences[predicate])
                * int(gt_counts[predicate])
            )

    total_gt = int(gt_counts.sum())
    delta_hits = int(candidate_hits - native_hits)
    delta_r = (
        delta_hits / (int(denominators.images) * total_gt)
        if denominators.images and total_gt
        else 0.0
    )
    return {
        "delta_mr": float(delta_mr),
        "delta_r": float(delta_r),
        "delta_hits": delta_hits,
    }


def population_substitution_teacher(
    relations: Sequence[Sequence[int]],
    baseline: Sequence[Mapping],
    denominators: PopulationDenominators,
    num_predicates: int,
    depth: int | str = 3,
) -> list[dict]:
    """Enumerate KEEP/SWAP labels with exact population-metric utilities."""
    from models.relation_decision_regret_v1.legal_oracle import _predicate_options

    labels: list[dict] = []
    for row in baseline:
        native = _native_pred(row)
        options = [native] + [
            int(value)
            for value in _predicate_options(row, depth)
            if int(value) != native
        ]
        for rank, predicate in enumerate(options):
            utility = exact_action_utility(
                relations, row, predicate, denominators, num_predicates
            )
            labels.append(
                {
                    "row": int(row["row"]),
                    "pair": (
                        None
                        if row.get("gt_pair") is None
                        else list(map(int, row["gt_pair"]))
                    ),
                    "native_pred": int(native),
                    "predicate": int(predicate),
                    "candidate_rank": int(rank),
                    "action": "KEEP" if int(predicate) == native else "SWAP",
                    "depth": depth,
                    **utility,
                    "is_positive": bool(utility["delta_mr"] > 0.0),
                }
            )
    return labels

