"""Exact micro and predicate-balanced fixed-K entity-set oracles.

The balanced objective uses split-level predicate counts.  Maximizing the sum
of ``1 / N_predicate`` for supported relation instances is exactly equivalent
to maximizing predicate-balanced endpoint support on the fixed evaluation
population.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from models.gssr_p0_v1.audit import MatchResult


def predicate_counts(relations_by_image: Sequence[np.ndarray], num_predicates: int) -> np.ndarray:
    """Return split-level non-self relation counts for every predicate."""
    counts = np.zeros(num_predicates, dtype=np.int64)
    for relations in relations_by_image:
        for subject, obj, predicate in np.asarray(relations, dtype=np.int64).reshape(-1, 3):
            if subject != obj:
                counts[int(predicate)] += 1
    return counts


def inverse_predicate_weights(counts: np.ndarray) -> np.ndarray:
    """Return 1/N_r for observed predicates and zero for absent predicates."""
    counts = np.asarray(counts, dtype=np.int64)
    return np.divide(
        1.0,
        counts,
        out=np.zeros(len(counts), dtype=np.float64),
        where=counts > 0,
    )


def oracle_gt_nodes(
    supplied: Sequence[int],
    relations: np.ndarray,
    k: int,
    predicate_weights: np.ndarray | None = None,
) -> set[int]:
    """Choose at most K supplied GT nodes with an exact weighted MILP."""
    supplied = sorted(set(int(value) for value in supplied))
    if k <= 0 or not supplied:
        return set()
    if len(supplied) <= k:
        return set(supplied)

    rels = np.asarray(relations, dtype=np.int64).reshape(-1, 3)
    usable = [
        tuple(map(int, relation))
        for relation in rels
        if int(relation[0]) != int(relation[1])
        and int(relation[0]) in supplied
        and int(relation[1]) in supplied
    ]
    if not usable:
        return set(supplied[:k])

    if predicate_weights is None:
        relation_weights = np.ones(len(usable), dtype=np.float64)
    else:
        predicate_weights = np.asarray(predicate_weights, dtype=np.float64)
        relation_weights = np.asarray(
            [predicate_weights[predicate] for _, _, predicate in usable], dtype=np.float64
        )
        if np.any(relation_weights < 0) or not np.isfinite(relation_weights).all():
            raise ValueError("predicate weights must be finite and non-negative")

    position = {gt_id: index for index, gt_id in enumerate(supplied)}
    num_nodes, num_relations = len(supplied), len(usable)
    # Binary variables are [selected_gt_node, supported_relation].
    objective = np.r_[np.zeros(num_nodes), -relation_weights]
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    rows.append(np.r_[np.ones(num_nodes), np.zeros(num_relations)])
    lower.append(-np.inf)
    upper.append(float(k))
    for relation_index, (subject, obj, _predicate) in enumerate(usable):
        for endpoint in (subject, obj):
            row = np.zeros(num_nodes + num_relations, dtype=np.float64)
            row[num_nodes + relation_index] = 1.0
            row[position[endpoint]] = -1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)

    result = milp(
        c=objective,
        integrality=np.ones(num_nodes + num_relations),
        bounds=Bounds(np.zeros(num_nodes + num_relations), np.ones(num_nodes + num_relations)),
        constraints=LinearConstraint(np.stack(rows), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact set-oracle MILP failed: {result.message}")

    chosen = {
        supplied[index]
        for index, value in enumerate(result.x[:num_nodes])
        if value > 0.5
    }
    # Zero-marginal slots are padded deterministically.  This preserves the
    # exact objective while keeping every candidate-level arm at exactly K.
    for gt_id in supplied:
        if len(chosen) >= k:
            break
        chosen.add(gt_id)
    return chosen


def gt_set_oracle(
    mapping: MatchResult,
    scores: np.ndarray,
    relations: np.ndarray,
    k: int,
    predicate_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Lift the exact GT-node solution to exactly K candidate indices."""
    if not 0 <= k <= len(scores):
        raise ValueError(f"k must lie in [0, {len(scores)}], got {k}")
    supplied = mapping.candidate_to_gt[mapping.candidate_to_gt >= 0]
    chosen_gt = oracle_gt_nodes(supplied, relations, k, predicate_weights)
    order = np.argsort(-np.asarray(scores), kind="stable")
    selected: list[int] = []
    for gt_id in sorted(chosen_gt):
        candidates = [
            int(index)
            for index in order
            if int(mapping.candidate_to_gt[index]) == gt_id
        ]
        if candidates:
            selected.append(candidates[0])
    used = set(selected)
    selected.extend(
        int(index)
        for index in order
        if int(index) not in used and len(selected) < k
    )
    if len(selected) != k or len(set(selected)) != k:
        raise AssertionError("oracle candidate lifting did not preserve the exact budget")
    return np.asarray(selected, dtype=np.int64)

