"""Parity audits for population-exact predicate surgery."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators,
    population_substitution_teacher,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action


def _metric(records: Sequence[Mapping], num_predicates: int) -> tuple[float, float]:
    result = evaluate_population(records, num_predicates)
    return float(result["mR"]), float(result["R"])


def single_action_parity(
    records: Sequence[Mapping],
    num_predicates: int,
    depth: int | str = 3,
    max_actions: int = 256,
    seed: int = 0,
) -> dict:
    """Compare closed-form deltas to full evaluator recomputation."""
    denominators = PopulationDenominators.from_records(records, num_predicates)
    base_mr, base_r = _metric(records, num_predicates)
    pool = []
    for image_index, record in enumerate(records):
        labels = population_substitution_teacher(
            record.get("relations", ()), record.get("selected", ()), denominators,
            num_predicates, depth,
        )
        pool.extend((image_index, action) for action in labels if action["action"] == "SWAP")
    rng = np.random.default_rng(seed)
    if len(pool) > int(max_actions):
        indices = rng.choice(len(pool), size=int(max_actions), replace=False)
        pool = [pool[int(index)] for index in indices]
    residual_mr, residual_r = [], []
    for image_index, action in pool:
        changed = [dict(record) for record in records]
        current = dict(changed[image_index])
        row = next(
            row for row in current["selected"] if int(row["row"]) == int(action["row"])
        )
        current["selected"] = apply_action(
            current["selected"], row, int(action["predicate"])
        )
        changed[image_index] = current
        mr, recall = _metric(changed, num_predicates)
        residual_mr.append((mr - base_mr) - float(action["delta_mr"]))
        residual_r.append((recall - base_r) - float(action["delta_r"]))
    return {
        "tested": len(pool),
        "max_abs_mr_residual": float(max(map(abs, residual_mr), default=0.0)),
        "max_abs_r_residual": float(max(map(abs, residual_r), default=0.0)),
    }


def distinct_row_additivity(
    records: Sequence[Mapping],
    num_predicates: int,
    depth: int | str = 3,
    set_sizes: Sequence[int] = (2, 3, 5, 10),
    trials_per_size: int = 16,
    seed: int = 0,
) -> dict:
    """Audit action sets with at most one substitution for each physical row."""
    denominators = PopulationDenominators.from_records(records, num_predicates)
    base_mr, base_r = _metric(records, num_predicates)
    actions = []
    for image_index, record in enumerate(records):
        labels = population_substitution_teacher(
            record.get("relations", ()), record.get("selected", ()), denominators,
            num_predicates, depth,
        )
        by_row: dict[int, list[Mapping]] = {}
        for action in labels:
            if action["action"] == "SWAP":
                by_row.setdefault(int(action["row"]), []).append(action)
        for row, candidates in by_row.items():
            # One representative edit per row is enough to audit separability;
            # choose deterministically, then randomize distinct row keys below.
            candidate = sorted(
                candidates,
                key=lambda item: (int(item["predicate"]), float(item["delta_mr"])),
            )[0]
            actions.append((image_index, row, candidate))

    rng = np.random.default_rng(seed)
    rows = []
    for size in set_sizes:
        size = int(size)
        if len(actions) < size:
            continue
        for _ in range(int(trials_per_size)):
            chosen_indices = rng.choice(len(actions), size=size, replace=False)
            chosen = [actions[int(index)] for index in chosen_indices]
            changed = [dict(record) for record in records]
            for image_index, row_id, action in chosen:
                current = dict(changed[image_index])
                row = next(
                    row
                    for row in current["selected"]
                    if int(row["row"]) == int(row_id)
                )
                current["selected"] = apply_action(
                    current["selected"], row, int(action["predicate"])
                )
                changed[image_index] = current
            mr, recall = _metric(changed, num_predicates)
            expected_mr = sum(float(action[2]["delta_mr"]) for action in chosen)
            expected_r = sum(float(action[2]["delta_r"]) for action in chosen)
            rows.append(
                {
                    "size": size,
                    "mr_residual": float((mr - base_mr) - expected_mr),
                    "r_residual": float((recall - base_r) - expected_r),
                }
            )
    return {
        "tested": len(rows),
        "set_sizes": sorted({int(row["size"]) for row in rows}),
        "max_abs_mr_residual": float(
            max((abs(row["mr_residual"]) for row in rows), default=0.0)
        ),
        "max_abs_r_residual": float(
            max((abs(row["r_residual"]) for row in rows), default=0.0)
        ),
        "by_size": {
            str(size): {
                "tested": sum(row["size"] == size for row in rows),
                "max_abs_mr_residual": float(
                    max(
                        (abs(row["mr_residual"]) for row in rows if row["size"] == size),
                        default=0.0,
                    )
                ),
            }
            for size in sorted({int(row["size"]) for row in rows})
        },
    }

