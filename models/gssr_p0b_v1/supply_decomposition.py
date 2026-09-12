"""Supply/capacity/composition decomposition for endpoint support."""

from __future__ import annotations

from collections.abc import Iterable


METRICS = (
    "predicate_balanced_endpoint_support",
    "endpoint_support_micro",
    "entity_recall_micro",
)


def decompose_summaries(
    summaries: Iterable[dict],
    score_arm: str = "score_topk",
    oracle_arm: str = "gt_set_oracle_balanced",
) -> list[dict]:
    """Compute TotalLoss = CapacityLoss + CompositionLoss at every K < M."""
    summaries = list(summaries)
    by_key = {(row["arm"], float(row["budget"])): row for row in summaries}
    full = by_key.get((score_arm, 1.0))
    if full is None:
        raise ValueError(f"missing full-pool ceiling for {score_arm} at budget 1.0")
    output: list[dict] = []
    budgets = sorted(
        budget for arm, budget in by_key if arm == score_arm and budget < 1.0
    )
    for budget in budgets:
        score = by_key[(score_arm, budget)]
        oracle = by_key.get((oracle_arm, budget))
        if oracle is None:
            raise ValueError(f"missing {oracle_arm} at budget {budget}")
        row = {"budget": budget, "full_pool_arm": score_arm, "oracle_arm": oracle_arm}
        for metric in METRICS:
            ceiling = float(full[metric])
            score_value = float(score[metric])
            oracle_value = float(oracle[metric])
            total = ceiling - score_value
            capacity = ceiling - oracle_value
            composition = oracle_value - score_value
            row[metric] = {
                "full_pool_ceiling": ceiling,
                "score": score_value,
                "oracle": oracle_value,
                "total_budget_loss": total,
                "unavoidable_capacity_loss": capacity,
                "wrong_composition_loss": composition,
                "identity_residual": total - capacity - composition,
            }
        output.append(row)
    return output

