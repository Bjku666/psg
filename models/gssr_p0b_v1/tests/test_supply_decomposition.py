import pytest

from models.gssr_p0b_v1.supply_decomposition import decompose_summaries


def row(arm, budget, value):
    return {
        "arm": arm,
        "budget": budget,
        "predicate_balanced_endpoint_support": value,
        "endpoint_support_micro": value,
        "entity_recall_micro": value,
    }


def test_loss_identity():
    result = decompose_summaries([
        row("score_topk", 0.5, 0.4),
        row("gt_set_oracle_balanced", 0.5, 0.55),
        row("score_topk", 1.0, 0.7),
    ])[0]["predicate_balanced_endpoint_support"]
    assert result["total_budget_loss"] == pytest.approx(0.3)
    assert result["unavoidable_capacity_loss"] == pytest.approx(0.15)
    assert result["wrong_composition_loss"] == pytest.approx(0.15)
    assert result["identity_residual"] == pytest.approx(0.0)

