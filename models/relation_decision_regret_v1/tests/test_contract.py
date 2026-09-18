from __future__ import annotations

import numpy as np

from models.relation_decision_regret_v1.legal_oracle import (
    baseline_selection,
    evidence_candidates,
    legal_oracle_selection,
)
from models.relation_decision_regret_v1.official_metric_adapter import (
    evaluate_image,
    evaluate_population,
)
from models.relation_decision_regret_v1.marginal_utility import marginal_slot_utilities, removal_utilities
from models.relation_decision_regret_v1.simple_decoders import decode_rows


def _rows():
    return [
        {"row": 0, "gt_pair": (0, 1), "score": 0.9, "pred_scores": np.array([0.1, 0.8, 0.2])},
        # Relation 1 is second-best evidence, so top-1 cannot recover it.
        {"row": 1, "gt_pair": (1, 2), "score": 0.8, "pred_scores": np.array([0.1, 0.7, 0.75])},
        {"row": 2, "gt_pair": (2, 3), "score": 0.7, "pred_scores": np.array([0.1, 0.6, 0.3])},
    ]


def test_evidence_depth_and_legal_single_pair_contract():
    rows = _rows()
    top1 = evidence_candidates(rows, 1)
    assert [row["pred"] for row in top1] == [0, 1, 0]
    oracle = legal_oracle_selection(rows, [(0, 1, 0), (1, 2, 0), (2, 3, 1)], 2, 2)
    assert len(oracle) == 2
    assert len({tuple(row["gt_pair"]) for row in oracle}) == 2
    assert any(row["pred"] == 1 for row in oracle)


def test_metric_is_imagewise_mean_recall_and_recall():
    selected = baseline_selection(_rows(), 1)
    row = evaluate_image([(0, 1, 0), (1, 2, 1)], selected, 2)
    assert row["hit_counts"].tolist() == [1, 0]
    assert row["r"] == 0.5
    population = evaluate_population([{"relations": [(0, 1, 0), (1, 2, 1)], "selected": selected,
                                      "budget": 1}], 2)
    assert population["mR@1"] == 0.5
    assert population["R@1"] == 0.5


def test_simple_controls_keep_unique_pairs():
    selected = decode_rows(_rows(), 3, "D7")
    assert len(selected) == 3
    assert len({tuple(row["gt_pair"]) for row in selected}) == 3


def test_marginal_utility_is_exact_and_legal():
    rows = _rows()
    baseline = baseline_selection(rows, 1)
    utilities = marginal_slot_utilities(rows, [(0, 1, 0), (1, 2, 1)], baseline, 1, 2, 2)
    assert utilities
    assert all("best_removal_row" in row for row in utilities)
    assert all(row["best_removal_row"] != row["row"] for row in utilities)
    removals = removal_utilities([(0, 1, 0), (1, 2, 1)], baseline, 2)
    assert len(removals) == 1
