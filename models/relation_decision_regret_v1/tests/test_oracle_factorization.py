import numpy as np

from models.relation_decision_regret_v1.legal_oracle import baseline_selection, legal_oracle_selection
from models.relation_decision_regret_v1.oracle_factorization import (
    additivity_check,
    fixed_pair_predicate_oracle,
)
from models.relation_decision_regret_v1.run_mpu_hpa_pilot import _hypotheses


def _rows():
    return [
        {"row": 0, "pair": (0, 1), "score": 0.9, "pred_scores": np.array([0., .8, .7, .1]), "gt_pair": (0, 1)},
        {"row": 1, "pair": (1, 2), "score": 0.8, "pred_scores": np.array([0., .9, .1, .8]), "gt_pair": (1, 2)},
        {"row": 2, "pair": (2, 3), "score": 0.7, "pred_scores": np.array([0., .7, .6, .2]), "gt_pair": (2, 3)},
    ]


def test_fixed_pair_oracle_keeps_physical_pairs():
    rows = _rows()
    relations = [(0, 1, 1), (1, 2, 2)]
    base = baseline_selection(rows, 2)
    fixed = fixed_pair_predicate_oracle(rows, relations, 2, 2)
    assert {x["pair"] for x in base} == {x["pair"] for x in fixed}
    assert {x["pred"] for x in fixed} == {1, 2}


def test_joint_oracle_is_singlempo():
    rows = _rows()
    selected = legal_oracle_selection(rows, [(0, 1, 1), (1, 2, 2)], 3, 3)
    assert len({x["pair"] for x in selected}) == len(selected)


def test_additivity_residual_is_small():
    records = [{"image_id": "x", "relations": [(0, 1, 1), (1, 2, 2)], "candidates": _rows()}]
    result = additivity_check(records, 3, budget=2, trials=64, seed=1)
    assert result["trials"] > 0
    assert result["max_abs_epsilon_mR"] < 1e-12
    assert result["max_abs_epsilon_R"] < 1e-12


def test_learner_keeps_unmapped_carrier_pairs_as_negatives():
    row = _rows()[0]
    row["gt_pair"] = None
    hypotheses = _hypotheses(row, 2, [(0, 1, 1)])
    assert len(hypotheses) == 2
    assert all(item["label"] == 0 for item in hypotheses)
