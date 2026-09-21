import numpy as np

from models.selective_predicate_surgery_v2.ccuv import (
    exact_top_mask,
    proposal_targets,
    select_curve_point,
    split_records,
    utility_curve,
)


def test_group_split_keeps_physical_files_together():
    records = [
        {"bootstrap_group": "same.jpg", "value": 1},
        {"bootstrap_group": "same.jpg", "value": 2},
        {"bootstrap_group": "other.jpg", "value": 3},
    ]
    split = split_records(records)
    locations = [key for key, values in split.items()
                 if any(row["bootstrap_group"] == "same.jpg" for row in values)]
    assert len(locations) == 1
    assert sum(row["bootstrap_group"] == "same.jpg"
               for row in split[locations[0]]) == 2


def test_exact_top_mask_does_not_expand_ties():
    mask = exact_top_mask(np.ones(100), 1.0)
    assert mask.sum() == 1
    assert mask[0]


def test_curve_reports_signed_utility_and_transitions():
    rows = [
        {"signed_best_utility": 0.3, "utilities": np.array([0.3, -0.2]),
         "delta_r": np.array([0.01, -0.01]),
         "transitions": ["wrong_to_correct", "correct_to_wrong"]},
        {"signed_best_utility": 0.0, "utilities": np.array([0.0, -0.4]),
         "delta_r": np.array([0.0, -0.02]),
         "transitions": ["wrong_to_wrong", "correct_to_wrong"]},
    ]
    targets = proposal_targets(rows, np.array([0, 1]))
    point = utility_curve(np.array([2.0, 1.0]), targets, rows, [100.0])[0]
    assert np.isclose(point["net_utility"], -0.1)
    assert point["rescue_precision"] == 0.5
    assert point["harm_rate"] == 0.5
    assert point["transitions"]["wrong_to_correct"] == 1
    assert point["transitions"]["correct_to_wrong"] == 1


def test_selection_obeys_r_guardrail_then_prefers_lower_q():
    curve = [
        {"q_percent": 0.5, "delta_mR_pp": 1.0, "delta_R_pp": 0.0},
        {"q_percent": 1.0, "delta_mR_pp": 2.0, "delta_R_pp": -0.6},
        {"q_percent": 2.0, "delta_mR_pp": 1.0, "delta_R_pp": 0.1},
    ]
    assert select_curve_point(curve)["q_percent"] == 0.5
