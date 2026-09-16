import numpy as np
import pytest

from models.relation_failure_decomp_v1.run_decomposition import (
    _pair_rows,
    _retained_mapped_pairs,
    single_mpo_mapping,
)


def test_single_mpo_keeps_best_prediction_per_gt():
    gt = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    mapping, quality = single_mpo_mapping(
        pred, np.array([3, 4]), gt, np.array([3, 4])
    )
    assert mapping == {0: 0, 1: 1}
    assert quality == {0: 1.0, 1: 1.0}


def test_unmatched_high_score_pair_consumes_pair_budget():
    item = {
        "pairs": np.array([[2, 3], [0, 1]]),
        "rel_scores": np.array([[0.01, 0.9], [0.20, 0.8]]),
    }
    rows = _pair_rows(item)
    mapping = {0: 0, 1: 1}
    assert _retained_mapped_pairs(rows, mapping, 1) == set()
    assert _retained_mapped_pairs(rows, mapping, 2) == {(0, 1)}


def test_duplicate_carrier_pairs_fail_contract():
    item = {
        "pairs": np.array([[0, 1], [0, 1]]),
        "rel_scores": np.array([[0.1, 0.9], [0.2, 0.8]]),
    }
    with pytest.raises(RuntimeError, match="dedup fail"):
        _pair_rows(item)
