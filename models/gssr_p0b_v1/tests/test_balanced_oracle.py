import numpy as np

from models.gssr_p0_v1.audit import MatchResult, image_counts
from models.gssr_p0b_v1.balanced_oracle import (
    gt_set_oracle,
    inverse_predicate_weights,
    predicate_counts,
)


def test_balanced_oracle_differs_from_micro_when_rare_predicate_matters():
    mapping = MatchResult(np.arange(4, dtype=np.int64), np.ones(4))
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    relations = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [2, 3, 1],
    ])
    weights = inverse_predicate_weights(np.array([100, 1]))
    micro = gt_set_oracle(mapping, scores, relations, 2)
    balanced = gt_set_oracle(mapping, scores, relations, 2, weights)
    assert set(micro) == {0, 1}
    assert set(balanced) == {2, 3}
    micro_counts = image_counts(4, relations, micro, mapping, 2)
    balanced_counts = image_counts(4, relations, balanced, mapping, 2)
    assert micro_counts["supported_gt_relations"] == 3
    assert balanced_counts["hit_per_predicate"].tolist() == [0, 1]


def test_predicate_counts_excludes_self_relations():
    counts = predicate_counts(
        [np.array([[0, 1, 0], [1, 1, 1]]), np.array([[2, 3, 1]])], 3
    )
    assert counts.tolist() == [1, 1, 0]
    assert inverse_predicate_weights(counts).tolist() == [1.0, 1.0, 0.0]

