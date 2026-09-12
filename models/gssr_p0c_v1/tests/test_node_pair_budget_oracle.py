import numpy as np

from models.gssr_p0c_v1.node_pair_budget_oracle import pair_oracle_relation_hits


def test_pair_oracle_uses_one_pair_to_support_all_predicates_on_it():
    relations = np.array([[0, 1, 0], [0, 1, 1], [1, 2, 0]])
    hits, selected = pair_oracle_relation_hits(
        {0, 1, 2}, relations, pair_budget=1, predicate_weights=np.array([0.1, 1.0])
    )
    assert selected == 1
    assert hits.tolist() == [True, True, False]


def test_pair_oracle_cannot_recover_unsupplied_endpoint():
    relations = np.array([[0, 1, 0], [1, 2, 0]])
    hits, selected = pair_oracle_relation_hits(
        {0, 1}, relations, pair_budget=10, predicate_weights=np.array([1.0])
    )
    assert selected == 1
    assert hits.tolist() == [True, False]

