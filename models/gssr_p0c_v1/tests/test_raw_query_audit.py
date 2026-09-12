import numpy as np

from models.gssr_p0c_v1.raw_query_audit import failure_decomposition


def test_supply_and_admission_failures_are_mutually_exclusive():
    relations = np.array([[0, 1, 0], [1, 2, 0], [2, 3, 1]])
    result = failure_decomposition(
        relations, supplied={0, 1, 2}, admitted={0, 1}, num_predicates=2
    )
    assert result["survives_entity_admission"].tolist() == [1, 0]
    assert result["entity_admission_failure"].tolist() == [1, 0]
    assert result["supply_failure"].tolist() == [0, 1]

