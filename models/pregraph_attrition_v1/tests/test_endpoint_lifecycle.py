import numpy as np

from models.pregraph_attrition_v1.endpoint_lifecycle import (
    build_endpoint_lifecycle, stage_transition_counts,
)


class Match:
    def __init__(self, values):
        self.candidate_to_gt = np.asarray(values)


def test_lifecycle_tracks_first_missing_stage_and_admission_filter():
    mappings = {
        "raw": Match([0, 1, -1]),
        "semantic": Match([0, -1]),
        "competition": Match([0, 1]),
    }
    lifecycle = build_endpoint_lifecycle(mappings, 2, admission_query_ids=[0])
    assert lifecycle.query_ids["raw"] == ((0,), (1,))
    assert lifecycle.query_ids["admission"] == ((0,), ())
    assert lifecycle.death_stage(0) is None
    assert lifecycle.death_stage(1) == "semantic"
    assert stage_transition_counts(lifecycle) == {
        "raw": 0, "semantic": 1, "competition": 0, "admission": 0, "survives": 1,
    }
