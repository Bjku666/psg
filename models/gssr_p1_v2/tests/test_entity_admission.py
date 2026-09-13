import numpy as np
import pytest

from models.gssr_p1_v2 import (
    assemble_admitted_entities, candidate_query_ids, native_candidate_query_ids,
    replay_entity_admission,
    validate_native_assembly,
)


def state():
    winner = np.array([[0, 0, 1], [0, 2, 2]], dtype=np.int16)
    candidates = []
    for q, label, score in [(0, 4, .9), (1, 5, .8), (2, 6, .7)]:
        mask = winner == q
        candidates.append({"query_id": q, "label_id": label, "score": score,
                           "won_area": int(mask.sum()), "original_area": int(mask.sum()),
                           "area_ratio": 1.0, "winning_mask": mask})
    return {"winner_map": winner, "candidates": candidates}


def test_selection_changes_admission_only_and_is_fixed_size():
    result = replay_entity_admission(state(), [2, 0])
    assert result.admitted_query_ids == (0, 2)
    assert result.segment_count == 2
    assert np.array_equal(result.segmentation, [[1, 1, 0], [1, 2, 2]])
    assert candidate_query_ids(state()["candidates"]) == (0, 1, 2)


def test_native_filter_uses_area_ratio_without_recompetition():
    s = state()
    s["candidates"][1]["area_ratio"] = .2
    assert native_candidate_query_ids(s["candidates"]) == (0, 2)
    # Candidate 1 remains replayable: admission selection itself is the action.
    assert replay_entity_admission(s, [1]).segment_count == 1


def test_winning_mask_must_match_map():
    s = state()
    s["candidates"][0]["winning_mask"] = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        replay_entity_admission(s, [0])


def test_native_assembly_checks_independent_metadata():
    s = state()
    native = replay_entity_admission(s, [0, 1, 2])
    info = [{key: value for key, value in segment.items() if key != "query_id"}
            for segment in native.segments]
    result = validate_native_assembly(native.segmentation, info, [0, 1, 2], s, [0, 1, 2])
    assert all(result.values())
    with pytest.raises(AssertionError):
        validate_native_assembly(np.zeros((2, 3), dtype=np.int32), info, [0, 1, 2], s, [0, 1, 2])
