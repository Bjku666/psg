import numpy as np
import pytest
from models.gssr_p1_v1 import (
    replay_queries, validate_native_replay, compare_replays,
    topk_query_ids, make_restricted_postprocessor,
)

def backend(qs):
    seg = np.zeros((2, 2), dtype=np.int32)
    out = []
    for n, q in enumerate(qs, 1):
        seg[n // 2:, :] = n
        out.append({"id": n, "query_id": q["query_id"], "label_id": q.get("label_id", 0)})
    return seg, out

def test_replay_restricts_and_preserves_ids():
    pool = [{"query_id": i} for i in range(3)]
    result = replay_queries(pool, [2, 0], backend)
    assert result.allowed_query_ids == (0, 2)
    assert [s["query_id"] for s in result.segments] == [0, 2]

def test_native_sanity_and_metrics():
    pool = [{"query_id": i} for i in range(2)]
    a = replay_queries(pool, [0, 1], backend)
    b = replay_queries(pool, [0, 1], backend)
    assert validate_native_replay(a, b)["segmentation_equal"]
    assert compare_replays(a, b, lambda r: r.segment_count, raw_oracle_gap=1.0)["delta_replay"] == 0

def test_missing_query_rejected():
    with pytest.raises(KeyError):
        replay_queries([{"query_id": 0}], [1], backend)


def test_replay_uses_decoder_pool_order_and_rejects_duplicate_ids():
    pool = [{"query_id": 2}, {"query_id": 0}, {"query_id": 1}]
    result = replay_queries(pool, (i for i in [0, 2]), backend)
    assert result.allowed_query_ids == (0, 2)
    with pytest.raises(ValueError):
        replay_queries(pool, [0, 0], backend)
    with pytest.raises(ValueError):
        replay_queries([{"query_id": 1}, {"query_id": 1}], [1], backend)


def test_topk_ties_follow_decoder_order_and_restricted_adapter_copies_records():
    pool = [{"query_id": 3, "joint_score": 0.5}, {"query_id": 1, "joint_score": 0.5},
            {"query_id": 2, "joint_score": 0.2}]
    assert topk_query_ids(pool, 2) == (3, 1)
    seen = []
    def mutating_backend(qs):
        qs[0]["mutated"] = True
        seen.append(qs[0])
        return np.zeros((1, 1), dtype=np.int32), []
    adapter = make_restricted_postprocessor(pool, mutating_backend)
    adapter([pool[0]])
    assert "mutated" not in pool[0]
    assert seen[0]["mutated"]
