"""Counterfactual native/oracle replay driver."""
from __future__ import annotations
from typing import Any, Mapping, Sequence, Iterable, Callable
from .panoptic_replay import ReplayResult, replay_queries

def replay_native_and_oracle(
    query_pool: Sequence[Mapping[str, Any]],
    native_ids: Iterable[int],
    oracle_ids: Iterable[int],
    postprocessor: Callable,
) -> tuple[ReplayResult, ReplayResult]:
    return (replay_queries(query_pool, native_ids, postprocessor),
            replay_queries(query_pool, oracle_ids, postprocessor))


def topk_query_ids(
    query_pool: Sequence[Mapping[str, Any]],
    k: int,
    *,
    score_key: str = "joint_score",
) -> tuple[int, ...]:
    """Deterministically select a fixed-size score baseline from raw queries.

    This helper is intentionally only a baseline (L0); relation-aware oracle
    selection should be supplied explicitly by the caller.  Ties are resolved
    by decoder/query order to match native semantics.
    """
    if k < 0 or k > len(query_pool):
        raise ValueError(f"k must lie in [0, {len(query_pool)}], got {k}")
    scored = []
    seen: set[int] = set()
    for position, query in enumerate(query_pool):
        query_id = int(query["query_id"])
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        try:
            score = float(query[score_key])
        except KeyError as exc:
            raise KeyError(f"query lacks score key {score_key!r}") from exc
        scored.append((-score, position, query_id))
    scored.sort()
    return tuple(item[2] for item in scored[:k])
