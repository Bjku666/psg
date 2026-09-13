"""Restricted-query panoptic replay with an injectable official backend."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class ReplayResult:
    segmentation: np.ndarray
    segments: tuple[dict[str, Any], ...]
    allowed_query_ids: tuple[int, ...]

    @property
    def segment_count(self) -> int:
        return len(self.segments)

def replay_queries(
    query_pool: Sequence[Mapping[str, Any]],
    allowed_query_ids: Iterable[int],
    postprocessor: Callable[[Sequence[Mapping[str, Any]]], tuple[np.ndarray, Sequence[Mapping[str, Any]]]],
) -> ReplayResult:
    """Run the official postprocessor on a restricted raw-query pool.

    ``postprocessor`` must be the exact frozen native implementation (including
    confidence filtering, pixel competition, overlap filtering and assembly).
    Keeping it injectable makes this module usable with HF processors without
    duplicating their version-sensitive internals.
    """
    # Query ids are decoder positions.  Keep the order in the original pool
    # (rather than the order supplied by a selector): this is required for
    # bit-identical replay because native post-processing resolves ties in
    # decoder order.  Reject malformed pools early instead of silently
    # shadowing duplicate ids.
    query_pool = tuple(query_pool)
    requested = tuple(int(i) for i in allowed_query_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("allowed_query_ids contains duplicates")
    by_id: dict[int, Mapping[str, Any]] = {}
    for query in query_pool:
        if "query_id" not in query:
            raise KeyError("every raw query must contain query_id")
        query_id = int(query["query_id"])
        if query_id in by_id:
            raise ValueError(f"duplicate query_id in pool: {query_id}")
        by_id[query_id] = query
    ids_set = set(requested)
    missing = [i for i in requested if i not in by_id]
    if missing:
        raise KeyError(f"allowed query ids absent from pool: {missing[:8]}")
    selected = [query for query in query_pool if int(query["query_id"]) in ids_set]
    # ``query_pool`` may be a generator in user code; materialise once and
    # ensure all requested ids are represented in the selected order.
    if len(selected) != len(requested):
        absent = sorted(ids_set - {int(q["query_id"]) for q in selected})
        raise KeyError(f"allowed query ids absent from pool: {absent[:8]}")
    segmentation, segments = postprocessor(selected)
    seg = np.asarray(segmentation)
    if seg.ndim != 2:
        raise ValueError("postprocessor segmentation must be a 2-D array")
    normalized = tuple(dict(s) for s in segments)
    segment_queries = [int(s["query_id"]) for s in normalized if "query_id" in s]
    if not set(segment_queries).issubset(ids_set):
        raise ValueError("postprocessor emitted a segment for a disallowed query")
    # Canonicalize the public contract; backend input remains decoder order.
    return ReplayResult(seg, normalized, tuple(sorted(ids_set)))
