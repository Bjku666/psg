"""Reference restricted postprocessor adapter for P1A.

Use :func:`make_restricted_postprocessor` with the frozen native callable.  The
adapter deliberately does not implement a second approximation of panoptic
assembly; this prevents silent divergence from the official path.
"""
from __future__ import annotations
from typing import Any, Callable, Mapping, Sequence

def make_restricted_postprocessor(
    query_pool: Sequence[Mapping[str, Any]],
    native_postprocessor: Callable[[Sequence[Mapping[str, Any]]], tuple[Any, Sequence[Mapping[str, Any]]]],
) -> Callable[[Sequence[Mapping[str, Any]]], tuple[Any, Sequence[Mapping[str, Any]]]]:
    pool = tuple(query_pool)
    known = {int(q["query_id"]) for q in pool}
    if len(known) != len(pool):
        raise ValueError("query_pool contains duplicate query_id values")

    def run(selected: Sequence[Mapping[str, Any]]):
        ids = [int(q["query_id"]) for q in selected]
        if len(ids) != len(set(ids)) or not set(ids).issubset(known):
            raise ValueError("invalid restricted query selection")
        # Pass fresh dictionaries so a processor that annotates/mutates query
        # records cannot contaminate subsequent native/oracle replays.
        return native_postprocessor(tuple(dict(q) for q in selected))
    return run
