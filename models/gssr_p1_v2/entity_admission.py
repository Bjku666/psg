"""Replay entity admission after full-pool pixel competition.

This module intentionally has no postprocessor call.  ``pre_admission_state``
is computed once on the complete decoder pool; all counterfactuals only choose
which mutually-exclusive winning masks become entity nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class EntityReplayResult:
    segmentation: np.ndarray
    segments: tuple[dict[str, Any], ...]
    candidate_query_ids: tuple[int, ...]
    admitted_query_ids: tuple[int, ...]

    @property
    def segment_count(self) -> int:
        return len(self.segments)


def _candidate_index(candidates: Iterable[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    index: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        query_id = int(candidate["query_id"])
        if query_id in index:
            raise ValueError(f"duplicate candidate query_id: {query_id}")
        mask = np.asarray(candidate["winning_mask"], dtype=bool)
        if mask.ndim != 2:
            raise ValueError("winning_mask must be a 2-D array")
        if int(candidate.get("won_area", int(mask.sum()))) != int(mask.sum()):
            raise ValueError(f"won_area disagrees with winning_mask for query {query_id}")
        index[query_id] = candidate
    return index


def candidate_query_ids(candidates: Iterable[Mapping[str, Any]]) -> tuple[int, ...]:
    """Return candidate ids in decoder order (the order of the records)."""
    index = _candidate_index(candidates)
    return tuple(index)


def native_candidate_query_ids(
    candidates: Iterable[Mapping[str, Any]], overlap_mask_area_threshold: float = 0.8,
) -> tuple[int, ...]:
    """Return the native admitted subset without changing pixel competition."""
    records = tuple(candidates)
    _candidate_index(records)
    return tuple(int(c["query_id"]) for c in records
                 if float(c.get("area_ratio", 0.0)) > overlap_mask_area_threshold)


def assemble_admitted_entities(
    winner_map: np.ndarray,
    candidates: Iterable[Mapping[str, Any]],
    admitted_query_ids: Iterable[int],
) -> EntityReplayResult:
    """Assemble fixed winning masks into a panoptic map.

    IDs are consumed in candidate/decoder order, while the returned public
    ``admitted_query_ids`` is canonicalized as a sorted set.  Every selected
    candidate has non-empty support, so segment count equals selection size.
    """
    winner_map = np.asarray(winner_map)
    if winner_map.ndim != 2:
        raise ValueError("winner_map must be a 2-D array")
    records = tuple(candidates)
    index = _candidate_index(records)
    requested = tuple(int(i) for i in admitted_query_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("admitted_query_ids contains duplicates")
    missing = [i for i in requested if i not in index]
    if missing:
        raise KeyError(f"admitted ids are not entity candidates: {missing[:8]}")
    selected = [c for c in records if int(c["query_id"]) in set(requested)]
    # Match the official processor's no-eligible-query sentinel.  With a
    # non-empty candidate pool, unadmitted pixels remain ordinary void zero.
    segmentation = (np.full(winner_map.shape, -1, dtype=np.int32)
                    if not records else np.zeros(winner_map.shape, dtype=np.int32))
    segments: list[dict[str, Any]] = []
    for segment_id, candidate in enumerate(selected, start=1):
        query_id = int(candidate["query_id"])
        mask = np.asarray(candidate["winning_mask"], dtype=bool)
        if mask.shape != winner_map.shape:
            raise ValueError(f"winning_mask shape mismatch for query {query_id}")
        if not np.all(winner_map[mask] == query_id):
            raise ValueError(f"winning_mask is not consistent with winner_map for query {query_id}")
        segmentation[mask] = segment_id
        segments.append({
            "id": segment_id,
            "query_id": query_id,
            "label_id": int(candidate["label_id"]),
            "score": round(float(candidate["score"]), 6),
            "was_fused": False,
        })
    return EntityReplayResult(segmentation, tuple(segments),
                              tuple(int(c["query_id"]) for c in records),
                              tuple(sorted(requested)))


def replay_entity_admission(state: Mapping[str, Any], admitted_query_ids: Iterable[int]) -> EntityReplayResult:
    """Replay a selector against a serialized ``pre_admission_state``."""
    if "winner_map" not in state or "candidates" not in state:
        raise KeyError("state requires winner_map and candidates")
    return assemble_admitted_entities(state["winner_map"], state["candidates"], admitted_query_ids)


def validate_native_assembly(
    official_segmentation: np.ndarray,
    official_segments_info: Iterable[Mapping[str, Any]],
    official_segment_query_ids: Iterable[int],
    state: Mapping[str, Any],
    native_query_ids: Iterable[int],
) -> dict[str, bool]:
    """Independently verify the stored official map and segment metadata.

    ``official_segments_info`` and query provenance are supplied separately
    from the replay result, preventing a caller from comparing a value to a
    copy of its own output.
    """
    official_info = tuple(dict(item) for item in official_segments_info)
    query_ids = tuple(int(i) for i in official_segment_query_ids)
    if len(official_info) != len(query_ids):
        raise AssertionError("official segment metadata/query provenance length differs")
    replay = replay_entity_admission(state, native_query_ids)
    expected = tuple(
        {key: value for key, value in segment.items() if key != "query_id"}
        for segment in replay.segments
    )
    # JSON records may preserve the official processor's insertion order, but
    # metadata equality is structural rather than order-sensitive.
    got = tuple(dict(item) for item in official_info)
    def metadata_equal(got_item: Mapping[str, Any], expected_item: Mapping[str, Any]) -> bool:
        if set(got_item) != set(expected_item):
            return False
        for key in got_item:
            if key == "score":
                if not np.isclose(float(got_item[key]), float(expected_item[key]), rtol=0.0, atol=1e-5):
                    return False
            elif got_item[key] != expected_item[key]:
                return False
        return True

    checks = {
        "segmentation_equal": bool(np.array_equal(np.asarray(official_segmentation), replay.segmentation)),
        "segments_info_equal": all(metadata_equal(dict(g), dict(e)) for g, e in zip(got, expected)) and len(got) == len(expected),
        "segment_query_ids_equal": query_ids == tuple(int(s["query_id"]) for s in replay.segments),
    }
    if not all(checks.values()):
        raise AssertionError(f"native fixed-competition assembly mismatch: {checks}")
    return checks
