"""P1A replay sanity checks and metric comparison helpers."""
from __future__ import annotations
from typing import Any, Callable, Mapping
import numpy as np
from .panoptic_replay import ReplayResult

def validate_native_replay(official: ReplayResult, replayed: ReplayResult, *, atol: float = 0.0) -> dict[str, Any]:
    """Check the frozen native path was reproduced exactly.

    ``atol`` is retained for callers comparing floating segmentation arrays,
    although official panoptic IDs are integer and therefore compared exactly.
    Segment records are compared as mappings (including id, label, area and
    query provenance when present).
    """
    same_shape = official.segmentation.shape == replayed.segmentation.shape
    same_seg = same_shape and (
        np.array_equal(official.segmentation, replayed.segmentation)
        if atol == 0
        else np.allclose(official.segmentation, replayed.segmentation, atol=atol, rtol=0)
    )
    same_segments = official.segments == replayed.segments
    # Selection is a set-valued contract.  Implementations may serialize ids
    # in selector order while the native backend consumes decoder order.
    same_ids = set(official.allowed_query_ids) == set(replayed.allowed_query_ids)
    result = {"segmentation_equal": bool(same_seg), "segments_equal": bool(same_segments),
              "segment_count_equal": official.segment_count == replayed.segment_count,
              "allowed_query_ids_equal": same_ids}
    if not (same_seg and same_segments and same_ids):
        raise AssertionError(f"native replay mismatch: {result}")
    return result

def compare_replays(
    native: ReplayResult,
    oracle: ReplayResult,
    score_fn: Callable[[ReplayResult], float],
    *,
    raw_oracle_gap: float | None = None,
) -> dict[str, float]:
    """Compare replay scores and report the fraction of raw oracle gap recovered."""
    if raw_oracle_gap is None:
        raise ValueError("raw_oracle_gap must be supplied for the same split; no test-gap default")
    if raw_oracle_gap < 0 or not np.isfinite(raw_oracle_gap):
        raise ValueError("raw_oracle_gap must be finite and non-negative")
    native_score, oracle_score = float(score_fn(native)), float(score_fn(oracle))
    gap = oracle_score - native_score
    fraction = gap / raw_oracle_gap if raw_oracle_gap > 0 else 0.0
    return {"native_score": native_score, "oracle_score": oracle_score,
            "delta_replay": gap, "actionable_fraction": fraction}
