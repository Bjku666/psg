"""Pure helpers for tracking GT relation endpoints through PSG stages.

The functions in this module deliberately operate on query-to-GT match
indices, rather than model tensors.  This makes the attribution contract
testable and keeps the stage runner free to use either stored RLE masks or a
one-pass model forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

STAGES = ("raw", "semantic", "competition", "admission")


@dataclass(frozen=True)
class EndpointLifecycle:
    """Availability of every GT entity at each pre-graph stage.

    ``query_ids`` stores the matching query IDs for one GT entity at each
    stage.  A stage is considered alive when its tuple is non-empty.  The
    admission stage is intentionally computed from the native selected IDs,
    not from a second pixel competition pass.
    """

    query_ids: Mapping[str, tuple[tuple[int, ...], ...]]

    @property
    def num_entities(self) -> int:
        return len(next(iter(self.query_ids.values()), ()))

    def alive(self, stage: str, gt_index: int) -> bool:
        if stage not in STAGES:
            raise KeyError(stage)
        return bool(self.query_ids[stage][gt_index])

    def death_stage(self, gt_index: int) -> str | None:
        """Return the first stage at which an endpoint is unavailable."""
        if not self.alive("raw", gt_index):
            return "raw"
        for previous, stage in zip(STAGES, STAGES[1:]):
            if self.alive(previous, gt_index) and not self.alive(stage, gt_index):
                return stage
        return None


def _query_matches(mapping: object, num_gt: int, query_ids: Sequence[int] | None = None) -> tuple[tuple[int, ...], ...]:
    candidate_to_gt = np.asarray(getattr(mapping, "candidate_to_gt"), dtype=np.int64)
    if query_ids is None:
        query_ids = range(len(candidate_to_gt))
    if len(query_ids) != len(candidate_to_gt):
        raise ValueError("query_ids and mapping have different lengths")
    result = [[] for _ in range(num_gt)]
    for query_id, gt_id in zip(query_ids, candidate_to_gt):
        if 0 <= int(gt_id) < num_gt:
            result[int(gt_id)].append(int(query_id))
    return tuple(tuple(values) for values in result)


def build_endpoint_lifecycle(
    mappings: Mapping[str, object],
    num_gt: int,
    admission_query_ids: Sequence[int] | None = None,
    query_id_orders: Mapping[str, Sequence[int]] | None = None,
) -> EndpointLifecycle:
    """Build lifecycle records from stage match results.

    ``mappings`` must contain raw, semantic, and competition match objects;
    admission may either be supplied as its own mapping or derived by filtering
    the competition mapping to ``admission_query_ids``.
    """
    required = {"raw", "semantic", "competition"}
    missing = required.difference(mappings)
    if missing:
        raise ValueError(f"missing lifecycle mappings: {sorted(missing)}")
    query_id_orders = query_id_orders or {}
    query_ids = {
        stage: _query_matches(mappings[stage], num_gt, query_id_orders.get(stage))
        for stage in required
    }
    if "admission" in mappings:
        query_ids["admission"] = _query_matches(
            mappings["admission"], num_gt, query_id_orders.get("admission")
        )
    else:
        allowed = {int(value) for value in (admission_query_ids or ())}
        query_ids["admission"] = tuple(
            tuple(query_id for query_id in values if query_id in allowed)
            for values in query_ids["competition"]
        )
    return EndpointLifecycle({stage: query_ids[stage] for stage in STAGES})


def stage_transition_counts(lifecycle: EndpointLifecycle) -> dict[str, int]:
    """Count endpoint deaths attributable to each transition."""
    counts = {stage: 0 for stage in STAGES}
    for gt_index in range(lifecycle.num_entities):
        death = lifecycle.death_stage(gt_index)
        if death is not None:
            counts[death] += 1
    counts["survives"] = sum(
        lifecycle.alive("admission", gt_index)
        for gt_index in range(lifecycle.num_entities)
    )
    return counts
