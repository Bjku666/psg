"""Stage-wise attribution of relation-endpoint attrition before graph construction."""

from .endpoint_lifecycle import (
    STAGES,
    EndpointLifecycle,
    build_endpoint_lifecycle,
    stage_transition_counts,
)

__all__ = [
    "STAGES",
    "EndpointLifecycle",
    "build_endpoint_lifecycle",
    "stage_transition_counts",
]
