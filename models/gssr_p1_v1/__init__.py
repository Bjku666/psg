"""P1A counterfactual entity-admission replay primitives.

The package intentionally contains no learner or training code.  P1A evaluates
whether selecting a different subset of raw queries is actionable after the
official panoptic postprocessor is rerun.
"""

from .panoptic_replay import ReplayResult, replay_queries
from .replay_metrics import compare_replays, validate_native_replay
from .replay_oracle import replay_native_and_oracle, topk_query_ids
from .restricted_postprocess import make_restricted_postprocessor

__all__ = [
    "ReplayResult", "replay_queries", "compare_replays", "validate_native_replay",
    "replay_native_and_oracle", "topk_query_ids", "make_restricted_postprocessor",
]
