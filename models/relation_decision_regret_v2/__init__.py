"""Metric-exact same-pair predicate surgery (v2).

The v1 qualification line studies slot insertion/removal.  This package keeps
that history immutable and implements the missing fixed-support action space:
KEEP or SWAP the predicate of an already selected physical pair.
"""

from .predicate_substitution_regret import (
    substitution_teacher,
    mean_recall,
    apply_action,
)
from .selective_repair_oracle import keep_vs_any_topl

__all__ = ["substitution_teacher", "mean_recall", "apply_action", "keep_vs_any_topl"]
