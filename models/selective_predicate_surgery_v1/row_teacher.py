"""Build row-level exact teachers from the locked compact carrier."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

from models.relation_decision_regret_v1.legal_oracle import _predicate_options
from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators, exact_action_utility,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import _native_pred


def build_row_teacher(records: Sequence[Mapping], num_predicates: int, depth: int = 3) -> list[dict]:
    """Return one record per selected carrier row.

    Ground truth fields are retained only in this teacher object.  Consumers must
    call :func:`row_features` before fitting a model; that function never reads
    ``relations`` or ``gt_pair``.
    """
    denominators = PopulationDenominators.from_records(records, num_predicates)
    output: list[dict] = []
    for image_index, image in enumerate(records):
        for row in image.get("selected", ()):
            native = _native_pred(row)
            candidates = [int(p) for p in _predicate_options(row, depth) if int(p) != native]
            if len(candidates) != int(depth) - 1:
                raise ValueError(f"row {row.get('row')} has {len(candidates)} alternatives")
            utilities = np.asarray([
                exact_action_utility(image.get("relations", ()), row, p, denominators,
                                     num_predicates)["delta_mr"]
                for p in candidates
            ], dtype=np.float32)
            output.append({
                "image_index": int(image_index), "row_id": int(row["row"]),
                "bootstrap_group": str(image.get("bootstrap_group", image.get("file_name", image_index))),
                "native": int(native), "candidates": candidates,
                "utilities": utilities, "edit_label": bool(float(utilities.max(initial=0.0)) > 0.0),
                "best_candidate": int(np.argmax(utilities)) if len(utilities) else -1,
                "best_utility": float(max(0.0, float(utilities.max(initial=0.0)))),
                "worst_damage": float(max(0.0, float(-utilities.min(initial=0.0)))),
                "pair_features": np.asarray(row["pair_features"], dtype=np.float32).reshape(-1),
                "pred_scores": np.asarray(row["pred_scores"], dtype=np.float32)[1:],
                "pair_score": float(row.get("score", 0.0)),
            })
    return output


def assert_teacher_is_clean_for_features(rows: Sequence[Mapping]) -> None:
    """Fail loudly if a feature path accidentally carries GT fields."""
    forbidden = {"relations", "gt_pair", "edit_label", "utilities", "best_candidate",
                 "best_utility", "worst_damage"}
    for row in rows:
        if forbidden.intersection(row.keys()):
            # This check is intentionally applied to feature dictionaries, not
            # teacher dictionaries; see row_features().
            raise AssertionError("ground-truth field leaked into model features")

