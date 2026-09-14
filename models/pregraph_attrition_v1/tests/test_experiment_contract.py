from __future__ import annotations

import numpy as np

from models.pregraph_attrition_v1.min_edit_oracle import minimum_change_curve
from models.pregraph_attrition_v1.run_stage_decomposition import _bootstrap_deltas


def _count(file_name: str, gt: list[int], hit: list[int]) -> dict:
    return {
        "bootstrap_group": file_name,
        "num_gt_entities": 2,
        "matched_gt_entities": 2,
        "num_gt_relations": sum(gt),
        "supported_gt_relations": sum(hit),
        "gt_per_predicate": np.asarray(gt),
        "hit_per_predicate": np.asarray(hit),
    }


def test_paired_bootstrap_reports_registered_adjacent_oracle_losses() -> None:
    stages = {
        "raw": [_count("a.jpg", [1, 1], [1, 1])],
        "semantic": [_count("a.jpg", [1, 1], [1, 0])],
        "competition": [_count("a.jpg", [1, 1], [0, 0])],
        "admission": [_count("a.jpg", [1, 1], [0, 0])],
    }
    result = _bootstrap_deltas(stages, num_predicates=2, seed=0, replicates=10)
    assert result["groups"] == 1
    assert result["deltas"]["semantic_loss"]["estimate"] == 0.5
    assert result["deltas"]["competition_loss"]["estimate"] == 0.5
    assert result["deltas"]["admission_loss"]["estimate"] == 0.0


def test_bootstrap_estimate_is_full_sample_point_difference() -> None:
    stages = {
        "raw": [_count("a.jpg", [1, 0], [1, 0]), _count("b.jpg", [0, 1], [0, 1])],
        "semantic": [_count("a.jpg", [1, 0], [0, 0]), _count("b.jpg", [0, 1], [0, 1])],
        "competition": [_count("a.jpg", [1, 0], [0, 0]), _count("b.jpg", [0, 1], [0, 1])],
        "admission": [_count("a.jpg", [1, 0], [0, 0]), _count("b.jpg", [0, 1], [0, 1])],
    }
    result = _bootstrap_deltas(stages, num_predicates=2, seed=0, replicates=20)
    assert result["deltas"]["semantic_loss"]["estimate"] == 0.5
    assert "bootstrap_mean" in result["deltas"]["semantic_loss"]


def test_min_edit_optimizes_predicate_balanced_not_micro_support() -> None:
    # Predicate 0 is rare (weight 1), predicate 1 frequent (weight .1).  Adding
    # query 2 supports the rare relation; adding query 3 supports two common
    # relations.  A micro objective picks query 3, the registered objective 2.
    winner_map = np.asarray([[0, 0, 1, 1, 4, 4], [0, 0, 1, 1, -1, -1]])
    weighted = np.zeros((5, 2, 6), dtype=float)
    semantic_masks = np.asarray([
        winner_map == 0, winner_map == 1,
        np.asarray([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=bool),
        np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=bool),
        winner_map == 4,
    ])
    gt_mask = np.asarray([[0, 0, 1, 1, 4, 4], [2, 2, 3, 3, -1, -1]])
    gt_labels = np.asarray([0, 0, 0, 0, 0])
    relations = np.asarray([[0, 2, 0], [0, 3, 1], [1, 3, 1]])
    curve = minimum_change_curve(
        winner_map, weighted, semantic_masks, np.arange(5), np.zeros(5, dtype=int),
        np.asarray([0, 1, 4]), np.zeros(3, dtype=int), gt_mask, gt_labels, relations,
        np.asarray([1.0, 0.1]), (1.0,), iou_threshold=0.49,
        overlap_mask_area_threshold=0.8,
    )
    assert curve[0]["swap"]["add_query_id"] == 2
