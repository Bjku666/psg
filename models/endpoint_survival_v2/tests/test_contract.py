import json

import numpy as np
import pytest

from models.endpoint_survival_v2.common import (
    aggregate_official_pq, aggregate_population, official_pq_image_stats, score_ranks,
)
from models.endpoint_survival_v2.pq_support_frontier import build_frontier


def test_balanced_support_is_aggregated_over_population():
    count_rows = [
        {"num_gt_entities": 1, "matched_gt_entities": 1, "num_gt_relations": 2, "supported_gt_relations": 1,
         "gt_per_predicate": np.array([1, 1]), "hit_per_predicate": np.array([1, 0])},
        {"num_gt_entities": 1, "matched_gt_entities": 0, "num_gt_relations": 9, "supported_gt_relations": 0,
         "gt_per_predicate": np.array([9, 0]), "hit_per_predicate": np.array([0, 0])},
    ]
    result = aggregate_population(count_rows, [], [], 2)
    assert result["metrics"]["predicate_balanced_endpoint_support"] == pytest.approx(0.05)
    assert result["metrics"]["endpoint_support"] == pytest.approx(1 / 11)


def test_official_pq_is_category_averaged_not_micro_aggregated():
    rows = [
        {0: {"iou": 1.0, "tp": 1, "fp": 0, "fn": 1}},
        {80: {"iou": 0.6, "tp": 1, "fp": 0, "fn": 0}},
    ]
    result = aggregate_official_pq(rows)
    assert result["pq_th"] == pytest.approx(2 / 3)
    assert result["pq_st"] == pytest.approx(0.6)
    assert result["pq"] == pytest.approx(((2 / 3) + 0.6) / 2)


def test_official_pq_aggregation_matches_pinned_panopticapi():
    evaluation = pytest.importorskip("panopticapi.evaluation")
    rows = [
        {0: {"iou": 1.7, "tp": 2, "fp": 1, "fn": 1}},
        {0: {"iou": 0.6, "tp": 1, "fp": 0, "fn": 1}, 80: {"iou": 0.8, "tp": 1, "fp": 1, "fn": 0}},
    ]
    ours = aggregate_official_pq(rows)
    reference = evaluation.PQStat()
    for row in rows:
        for category, values in row.items():
            reference[category].iou += values["iou"]
            reference[category].tp += values["tp"]
            reference[category].fp += values["fp"]
            reference[category].fn += values["fn"]
    categories = {index: {"id": index, "isthing": int(index < 80)} for index in range(133)}
    all_metrics, _ = reference.pq_average(categories, isthing=None)
    thing_metrics, _ = reference.pq_average(categories, isthing=True)
    stuff_metrics, _ = reference.pq_average(categories, isthing=False)
    assert ours["pq"] == pytest.approx(all_metrics["pq"])
    assert ours["sq"] == pytest.approx(all_metrics["sq"])
    assert ours["rq"] == pytest.approx(all_metrics["rq"])
    assert ours["pq_th"] == pytest.approx(thing_metrics["pq"])
    assert ours["pq_st"] == pytest.approx(stuff_metrics["pq"])


def test_official_pq_removes_void_from_union():
    gt = np.array([[0, 0], [-1, -1]], dtype=np.int32)
    state = {"candidates": [{"native_keep": True, "label_id": 0, "winning_mask": np.ones((2, 2), dtype=bool)}]}
    stats = official_pq_image_stats(gt, [{"category_id": 0, "iscrowd": 0, "area": 2}], state)
    assert stats[0]["tp"] == 1
    assert stats[0]["iou"] == pytest.approx(1.0)


def test_score_rank_is_not_decoder_position():
    ranks = score_ranks([{"joint_score": 0.2}, {"joint_score": 0.9}, {"joint_score": 0.5}])
    assert ranks.tolist() == [3, 1, 2]


def test_frontier_requires_real_native_baseline(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mode": "semantic", "margin": 0, "metrics": {}}))
    with pytest.raises(ValueError, match="native"):
        build_frontier(bad, [])


def test_frontier_uses_normalized_one_point_pq_gate(tmp_path):
    metrics = {"endpoint_support": .7, "predicate_balanced_endpoint_support": .7, "pq": .5, "pq_th": .5, "pq_st": .5,
               "changed_pixel_fraction": 0., "changed_queries": 0, "semantic_eligibility_flips": 0,
               "competition_winner_flips": 0, "native_segment_count_change": 0}
    baseline = tmp_path / "native.json"
    baseline.write_text(json.dumps({"mode": "native", "margin": 0, "metrics": metrics}))
    point = tmp_path / "point.json"
    point.write_text(json.dumps({"mode": "semantic", "margin": .1, "metrics": dict(metrics, predicate_balanced_endpoint_support=.73, pq=.49)}))
    result = build_frontier(baseline, [point])
    assert result["gates"]["pq_delta_min"] == -0.01
    assert result["points"][0]["pq_gate_passed"]
    assert result["points"][0]["support_gate_passed"]
