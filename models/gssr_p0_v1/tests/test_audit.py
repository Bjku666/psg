import numpy as np

from models.gssr_p0_v1.audit import (
    aggregate_counts,
    box_iou_xyxy,
    gt_nodewise_topk,
    gt_set_oracle,
    image_counts,
    resolve_budget,
    score_topk,
    single_mpo_candidate_mapping,
    single_mpo_mask_mapping,
)
from models.gssr_p0_v1.run_budget_audit import bootstrap_paired_delta


def fixture():
    gt_boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]])
    # Two high-score duplicates of node 0 crowd out relation endpoints 1 and 2.
    cand_boxes = np.array([[0, 0, 10, 10], [0, 0, 9, 10], [20, 0, 30, 10], [40, 0, 50, 10]])
    labels = np.zeros(3, dtype=int)
    cand_labels = np.zeros(4, dtype=int)
    scores = np.array([0.99, 0.98, 0.60, 0.50])
    relations = np.array([[0, 1, 0], [0, 2, 1]])
    mapping = single_mpo_candidate_mapping(gt_boxes, cand_boxes, labels, cand_labels)
    return mapping, scores, relations


def test_iou_and_strict_threshold():
    iou = box_iou_xyxy(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 5, 10]]))
    assert iou[0, 0] == 0.5
    mapping = single_mpo_candidate_mapping(
        np.array([[0, 0, 5, 10]]), np.array([[0, 0, 10, 10]]), np.array([1]), np.array([1])
    )
    assert mapping.candidate_to_gt.tolist() == [-1]


def test_budget_resolution():
    assert resolve_budget(10, 0.25) == 3
    assert resolve_budget(10, 5.0) == 5
    assert resolve_budget(0, 0.5) == 0


def test_mask_mapping_is_class_compatible_and_strict():
    gt_mask = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    candidate_mask = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    mapping = single_mpo_mask_mapping(
        gt_mask, candidate_mask, np.array([3, 4]), np.array([3, 9])
    )
    assert mapping.candidate_to_gt.tolist() == [0, -1]


def test_set_oracle_repairs_duplicate_crowding():
    mapping, scores, relations = fixture()
    native = score_topk(scores, 2)
    nodewise = gt_nodewise_topk(mapping, scores, relations, 3, 2)
    oracle = gt_set_oracle(mapping, scores, relations, 2)
    native_counts = image_counts(3, relations, native, mapping, 2)
    nodewise_counts = image_counts(3, relations, nodewise, mapping, 2)
    oracle_counts = image_counts(3, relations, oracle, mapping, 2)
    assert native.tolist() == [0, 1]
    assert nodewise.tolist() == [0, 1]
    assert native_counts["supported_gt_relations"] == 0
    assert nodewise_counts["supported_gt_relations"] == 0
    assert oracle_counts["supported_gt_relations"] == 1
    assert len(set(oracle)) == 2


def test_aggregate_predicate_balancing():
    mapping, scores, relations = fixture()
    oracle = gt_set_oracle(mapping, scores, relations, 2)
    row = image_counts(3, relations, oracle, mapping, 2)
    summary = aggregate_counts([row], 2)
    assert summary["endpoint_support_micro"] == 0.5
    assert summary["predicate_balanced_endpoint_support"] == 0.5


def test_paired_bootstrap_is_deterministic():
    mapping, scores, relations = fixture()
    native = image_counts(3, relations, score_topk(scores, 2), mapping, 2)
    oracle = image_counts(3, relations, gt_set_oracle(mapping, scores, relations, 2), mapping, 2)
    result = bootstrap_paired_delta([native], [oracle], 20, 0)
    assert result["ci95_low"] == result["ci95_high"] == 0.5
