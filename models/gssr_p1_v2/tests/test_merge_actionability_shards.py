from models.gssr_p1_v2.merge_actionability_shards import merge_metric


def test_merge_metric_recomputes_balanced_support_from_counts():
    left = {
        "images": 1, "gt_entities": 2, "matched_gt_entities": 1,
        "gt_per_predicate": [2, 0], "hit_per_predicate": [1, 0],
        "endpoint_support_macro_image": 0.5,
    }
    right = {
        "images": 1, "gt_entities": 4, "matched_gt_entities": 3,
        "gt_per_predicate": [0, 4], "hit_per_predicate": [0, 4],
        "endpoint_support_macro_image": 1.0,
    }
    merged = merge_metric([left, right])
    assert merged["images"] == 2
    assert merged["entity_recall_micro"] == 4 / 6
    assert merged["endpoint_support_micro"] == 5 / 6
    assert merged["endpoint_support_macro_image"] == 0.75
    assert merged["predicate_balanced_endpoint_support"] == 0.75
