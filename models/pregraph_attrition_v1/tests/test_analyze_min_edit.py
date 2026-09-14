from models.pregraph_attrition_v1.analyze_min_edit import analyze


def _point(epsilon, hit, changed=0.0):
    return {
        "epsilon": epsilon,
        "hit_per_predicate": hit,
        "gt_per_predicate": [1, 1],
        "endpoint_support": sum(hit),
        "native_endpoint_support": 0.0,
        "changed_fraction": changed,
    }


def test_analysis_recomputes_balanced_gain_and_groups_physical_files():
    rows = [
        {"file_name": "same.jpg", "curve": [_point(0.0, [0, 0]), _point(0.005, [1, 0], 0.001)]},
        {"file_name": "same.jpg", "curve": [_point(0.0, [0, 0]), _point(0.005, [0, 1], 0.002)]},
    ]
    result = analyze(rows, replicates=10)
    assert result["bootstrap"]["groups"] == 1
    assert result["curve"][1]["predicate_balanced_endpoint_support"] == 0.5
    assert result["curve"][1]["gain_vs_native"] == 0.5
    assert result["curve"][1]["budget_violations"] == 0
