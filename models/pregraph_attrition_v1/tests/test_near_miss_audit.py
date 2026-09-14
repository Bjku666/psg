from models.pregraph_attrition_v1.near_miss_audit import compare_cohorts, summarize


def test_near_miss_summary_is_finite_for_empty_optional_margins():
    result = summarize([
        {"relation_critical": True, "competition_candidate": False,
         "original_area": 100, "won_area": 99, "area_ratio": 0.99,
         "deficit_pixels": 0, "rescue_cost": 0.0},
        {"relation_critical": False, "original_area": 40, "won_area": 0,
         "area_ratio": 0.0, "deficit_pixels": 32, "rescue_cost": 1.0},
    ], relation_only=True)
    assert result["relation_critical_queries"] == 1
    assert result["area_ratio"]["under_0.8"] == 0
    assert result["near_miss_le_0.5pct_area"] == 1


def test_all_summary_retains_non_relation_control_rows():
    result = summarize([
        {"relation_critical": True, "original_area": 100, "won_area": 99,
         "area_ratio": 0.99, "deficit_pixels": 0, "rescue_cost": 0.0},
        {"relation_critical": False, "original_area": 40, "won_area": 0,
         "area_ratio": 0.0, "deficit_pixels": 32, "rescue_cost": 1.0},
    ])
    assert result["semantic_matched_queries"] == 2
    assert result["relation_critical_queries"] == 1


def test_control_comparison_bootstraps_by_physical_file():
    rows = [
        {"bootstrap_group": "a.jpg", "relation_critical": True,
         "original_area": 10, "area_ratio": 0.1, "deficit_pixels": 7, "rescue_cost": 4},
        {"bootstrap_group": "a.jpg", "relation_critical": False,
         "original_area": 10, "area_ratio": 0.5, "deficit_pixels": 3, "rescue_cost": 1},
    ]
    result = compare_cohorts(rows, replicates=10)
    assert result["groups"] == 1
    comparison = result["relation_minus_non_relation_median"]
    assert comparison["area_ratio"]["estimate"] == -0.4
    assert comparison["rescue_cost"]["estimate"] == 3.0
