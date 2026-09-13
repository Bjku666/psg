from models.pregraph_attrition_v1.near_miss_audit import summarize


def test_near_miss_summary_is_finite_for_empty_optional_margins():
    result = summarize([
        {"relation_critical": True, "competition_candidate": False,
         "original_area": 100, "won_area": 99, "area_ratio": 0.99,
         "deficit_pixels": 0, "rescue_cost": 0.0},
        {"relation_critical": False, "original_area": 40, "won_area": 0,
         "area_ratio": 0.0, "deficit_pixels": 32, "rescue_cost": 1.0},
    ])
    assert result["relation_critical_queries"] == 1
    assert result["area_ratio"]["under_0.8"] == 0
    assert result["near_miss_le_0.5pct_area"] == 1
