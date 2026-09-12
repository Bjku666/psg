import numpy as np

from models.gssr_p0b_v1.run_formal_audit import bootstrap_paired_delta


def make_row(image_id, group, hit):
    return {
        "image_id": image_id,
        "bootstrap_group": group,
        "gt_per_predicate": np.array([1]),
        "hit_per_predicate": np.array([hit]),
    }


def test_bootstrap_reports_physical_image_groups():
    baseline = [make_row("a", "same.jpg", 0), make_row("b", "same.jpg", 0)]
    oracle = [make_row("a", "same.jpg", 1), make_row("b", "same.jpg", 1)]
    result = bootstrap_paired_delta(baseline, oracle, 20, 0)
    assert result["bootstrap_groups"] == 1
    assert result["bootstrap_unit"] == "physical_image_file_name"
    assert result["ci95_low"] == result["ci95_high"] == 1.0

