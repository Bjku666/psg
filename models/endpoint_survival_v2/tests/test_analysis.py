import json

import pytest

from models.endpoint_survival_v2.analyze_frontier import analyze


def write_rows(path, hits):
    rows = [
        {"file_name": "same.jpg", "gt_per_predicate": [1, 0], "hit_per_predicate": hits[0]},
        {"file_name": "same.jpg", "gt_per_predicate": [0, 1], "hit_per_predicate": hits[1]},
        {"file_name": "other.jpg", "gt_per_predicate": [1, 1], "hit_per_predicate": hits[2]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_bootstrap_groups_duplicate_rows_by_physical_file(tmp_path):
    baseline = tmp_path / "base.jsonl"
    semantic = tmp_path / "semantic.jsonl"
    competition = tmp_path / "competition.jsonl"
    write_rows(baseline, [[0, 0], [0, 0], [0, 0]])
    write_rows(semantic, [[1, 0], [0, 0], [0, 0]])
    write_rows(competition, [[0, 0], [0, 1], [0, 0]])
    result = analyze(baseline, [semantic, competition], semantic, competition, 2, 100, 0)
    assert result["physical_file_groups"] == 2
    assert result["points"][0]["observed_delta"] == pytest.approx(0.25)
    assert result["rescue_complementarity"]["both_positive_groups"] == 1
