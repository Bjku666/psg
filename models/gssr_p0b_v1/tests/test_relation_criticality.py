from models.gssr_p0b_v1.relation_criticality import summarize


def test_rescue_rate_increases_for_high_degree_fixture():
    rows = [
        {"score_selected": False, "balanced_oracle_selected": False, "best_candidate_score": 0.9,
         "relation_degree": 1, "thing_or_stuff": "thing", "has_rare_predicate": False},
        {"score_selected": False, "balanced_oracle_selected": True, "best_candidate_score": 0.4,
         "relation_degree": 4, "thing_or_stuff": "stuff", "has_rare_predicate": True},
    ]
    result = summarize(rows)
    assert result["rescue_by_relation_degree"]["1"]["rescue_rate_among_score_dropped"] == 0.0
    assert result["rescue_by_relation_degree"]["4+"]["rescue_rate_among_score_dropped"] == 1.0
    assert result["selection_categories"]["balanced_oracle_only"] == 1

