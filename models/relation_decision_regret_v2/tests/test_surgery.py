import numpy as np
import torch

from models.relation_decision_regret_v2.predicate_substitution_regret import (
    apply_action, substitution_teacher, mean_recall,
)
from models.relation_decision_regret_v2.selective_repair_oracle import keep_vs_any_topl
from models.relation_decision_regret_v2.flip_accounting import account_flips
from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators,
    exact_action_utility,
    population_substitution_teacher,
)
from models.relation_decision_regret_v2.repair_accounting import account_repairs
from models.relation_decision_regret_v2.additivity_audit_v2 import (
    distinct_row_additivity,
    single_action_parity,
)
from models.relation_decision_regret_v2.simple_action_controls import (
    PriorAdjustedControl,
    ThresholdControl,
    action_features,
)
from models.relation_decision_regret_v2.metric_exact_action_ranker import (
    MetricExactActionRanker,
    pack_records,
)


def _row(index, pred=0):
    return {"row": index, "pair": (index, index + 1), "gt_pair": (index, index + 1),
            "score": 1.0 - index * .1, "pred": pred,
            "pred_scores": np.asarray([0., 2., 1., .5])}


def test_swap_keeps_fixed_support_and_only_changes_predicate():
    base = [_row(0), _row(1, 1)]
    changed = apply_action(base, base[0], 2)
    assert [x["row"] for x in changed] == [0, 1]
    assert [x["pair"] for x in changed] == [x["pair"] for x in base]
    assert changed[0]["pred"] == 2 and changed[1]["pred"] == 1
    assert changed[0]["score"] == base[0]["score"]


def test_keep_is_explicit_zero_action_and_teacher_has_same_pair_actions():
    base = [_row(0)]
    labels = substitution_teacher([(0, 1, 0)], base, 3, depth=3)
    keep = [x for x in labels if x["action"] == "KEEP"]
    assert len(keep) == 1 and keep[0]["delta_mr"] == 0.0 and keep[0]["delta_r"] == 0.0
    assert all(x["row"] == 0 for x in labels)


def test_selective_oracle_never_changes_pair_support():
    base = [_row(0), _row(1, 1)]
    selected, diagnostic = keep_vs_any_topl([(0, 1, 0), (1, 2, 1)], base, 3, depth=3)
    assert {x["row"] for x in selected} == {0, 1}
    assert len(diagnostic["decisions"]) == 2


def test_flip_accounting_is_complete():
    native = [_row(0, 1)]
    repaired = [_row(0, 0)]
    result = account_flips([(0, 1, 0)], native, repaired)
    assert result["counts"]["wrong_to_correct"] == 1
    assert result["total"] == 1


def test_population_exact_utility_matches_evaluator_and_is_additive():
    records = [
        {"image_id": "a", "file_name": "a", "relations": [(0, 1, 0), (2, 3, 1)],
         "selected": [_row(0, 1), _row(2, 1)], "budget": 2},
        {"image_id": "b", "file_name": "b", "relations": [(0, 1, 0)],
         "selected": [_row(0, 1)], "budget": 2},
    ]
    denominators = PopulationDenominators.from_records(records, 3)
    utility = exact_action_utility(records[0]["relations"], records[0]["selected"][0], 0,
                                   denominators, 3)
    assert utility["delta_mr"] == 0.25
    labels = population_substitution_teacher(records[0]["relations"], records[0]["selected"],
                                             denominators, 3, depth=3)
    assert any(action["predicate"] == 0 and action["delta_mr"] == 0.25 for action in labels)
    parity = single_action_parity(records, 3, max_actions=20)
    additive = distinct_row_additivity(records, 3, set_sizes=(2,), trials_per_size=2)
    assert parity["max_abs_mr_residual"] < 1e-12
    assert additive["max_abs_mr_residual"] < 1e-12


def test_repair_accounting_counts_only_applied_label_changes():
    native = [_row(0, 1), _row(1, 1)]
    repaired = [_row(0, 0), _row(1, 1)]
    result = account_repairs([(0, 1, 0)], native, repaired)
    assert result["applied_swaps"] == 1
    assert result["intervention_rate"] == 0.5
    assert result["transitions"]["wrong_to_correct"] == 1


def test_action_features_expose_full_logits_identities_and_hidden():
    row = _row(0, 1)
    row["pair_features"] = np.arange(384, dtype=np.float32)
    frequencies = np.asarray([10, 20, 30])
    without_hidden = action_features(row, 0, frequencies, False)
    with_hidden = action_features(row, 0, frequencies, True)
    assert len(with_hidden) - len(without_hidden) == 384
    # 3 logits + two 3-way identities + six scalar action features.
    assert len(without_hidden) == 15


def test_control_fit_returns_actual_threshold_and_nonnegative_alpha():
    group = {
        "row": _row(0, 0), "native": 0, "margin": 0.2, "entropy": 0.5,
        "frequencies": np.asarray([100, 1, 1]),
        "actions": [{"predicate": 1, "delta_mr": 0.1}],
    }
    entropy = ThresholdControl.fit([group], "S1")
    prior = PriorAdjustedControl.fit([group])
    assert entropy.threshold >= 0.0
    assert prior.alpha >= 0.0


def test_mear_pack_and_candidate_relative_forward():
    rows = [_row(0, 0), _row(2, 1)]
    for row in rows:
        row["pair_features"] = np.linspace(-1, 1, 384, dtype=np.float32)
    records = [{"relations": [(0, 1, 1), (2, 3, 2)], "selected": rows}]
    packed = pack_records(records, 3, 3, np.asarray([10, 5, 2]))
    assert packed.hidden.shape == (2, 384)
    assert packed.candidates.shape == (2, 2)
    model = MetricExactActionRanker(num_predicates=3)
    scores = model(
        torch.tensor(packed.hidden.astype(np.float32)),
        torch.tensor(packed.logits.astype(np.float32)),
        torch.tensor(packed.pair_score),
        torch.tensor(packed.native.astype(np.int64)),
        torch.tensor(packed.candidates.astype(np.int64)),
        torch.tensor(np.log1p(packed.frequencies).astype(np.float32)),
        packed.depth,
    )
    assert tuple(scores.shape) == (2, 2)
