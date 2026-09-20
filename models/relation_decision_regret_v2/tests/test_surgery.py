import numpy as np

from models.relation_decision_regret_v2.predicate_substitution_regret import (
    apply_action, substitution_teacher, mean_recall,
)
from models.relation_decision_regret_v2.selective_repair_oracle import keep_vs_any_topl
from models.relation_decision_regret_v2.flip_accounting import account_flips


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
