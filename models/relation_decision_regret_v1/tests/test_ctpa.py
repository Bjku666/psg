from __future__ import annotations

import numpy as np

from models.relation_decision_regret_v1.ctpa import (
    _top_predicates,
    _fit_tournament,
)


def _image():
    scores = np.zeros(5, dtype=np.float32)
    scores[1:] = [4.0, 3.0, 2.0, 1.0]
    hidden = np.arange(384, dtype=np.float32)
    return {
        "image_id": "1", "file_name": "one.jpg", "relations": [[0, 1, 2]],
        "candidates": [{"row": 0, "pair": (0, 1), "gt_pair": (0, 1),
                        "score": 0.9, "pred_scores": scores,
                        "pair_features": hidden}],
    }


def test_top_predicates_excludes_none_column():
    assert _top_predicates(_image()["candidates"][0], 3) == [0, 1, 2]


def test_tournament_is_antisymmetric():
    model, stats = _fit_tournament([_image()], 4, depth=3, seed=0, max_pairs=8)
    row = _image()["candidates"][0]
    first, second = 0, 1
    assert abs(model.margin(row, first, second) + model.margin(row, second, first)) < 1e-6
    assert stats["training_pairs"] > 0


def test_regret_aware_training_keeps_contract():
    model, stats = _fit_tournament([_image()], 4, depth=3, seed=0,
                                   max_pairs=8, regret_aware=True)
    assert stats["regret_aware"] is True
    row = _image()["candidates"][0]
    assert model.choose(row) in {0, 1, 2}
