import numpy as np

from models.pregraph_attrition_v1.min_edit_oracle import minimum_change_curve


def test_curve_zero_budget_preserves_native_support():
    winner = np.array([[1, 1], [2, 2]], dtype=np.int32)
    weighted = np.ones((3, 2, 2), dtype=float)
    semantic_masks = np.array([[[1, 1], [0, 0]]], dtype=bool)
    rows = minimum_change_curve(
        winner, weighted, semantic_masks, np.array([0]), np.array([0]),
        np.array([1, 2]), np.array([0, 1]), np.array([[0, 0], [1, 1]]),
        np.array([0, 1]), np.array([[0, 1, 0]]), np.ones(1), (0.0,),
    )
    assert rows[0]["changed_pixels"] == 0
    assert rows[0]["swap"] is None
