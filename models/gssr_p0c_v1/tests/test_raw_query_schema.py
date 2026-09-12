import numpy as np

from models.gssr_p0c_v1.raw_query_schema import (
    decode_binary_mask,
    encode_binary_mask,
    query_scores,
)


def test_scores_are_independent_and_joint_is_explicit_product():
    class_logits = np.array([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]])
    mask_logits = np.array([[[2.0, -2.0]], [[0.5, 0.5]]])
    scores = query_scores(class_logits, mask_logits)
    assert scores["predicted_class"].tolist() == [0, 1]
    assert np.allclose(
        scores["joint_score"], scores["class_score"] * scores["mask_quality"]
    )
    assert not np.allclose(scores["class_score"], scores["mask_quality"])


def test_mask_rle_roundtrip():
    mask = np.array([[0, 1, 1], [1, 0, 0]], dtype=bool)
    assert np.array_equal(decode_binary_mask(encode_binary_mask(mask)), mask)

