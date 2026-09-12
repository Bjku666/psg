from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.gssr_p0b_v1.mask_matcher import (
    load_index_mask,
    single_mpo_binary_mask_mapping,
    single_mpo_panoptic_mapping,
)


def test_binary_mask_matching_is_class_compatible_and_strict():
    gt = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    masks = np.stack([gt == 0, gt == 1])
    mapping = single_mpo_binary_mask_mapping(
        gt, masks, np.array([3, 4]), np.array([3, 9])
    )
    assert mapping.candidate_to_gt.tolist() == [0, -1]

    half = np.array([[[1, 0, 0, 0], [1, 0, 0, 0]]], dtype=bool)
    strict = single_mpo_binary_mask_mapping(
        gt, half, np.array([3, 4]), np.array([3]), threshold=0.5
    )
    assert strict.candidate_to_gt.tolist() == [-1]


def test_panoptic_mapping_checks_dimensions():
    with pytest.raises(ValueError, match="dimensions differ"):
        single_mpo_panoptic_mapping(
            np.zeros((2, 2), dtype=int),
            np.zeros((3, 2), dtype=int),
            np.array([0]),
            np.array([0]),
        )


def test_load_index_mask_uses_declared_segment_order(tmp_path: Path):
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 5
    rgb[0, 1, 0] = 7
    path = tmp_path / "mask.png"
    Image.fromarray(rgb).save(path)
    indexed = load_index_mask(path, [{"id": 7}, {"id": 5}])
    assert indexed.tolist() == [[1, 0]]

