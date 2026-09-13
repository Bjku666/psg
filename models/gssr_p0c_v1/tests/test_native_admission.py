import numpy as np
import torch

from models.gssr_p0c_v1.native_admission import native_panoptic_admission, pre_admission_state


def test_pre_admission_state_uses_full_pool_and_native_assembly_matches():
    class_logits = torch.tensor([[5.0, -5.0, -5.0], [4.0, -5.0, -5.0], [-5.0, -5.0, 5.0]])
    mask_logits = torch.tensor([
        [[4.0, 4.0], [4.0, -4.0]],
        [[3.0, 3.0], [3.0, 3.0]],
        [[3.0, 3.0], [3.0, 3.0]],
    ])
    segmentation, segments = native_panoptic_admission(class_logits, mask_logits, (2, 2))
    state = pre_admission_state(class_logits, mask_logits, (2, 2))
    assert state["winner_map"].shape == (2, 2)
    assert [c["query_id"] for c in state["candidates"]] == [0, 1]
    assert np.array_equal(segmentation, [[1, 1], [1, 0]])
    assert [s["query_id"] for s in segments] == [0]
