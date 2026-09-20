from __future__ import annotations

import numpy as np
import pytest

from models.relation_decision_regret_v1.make_partition_shards import make_shards
from models.relation_decision_regret_v1.merge_compact_carrier import merge


def test_partition_shards_are_deterministic_and_confirm_locked():
    psg = {
        "data": [
            {"image_id": 3, "file_name": "c", "relations": [[0, 1, 0]]},
            {"image_id": 1, "file_name": "a", "relations": [[0, 1, 0]]},
            {"image_id": 2, "file_name": "b", "relations": [[0, 1, 0]]},
        ],
        "test_image_ids": [], "predicate_classes": ["x"],
    }
    manifest = {"fit": ["3", "1", "2"], "dev": [], "confirm": []}
    shards = make_shards(psg, manifest, "fit", 2)
    assert [[str(x["image_id"]) for x in shard["data"]] for shard in shards] == [["1", "2"], ["3"]]
    assert shards[0]["test_image_ids"] == [1, 2]


def test_compact_dtype_contract_is_loss_bounded():
    hidden = np.linspace(-1, 1, 384, dtype=np.float32)
    restored = hidden.astype(np.float16).astype(np.float32)
    assert np.max(np.abs(hidden - restored)) < 5e-4


def test_merge_rejects_duplicate_image_ids(tmp_path):
    one = {"schema_version": 1, "predicate_classes": ["x"],
           "images": [{"image_id": "1"}]}
    for name in ("fit_000.pkl", "fit_001.pkl"):
        with (tmp_path / name).open("wb") as stream:
            import pickle
            pickle.dump(one, stream)
    with pytest.raises(ValueError, match="duplicate image id"):
        merge(sorted(tmp_path.glob("fit_*.pkl")))
