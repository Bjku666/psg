from models.counterfactual_relation_verification_v1.action_schema import action_key, validate_records


def _record():
    row = {
        "dataset_sha": "d", "checkpoint_sha": "c", "file_name": "000.jpg",
        "carrier_commit": "rev", "image_id": 1, "subject_mask_uid": "s",
        "object_mask_uid": "o", "mask_pair_hash": "pair", "pair_rank": 1,
        "native_predicate": 2, "candidate_predicate": 3,
        "predicate_logits": [0.1], "subject_feature": [0.1],
        "object_feature": [0.1], "union_feature": [0.1], "geometry": [0.1],
        "bootstrap_group": "000.jpg",
    }
    row["action_id"] = action_key(row)
    return row


def test_valid_record_and_unique_key():
    result = validate_records([_record()])
    assert result["pass"]
    assert result["coverage"] == 1.0


def test_duplicate_is_rejected():
    row = _record()
    result = validate_records([row, row])
    assert not result["pass"]
    assert result["duplicate_action_ids"]

