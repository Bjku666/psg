"""Immutable single-source action records and strict B0 validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


REQUIRED_FIELDS = (
    "dataset_sha", "carrier_commit", "checkpoint_sha", "file_name", "image_id",
    "subject_mask_uid", "object_mask_uid", "mask_pair_hash", "pair_rank",
    "native_predicate", "candidate_predicate", "predicate_logits",
    "subject_feature", "object_feature", "union_feature", "geometry",
    "bootstrap_group", "action_id",
)


def action_key(fields: Mapping[str, Any]) -> str:
    """Return the stable key for one concrete native->candidate edit."""
    values = [
        fields["dataset_sha"], fields["checkpoint_sha"], fields["file_name"],
        fields["subject_mask_uid"], fields["object_mask_uid"],
        fields["mask_pair_hash"], fields["native_predicate"],
        fields["candidate_predicate"],
    ]
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate coverage, uniqueness, and key determinism before any fitting."""
    count = 0
    seen: set[str] = set()
    missing: list[dict[str, Any]] = []
    mismatched: list[str] = []
    duplicate: list[str] = []
    for index, record in enumerate(records):
        count += 1
        absent = [field for field in REQUIRED_FIELDS if field not in record]
        if absent:
            missing.append({"index": index, "fields": absent})
            continue
        expected = action_key(record)
        actual = str(record["action_id"])
        if expected != actual:
            mismatched.append(actual)
        if actual in seen:
            duplicate.append(actual)
        seen.add(actual)
    return {
        "records": count,
        "unique_action_ids": len(seen),
        "coverage": 1.0 if count and not missing and not mismatched and not duplicate else 0.0,
        "missing": missing,
        "mismatched_action_ids": mismatched,
        "duplicate_action_ids": duplicate,
        "pass": bool(count and not missing and not mismatched and not duplicate),
    }

