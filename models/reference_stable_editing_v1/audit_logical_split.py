"""Audit logical split use of the paired population before new-method work.

This is a read-only B0 audit.  It joins the immutable OpenPSG split labels to
the physical groups that actually entered the p2.6 action carrier, then
reconstructs the registered 119/119 fit/dev cut.  A single official-test
group in this population is sufficient to stop the next stage: the paired
diagnostic population cannot be presented as an untouched clean test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True,
                        help="A0--A3 audit containing logical split labels")
    parser.add_argument("--a4", type=Path, required=True,
                        help="p2.6 action-sign artifact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = load_json(args.psg)
    audit = load_json(args.audit)
    a4 = load_json(args.a4)

    test_ids = {str(value) for value in source.get("test_image_ids", [])}
    source_rows = {}
    for row in source.get("data", []):
        source_rows.setdefault(str(row["file_name"]), []).append(row)

    # The action carrier is the authoritative population for p2.6.  It has
    # one record per action, so deduplicate by physical file_name.
    carrier_files = sorted({str(row["file_name"]) for row in a4["actions"]})
    audit_by_file = {str(row["file_name"]): row for row in audit["groups"]}
    missing_from_audit = sorted(set(carrier_files) - set(audit_by_file))

    records = []
    for file_name in carrier_files:
        rows = source_rows.get(file_name, [])
        row_ids = [str(row["image_id"]) for row in rows]
        labels = ["test" if image_id in test_ids else "train" for image_id in row_ids]
        # The previous audit is retained as an independent cross-check.  Do
        # not silently repair disagreements: surface them in the artifact.
        prior = audit_by_file.get(file_name, {})
        prior_labels = [str(value) for value in prior.get("split", [])]
        records.append({
            "file_name": file_name,
            "image_ids": row_ids,
            "logical_splits": labels,
            "prior_audit_splits": prior_labels,
            "split_consistent": labels == prior_labels,
            "logical_split": "test" if "test" in labels else "train",
        })

    # run_robust_simple_controls.py sorts its mapped groups and cuts at 119.
    fit_records = records[:119]
    dev_records = records[119:]

    def split_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "groups": len(items),
            "train": sum(item["logical_split"] == "train" for item in items),
            "test": sum(item["logical_split"] == "test" for item in items),
        }

    test_records = [item for item in records if item["logical_split"] == "test"]
    train_records = [item for item in records if item["logical_split"] == "train"]
    inconsistent = [item["file_name"] for item in records if not item["split_consistent"]]
    test_files = [item["file_name"] for item in test_records]

    report = {
        "schema_version": 1,
        "name": "reference_stable_editing_v1_paired_logical_split_audit",
        "status": "stopped_at_B0",
        "decision": "STOP_NO_METHOD_TRAINING",
        "stop_reason": (
            "the p2.6 paired action population contains official logical-test "
            "groups; it is not an untouched clean test"
        ),
        "source": {
            "psg": str(args.psg),
            "dataset_sha256": sha256(args.psg),
            "test_image_ids": len(test_ids),
            "audit_artifact": str(args.audit),
            "action_artifact": str(args.a4),
        },
        "population": {
            "carrier_physical_groups": len(records),
            "carrier_missing_from_prior_audit": missing_from_audit,
            "prior_audit_collision_groups": len(audit.get("groups", [])),
            "logical_split_counts": split_counts(records),
            "fit_cut": split_counts(fit_records),
            "dev_cut": split_counts(dev_records),
            "cut_rule": "sorted carrier file_name; first 119 fit, remaining 119 dev",
        },
        "test_use": {
            "already_inspected_test_groups": test_files,
            "already_inspected_test_group_count": len(test_files),
            "safe_teacher_groups": [item["file_name"] for item in train_records],
            "safe_teacher_group_count": len(train_records),
            "safe_calibration_groups": [],
            "safe_calibration_group_count": 0,
            "clean_confirmation_authorized": False,
        },
        "integrity": {
            "split_label_disagreements": inconsistent,
            "split_label_disagreement_count": len(inconsistent),
            "carrier_groups_have_two_source_rows": all(
                len(source_rows.get(item["file_name"], [])) == 2 for item in records
            ),
        },
        "gates": {
            "B0_logical_test_contamination": {
                "test_groups": len(test_records),
                "threshold": 0,
                "pass": len(test_records) == 0,
            },
            "method_training_authorized": False,
        },
        "groups": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "carrier_physical_groups": len(records),
        "fit": report["population"]["fit_cut"],
        "dev": report["population"]["dev_cut"],
        "test_groups": len(test_records),
        "safe_teacher_groups": len(train_records),
        "safe_calibration_groups": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
