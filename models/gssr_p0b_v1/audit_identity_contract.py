#!/usr/bin/env python3
"""Audit OpenPSG scene-row, filename, and candidate-join identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def select_rows(psg: dict, split: str, relations_only: bool) -> list[dict]:
    test_ids = {str(value) for value in psg["test_image_ids"]}
    rows = []
    for item in psg["data"]:
        is_test = str(item["image_id"]) in test_ids
        if split == "test" and not is_test:
            continue
        if split == "train" and is_test:
            continue
        if relations_only and not item.get("relations"):
            continue
        rows.append(item)
    return rows


def audit_identity(psg: dict, candidates: dict | None, split: str, relations_only: bool) -> dict:
    rows = select_rows(psg, split, relations_only)
    test_ids = {str(value) for value in psg["test_image_ids"]}
    candidate_groups: dict[str, list[dict]] = defaultdict(list)
    if candidates is not None:
        for candidate in candidates.get("data", []):
            candidate_groups[str(candidate.get("file_name"))].append(candidate)

    row_records = []
    filename_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        filename = str(row.get("file_name"))
        record = {
            "image_id": str(row["image_id"]),
            "file_name": filename,
            "split": "test" if str(row["image_id"]) in test_ids else "train",
            "relations": len(row.get("relations", [])),
            "annotations": len(row.get("annotations", [])),
            "relation_digest": stable_digest(row.get("relations", [])),
            "annotation_digest": stable_digest(row.get("annotations", [])),
            "pan_seg_file_name": row.get("pan_seg_file_name"),
            "candidate_rows_for_filename": len(candidate_groups.get(filename, [])),
        }
        row_records.append(record)
        filename_groups[filename].append(record)

    duplicate_groups = []
    for filename, group in sorted(filename_groups.items()):
        if len(group) <= 1:
            continue
        duplicate_groups.append({
            "file_name": filename,
            "row_count": len(group),
            "image_ids": [row["image_id"] for row in group],
            "relations_per_row": [row["relations"] for row in group],
            "annotations_per_row": [row["annotations"] for row in group],
            "identical_annotations": len({row["annotation_digest"] for row in group}) == 1,
            "identical_relations": len({row["relation_digest"] for row in group}) == 1,
            "identical_panoptic_file": len({row["pan_seg_file_name"] for row in group}) == 1,
        })

    candidate_duplicate_counts = Counter(
        filename for filename, group in candidate_groups.items() if len(group) > 1
    )
    summary = {
        "schema_version": 1,
        "split": split,
        "relations_only": relations_only,
        "scene_rows": len(row_records),
        "unique_filenames": len(filename_groups),
        "duplicate_filename_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(group["row_count"] for group in duplicate_groups),
        "extra_scene_rows_from_duplicate_filenames": len(row_records) - len(filename_groups),
        "duplicate_groups_have_identical_annotations": all(
            group["identical_annotations"] for group in duplicate_groups
        ),
        "duplicate_groups_have_distinct_relations": all(
            not group["identical_relations"] for group in duplicate_groups
        ),
        "candidate_rows": sum(len(group) for group in candidate_groups.values()),
        "candidate_unique_filenames": len(candidate_groups),
        "candidate_duplicate_filename_groups": len(candidate_duplicate_counts),
        "missing_candidate_filenames": sorted(
            filename for filename in filename_groups if filename not in candidate_groups
        ) if candidates is not None else [],
        "evaluation_population": "scene_row",
        "bootstrap_sampling_unit": "physical_image_file_name",
    }
    return {"summary": summary, "rows": row_records, "duplicate_groups": duplicate_groups}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--include-zero-relation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    with args.psg.open() as stream:
        psg = json.load(stream)
    candidates = None
    if args.candidates:
        with args.candidates.open() as stream:
            candidates = json.load(stream)
    report = audit_identity(psg, candidates, args.split, not args.include_zero_relation)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(report["summary"], indent=2) + "\n"
    )
    (args.output / "duplicate_groups.json").write_text(
        json.dumps(report["duplicate_groups"], indent=2) + "\n"
    )
    with (args.output / "rows.jsonl").open("w") as stream:
        for row in report["rows"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
