#!/usr/bin/env python3
"""Validate and safely extract the COCO panoptic val2017 ground truth."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_ARCHIVE_BYTES = 860_725_834


def selected_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Select the outer val annotation JSON and nested val mask archive."""
    selected = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {info.filename}")
        if info.filename in {
            "annotations/panoptic_val2017.json",
            "annotations/panoptic_val2017.zip",
        }:
            selected.append(info)
    return selected


def safe_nested_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe nested archive member: {info.filename}")
        if path.parts and path.parts[0] == "__MACOSX":
            continue
        if info.filename.endswith(".png") and (
            len(path.parts) != 2 or path.parts[0] != "panoptic_val2017"
        ):
            raise ValueError(f"unexpected nested PNG path: {info.filename}")
        if info.filename.endswith(".png"):
            members.append(info)
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-bytes", type=int, default=EXPECTED_ARCHIVE_BYTES)
    args = parser.parse_args()
    actual_bytes = args.archive.stat().st_size
    if actual_bytes != args.expected_bytes:
        raise ValueError(
            f"archive is incomplete or unexpected: {actual_bytes} bytes, expected {args.expected_bytes}"
        )
    target = args.output / "annotations" / "panoptic_val2017"
    target_json = args.output / "annotations" / "panoptic_val2017.json"
    if target.exists() or target_json.exists():
        raise FileExistsError("refusing to overwrite an existing panoptic val2017 extraction")
    with zipfile.ZipFile(args.archive) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"archive CRC check failed at {bad_member}")
        members = selected_members(archive)
        member_names = {info.filename for info in members}
        required = {
            "annotations/panoptic_val2017.json",
            "annotations/panoptic_val2017.zip",
        }
        if member_names != required:
            raise ValueError(
                f"unexpected outer archive contents: found {sorted(member_names)}, "
                f"required {sorted(required)}"
            )
        nested_bytes = archive.read("annotations/panoptic_val2017.zip")
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
            bad_nested_member = nested.testzip()
            if bad_nested_member is not None:
                raise ValueError(f"nested val archive CRC check failed at {bad_nested_member}")
            png_members = safe_nested_members(nested)
            png_count = len(png_members)
            if png_count != 5_000:
                raise ValueError(f"unexpected nested archive contents: {png_count} val PNGs")
            target.parent.mkdir(parents=True)
            nested.extractall(target.parent, members=png_members)
        target_json.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("annotations/panoptic_val2017.json") as source, target_json.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    report = {
        "archive": str(args.archive.resolve()),
        "archive_bytes": actual_bytes,
        "output": str(args.output.resolve()),
        "panoptic_val2017_pngs": png_count,
        "gt_seg_root_for_audits": str((args.output / "annotations").resolve()),
    }
    (args.output / "PANOPTIC_VAL2017_READY.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
