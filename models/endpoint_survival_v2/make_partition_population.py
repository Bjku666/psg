#!/usr/bin/env python3
"""Materialize a relation-bearing population from a frozen grouped split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--partition", choices=("fit", "dev", "confirm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.file_list.exists():
        raise FileExistsError("refusing to overwrite population outputs")
    psg = json.loads(args.psg.read_text())
    split = json.loads(args.split_manifest.read_text())
    selected_ids = {str(value) for value in split[args.partition]}
    rows = [row for row in psg["data"] if str(row["image_id"]) in selected_ids and row.get("relations")]
    rows.sort(key=lambda row: (str(row["file_name"]), str(row["image_id"])))
    file_names = sorted({str(row["file_name"]) for row in rows})
    document = {
        "schema_version": 1,
        "contract": "relation-bearing rows from frozen physical-file grouped split",
        "source_split_manifest": str(args.split_manifest.resolve()),
        "source_split_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "partition": args.partition,
        "group_key": "physical_file_name",
        "rows": len(rows), "groups": len(file_names),
        "image_ids": [str(row["image_id"]) for row in rows],
        "file_names": file_names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    args.file_list.write_text("\n".join(file_names) + "\n")
    print(json.dumps({key: value for key, value in document.items() if key not in ("image_ids", "file_names")}, indent=2))


if __name__ == "__main__":
    main()
