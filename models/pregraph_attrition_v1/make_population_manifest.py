#!/usr/bin/env python3
"""Freeze a grouped, hash-sampled population for pre-graph audits.

The manifest is deliberately independent of image-id ordering.  Sampling is
performed over ``physical_file_name`` groups using SHA-256, then all relation-
bearing PSG rows in the selected groups are included.  Downstream audits can
consume this one file verbatim via ``--population-manifest``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--partition", choices=("fit", "dev", "confirm"), default="fit")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_images <= 0:
        raise ValueError("--max-images must be positive")

    psg = json.loads(args.psg.read_text())
    split = json.loads(args.split_manifest.read_text())
    allowed = {str(value) for value in split[args.partition]}
    groups: dict[str, list[dict]] = {}
    for item in psg.get("data", []):
        if str(item["image_id"]) not in allowed or not item.get("relations"):
            continue
        # The dataset's physical filename is the unit used by bootstrap and
        # grouped train/dev/confirm splitting.  Fall back only for old PSG
        # records that do not expose it explicitly.
        key = str(item.get("physical_file_name", item["file_name"]))
        groups.setdefault(key, []).append(item)

    def digest(key: str) -> str:
        return hashlib.sha256(f"{args.seed}:{key}".encode("utf-8")).hexdigest()

    selected_groups: list[str] = []
    selected_count = 0
    for key in sorted(groups, key=lambda value: (digest(value), value)):
        if selected_count >= args.max_images:
            break
        selected_groups.append(key)
        selected_count += len(groups[key])

    selected_items = [item for key in selected_groups for item in groups[key]]
    selected_items.sort(key=lambda item: (str(item.get("physical_file_name", item["file_name"])), str(item["image_id"])))
    document = {
        "schema_version": 1,
        "contract": "grouped physical-file hash-sampled population",
        "source_psg": str(args.psg.resolve()),
        "source_split_manifest": str(args.split_manifest.resolve()),
        "partition": args.partition,
        "group_key": "physical_file_name",
        "seed": args.seed,
        "requested_images": args.max_images,
        "images": len(selected_items),
        "groups": len(selected_groups),
        "image_ids": [str(item["image_id"]) for item in selected_items],
        "file_names": [str(item["file_name"]) for item in selected_items],
        "physical_file_names": selected_groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    file_list = args.output.with_suffix(".files.txt")
    file_list.write_text("\n".join(document["file_names"]) + "\n")
    print(json.dumps({k: document[k] for k in ("partition", "seed", "requested_images", "images", "groups")}, indent=2))


if __name__ == "__main__":
    main()
