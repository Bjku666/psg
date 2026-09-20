#!/usr/bin/env python3
"""Materialize deterministic fit/dev inference shards for the P1.4 carrier.

Each shard is a valid Fair-PSG validation annotation: only registered image
ids are present and ``test_image_ids`` names exactly those rows.  Confirm is
intentionally forbidden here so it cannot be consumed during development.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_shards(psg: dict, manifest: dict, partition: str, shard_size: int) -> list[dict]:
    if partition not in {"fit", "dev"}:
        raise ValueError("P1.4 carrier export is restricted to fit/dev")
    official_test = {str(value) for value in psg.get("test_image_ids", [])}
    selected = {str(value) for value in manifest[partition]}
    overlap = selected & official_test
    if overlap:
        raise ValueError(f"{partition} overlaps official test ({len(overlap)} ids)")
    # Keep relation-free images too.  They are part of the frozen physical-file
    # population and are legal in a Fair-PSG ``test`` subset; downstream mR
    # aggregation simply has no predicate denominator for those rows.
    rows = [item for item in psg["data"] if str(item["image_id"]) in selected]
    rows.sort(key=lambda item: str(item["image_id"]))
    if len(rows) != len(selected):
        missing = selected - {str(item["image_id"]) for item in rows}
        raise ValueError(f"manifest rows missing relation annotations: {len(missing)}")
    shards = []
    for start in range(0, len(rows), int(shard_size)):
        values = rows[start:start + int(shard_size)]
        document = dict(psg)
        document["data"] = values
        document["test_image_ids"] = [item["image_id"] for item in values]
        document["subset_contract"] = {
            "source": manifest.get("source_psg"),
            "partition": partition,
            "group_key": manifest.get("group_key", "physical_file_name"),
            "shard_start": start,
            "shard_size": len(values),
        }
        shards.append(document)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--partition", required=True, choices=("fit", "dev"))
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    psg = json.loads(args.psg.read_text())
    manifest = json.loads(args.split_manifest.read_text())
    shards = make_shards(psg, manifest, args.partition, args.shard_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for shard_id, document in enumerate(shards):
        name = f"{args.partition}_{shard_id:03d}.json"
        path = args.output_dir / name
        path.write_text(json.dumps(document, separators=(",", ":")) + "\n")
        index.append({
            "shard": shard_id,
            "annotation": str(path),
            "images": len(document["data"]),
            "first_image_id": str(document["data"][0]["image_id"]),
            "last_image_id": str(document["data"][-1]["image_id"]),
        })
    index_path = args.output_dir / f"{args.partition}_index.json"
    index_path.write_text(json.dumps({
        "schema_version": 1,
        "partition": args.partition,
        "shard_size": args.shard_size,
        "images": sum(item["images"] for item in index),
        "shards": index,
    }, indent=2) + "\n")
    print(json.dumps({"partition": args.partition, "images": sum(x["images"] for x in index),
                      "shards": len(index), "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()
