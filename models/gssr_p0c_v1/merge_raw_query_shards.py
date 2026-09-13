#!/usr/bin/env python3
"""Validate and merge sharded raw-query manifests without copying artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0c_v1.raw_query_schema import validate_image_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--shards", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    with args.psg.open() as stream:
        psg = json.load(stream)
    test_ids = {str(value) for value in psg["test_image_ids"]}
    expected_files = set()
    for row in psg["data"]:
        is_test = str(row["image_id"]) in test_ids
        if args.split == "all" or (args.split == "test" and is_test) or (args.split == "train" and not is_test):
            expected_files.add(str(row["file_name"]))
    records: dict[str, dict] = {}
    contracts = []
    for shard in args.shards:
        with (shard / "contract.json").open() as stream:
            contract = json.load(stream)
        contracts.append(contract)
        with (shard / "manifest.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                validate_image_record(record)
                filename = str(record["file_name"])
                if filename in records:
                    raise ValueError(f"duplicate filename across shards: {filename}")
                artifact = Path(record["artifact_file"])
                if not artifact.is_absolute():
                    artifact = (shard / artifact).resolve()
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
                record["artifact_file"] = str(artifact)
                records[filename] = record
    if set(records) != expected_files:
        missing = sorted(expected_files.difference(records))
        extra = sorted(set(records).difference(expected_files))
        raise ValueError(f"shard coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")
    shard_indices = sorted(int(contract["shard_index"]) for contract in contracts)
    declared_shards = {int(contract["num_shards"]) for contract in contracts}
    if declared_shards != {len(args.shards)} or shard_indices != list(range(len(args.shards))):
        raise ValueError("contracts do not form one complete shard set")
    invariant_keys = ("model", "model_config_sha256", "score_definitions", "native_admission",
                      "split", "artifact_profile")
    for key in invariant_keys:
        values = {json.dumps(contract.get(key, "p0c_full" if key == "artifact_profile" else None),
                             sort_keys=True) for contract in contracts}
        if len(values) != 1:
            raise ValueError(f"shard contract mismatch for {key}")
    args.output.mkdir(parents=True)
    with (args.output / "manifest.jsonl").open("w") as stream:
        for filename in sorted(records):
            stream.write(json.dumps(records[filename], separators=(",", ":")) + "\n")
    merged_contract = {
        "schema_version": contracts[0]["schema_version"],
        "kind": "merged_raw_query_manifest",
        "images": len(records),
        "model": contracts[0]["model"],
        "model_config_sha256": contracts[0]["model_config_sha256"],
        "score_definitions": contracts[0]["score_definitions"],
        "native_admission": contracts[0]["native_admission"],
        "split": args.split,
        "artifact_profile": contracts[0].get("artifact_profile", "p0c_full"),
        "shards": [str(path.resolve()) for path in args.shards],
    }
    (args.output / "contract.json").write_text(json.dumps(merged_contract, indent=2) + "\n")
    print(json.dumps(merged_contract, indent=2))


if __name__ == "__main__":
    main()
