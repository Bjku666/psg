#!/usr/bin/env python3
"""Materialize a small registered split subset for resource-bounded smoke runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--partition", choices=("fit", "dev", "confirm"), required=True)
    parser.add_argument("--max-images", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    psg = json.loads(args.psg.read_text())
    manifest = json.loads(args.split_manifest.read_text())
    allowed = {str(value) for value in manifest[args.partition]}
    test_ids = {str(value) for value in psg.get("test_image_ids", [])}
    if allowed & test_ids:
        raise ValueError("partition overlaps official test")
    data = [item for item in psg["data"] if str(item["image_id"]) in allowed and item.get("relations")]
    data.sort(key=lambda item: str(item["image_id"]))
    data = data[: int(args.max_images)]
    subset = dict(psg)
    subset["data"] = data
    # Fair PSG's loader treats test_image_ids as its val/test population. By
    # making exactly this registered subset test, inference is bounded without
    # changing any image annotations or class definitions.
    subset["test_image_ids"] = [item["image_id"] for item in data]
    subset["subset_contract"] = {"source": str(args.psg), "partition": args.partition,
                                  "max_images": int(args.max_images), "group_key": "physical_file_name"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset) + "\n")
    print(json.dumps({"partition": args.partition, "images": len(data), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
