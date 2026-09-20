#!/usr/bin/env python3
"""Merge compact P1.4 shards with duplicate/missing-id checks."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def merge(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("no compact shards supplied")
    documents = []
    seen: set[str] = set()
    predicate_classes = None
    for path in sorted(paths):
        with path.open("rb") as stream:
            document = pickle.load(stream)
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError(f"unsupported compact carrier: {path}")
        if predicate_classes is None:
            predicate_classes = document.get("predicate_classes")
        elif document.get("predicate_classes") != predicate_classes:
            raise ValueError(f"predicate class mismatch: {path}")
        for row in document.get("images", []):
            image_id = str(row["image_id"])
            if image_id in seen:
                raise ValueError(f"duplicate image id across shards: {image_id}")
            seen.add(image_id)
            documents.append(row)
    documents.sort(key=lambda row: str(row["image_id"]))
    return {
        "schema_version": 1,
        "contract": {
            "support": "frozen carrier compact shards",
            "confirm_locked": True,
            "shards": len(paths),
        },
        "predicate_classes": predicate_classes,
        "images": documents,
        "missing_predictions": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", choices=("fit", "dev"), required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob(f"{args.partition}_*.pkl"))
    result = merge(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print({"partition": args.partition, "shards": len(paths),
           "images": len(result["images"]), "bytes": args.output.stat().st_size})


if __name__ == "__main__":
    main()
