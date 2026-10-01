"""Audit physical-file identity collisions in the immutable OpenPSG source."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--psg", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.psg.read_text())
    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for row in source["data"]:
        by_file[row["file_name"]].append(row)
    collisions = []
    for file_name, rows in sorted(by_file.items()):
        if len(rows) <= 1:
            continue
        collisions.append({
            "file_name": file_name,
            "rows": len(rows),
            "image_ids": [str(row["image_id"]) for row in rows],
            "coco_image_ids": [str(row.get("coco_image_id")) for row in rows],
            "relation_counts": [len(row.get("relations", [])) for row in rows],
        })
    report = {
        "schema_version": 1,
        "contract": "physical file identity audit",
        "dataset_sha": digest(args.psg),
        "rows": len(source["data"]),
        "unique_file_names": len(by_file),
        "collision_groups": len(collisions),
        "collision_rows": sum(item["rows"] for item in collisions),
        "dedup_policy": "fail",
        "pass": not collisions,
        "collisions": collisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "unique_file_names", "collision_groups", "pass")}, indent=2))
    if collisions:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

