"""Create a deterministic image manifest from the registered OpenPSG source."""

from __future__ import annotations

import argparse
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
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()
    source = json.loads(args.psg.read_text())
    rows = [row for row in source["data"] if row["file_name"].startswith("val2017/")]
    rows = sorted(rows, key=lambda row: (row["file_name"], str(row["image_id"])))[: args.limit]
    names = [row["file_name"] for row in rows]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate file_name in source annotation")
    manifest = {
        "schema_version": 1,
        "contract": "single-source carrier/action manifest",
        "dataset_sha": digest(args.psg),
        "source": str(args.psg),
        "count": len(rows),
        "rows": [
            {
                "file_name": row["file_name"],
                "image_id": str(row["image_id"]),
                "pan_seg_file_name": row["pan_seg_file_name"],
                "bootstrap_group": row["file_name"],
            }
            for row in rows
        ],
        "pass": len(rows) == args.limit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("dataset_sha", "count", "pass")}, indent=2))


if __name__ == "__main__":
    main()

