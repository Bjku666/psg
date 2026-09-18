#!/usr/bin/env python3
"""Create the registered physical-file grouped fit/dev/confirm manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p1_v1.split_train_dev import make_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    psg = json.loads(args.psg.read_text())
    split = make_split(psg, seed=args.seed)
    document = {"schema_version": 1, "source_psg": str(args.psg),
                "group_key": "physical_file_name", "fractions": {"fit": 0.8, "dev": 0.1, "confirm": 0.1},
                "seed": args.seed, **split}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: len(value) for key, value in split.items()}, indent=2))


if __name__ == "__main__":
    main()
