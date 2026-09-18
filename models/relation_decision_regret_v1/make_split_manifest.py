#!/usr/bin/env python3
"""Create the registered physical-file grouped fit/dev/confirm manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.stratified_split import make_stratified_split, predicate_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                        metavar=("FIT", "DEV", "CONFIRM"))
    args = parser.parse_args()
    psg = json.loads(args.psg.read_text())
    fractions = tuple(args.fractions)
    split = make_stratified_split(psg, seed=args.seed, fractions=fractions)
    document = {"schema_version": 1, "source_psg": str(args.psg),
                "protocol": "v2_grouped_predicate_coverage",
                "group_key": "physical_file_name",
                "fractions": {"fit": fractions[0], "dev": fractions[1], "confirm": fractions[2]},
                "seed": args.seed,
                "predicate_coverage": {name: predicate_coverage(psg, values)
                                        for name, values in split.items()},
                **split}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: len(value) for key, value in split.items()}, indent=2))


if __name__ == "__main__":
    main()
