#!/usr/bin/env python3
"""Summarize relation-critical semantic queries that miss native admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(rows: list[dict]) -> dict:
    critical = [row for row in rows if row.get("relation_critical")]
    with_support = [row for row in critical if int(row.get("original_area", 0)) > 0]
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in with_support if row.get(key) is not None], dtype=float)
    deficits = values("deficit_pixels")
    costs = values("rescue_cost")
    ratios = values("area_ratio")
    return {
        "semantic_matched_queries": len(rows),
        "relation_critical_queries": len(critical),
        "relation_critical_with_mask_support": len(with_support),
        "competition_candidates": int(sum(bool(row.get("competition_candidate")) for row in critical)),
        "zero_support": int(sum(int(row.get("original_area", 0)) == 0 for row in critical)),
        "area_ratio": {
            "mean": float(ratios.mean()) if len(ratios) else None,
            "median": float(np.median(ratios)) if len(ratios) else None,
            "under_0.8": int(np.sum(ratios < 0.8)) if len(ratios) else 0,
        },
        "deficit_pixels": {
            "mean": float(deficits.mean()) if len(deficits) else None,
            "median": float(np.median(deficits)) if len(deficits) else None,
            "p90": float(np.quantile(deficits, 0.9)) if len(deficits) else None,
        },
        "rescue_cost": {
            "mean": float(costs.mean()) if len(costs) else None,
            "median": float(np.median(costs)) if len(costs) else None,
            "p90": float(np.quantile(costs, 0.9)) if len(costs) else None,
        },
        "near_miss_le_0.5pct_area": int(sum(
            float(row.get("deficit_pixels", 0)) / max(1, int(row.get("original_area", 0))) <= 0.005
            for row in with_support
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="stage decomposition per_image.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    with args.input.open() as stream:
        for line in stream:
            document = json.loads(line)
            rows.extend(document.get("near_miss", []))
    if not rows:
        raise ValueError("input contains no near-miss rows")
    document = {
        "schema_version": 1,
        "contract": "semantic-matched relation endpoint -> competition/admission near miss",
        "source": str(args.input.resolve()),
        "all": summarize(rows),
        "relation_critical": summarize([row for row in rows if row.get("relation_critical")]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
