#!/usr/bin/env python3
"""Summarize relation-critical semantic queries that miss native admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(rows: list[dict], relation_only: bool = False) -> dict:
    """Summarize one explicitly defined query population.

    ``rows`` is already the population to summarize.  The optional filter is
    retained for the relation-critical report; the all-query report must not
    silently discard non-relation controls.
    """
    population = [row for row in rows if row.get("relation_critical")] if relation_only else list(rows)
    with_support = [row for row in population if int(row.get("original_area", 0)) > 0]
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in with_support if row.get(key) is not None], dtype=float)
    deficits = values("deficit_pixels")
    costs = values("rescue_cost")
    ratios = values("area_ratio")
    deficit_fractions = np.asarray([
        float(row.get("deficit_pixels", 0)) / max(1, int(row.get("original_area", 0)))
        for row in with_support
    ], dtype=float)
    return {
        "semantic_matched_queries": len(population),
        "relation_critical_queries": int(sum(bool(row.get("relation_critical")) for row in population)),
        "queries_with_mask_support": len(with_support),
        "relation_critical_with_mask_support": int(sum(
            bool(row.get("relation_critical")) for row in with_support
        )),
        "competition_candidates": int(sum(bool(row.get("competition_candidate")) for row in population)),
        "zero_support": int(sum(int(row.get("original_area", 0)) == 0 for row in population)),
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
        "deficit_fraction_of_original_area": {
            "mean": float(deficit_fractions.mean()) if len(deficit_fractions) else None,
            "median": float(np.median(deficit_fractions)) if len(deficit_fractions) else None,
            "p90": float(np.quantile(deficit_fractions, 0.9)) if len(deficit_fractions) else None,
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


def compare_cohorts(rows: list[dict], seed: int = 0, replicates: int = 2000) -> dict:
    """Cluster-bootstrap relation minus non-relation median differences."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["bootstrap_group"]), []).append(row)
    groups = [grouped[key] for key in sorted(grouped)]

    def metrics(sample: list[list[dict]]) -> dict[str, float] | None:
        flat = [row for group in sample for row in group if int(row.get("original_area", 0)) > 0]
        relation = [row for row in flat if row.get("relation_critical")]
        control = [row for row in flat if not row.get("relation_critical")]
        if not relation or not control:
            return None
        def median(population: list[dict], key: str) -> float:
            return float(np.median([float(row[key]) for row in population]))
        return {
            "area_ratio": median(relation, "area_ratio") - median(control, "area_ratio"),
            "deficit_fraction_of_original_area": float(np.median([
                float(row["deficit_pixels"]) / max(1, int(row["original_area"])) for row in relation
            ])) - float(np.median([
                float(row["deficit_pixels"]) / max(1, int(row["original_area"])) for row in control
            ])),
            "rescue_cost": median(relation, "rescue_cost") - median(control, "rescue_cost"),
        }

    point = metrics(groups)
    if point is None:
        return {"unit": "physical_file_name", "groups": len(groups), "replicates": 0,
                "relation_minus_non_relation_median": {}}
    draws = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(int(replicates)):
        sample = [groups[int(index)] for index in rng.integers(0, len(groups), size=len(groups))]
        values = metrics(sample)
        if values is not None:
            for key, value in values.items():
                draws[key].append(value)
    return {
        "unit": "physical_file_name", "groups": len(groups), "replicates": int(replicates),
        "seed": int(seed),
        "relation_minus_non_relation_median": {
            key: {"estimate": value,
                  "ci95": [float(np.quantile(draws[key], 0.025)),
                           float(np.quantile(draws[key], 0.975))]}
            for key, value in point.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="stage decomposition per_image.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    with args.input.open() as stream:
        for line in stream:
            document = json.loads(line)
            for row in document.get("near_miss", []):
                row["bootstrap_group"] = str(document["file_name"])
                rows.append(row)
    if not rows:
        raise ValueError("input contains no near-miss rows")
    document = {
        "schema_version": 1,
        "contract": "semantic-matched relation endpoint -> competition/admission near miss",
        "source": str(args.input.resolve()),
        "all": summarize(rows, relation_only=False),
        "relation_critical": summarize(rows, relation_only=True),
        "non_relation_control": summarize([
            row for row in rows if not row.get("relation_critical")
        ]),
        "paired_control_comparison": compare_cohorts(rows, args.seed, args.replicates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
