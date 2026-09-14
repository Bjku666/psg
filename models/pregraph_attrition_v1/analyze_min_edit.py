#!/usr/bin/env python3
"""Recompute and bootstrap a stored minimum-change oracle curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _balanced(points: list[dict]) -> float:
    if not points:
        return 0.0
    hit = np.sum([np.asarray(point["hit_per_predicate"], dtype=np.int64) for point in points], axis=0)
    gt = np.sum([np.asarray(point["gt_per_predicate"], dtype=np.int64) for point in points], axis=0)
    valid = gt > 0
    return float(np.mean(hit[valid] / gt[valid])) if valid.any() else 0.0


def analyze(rows: list[dict], seed: int = 0, replicates: int = 2000) -> dict:
    if not rows:
        raise ValueError("minimum-change input is empty")
    epsilons = [float(point["epsilon"]) for point in rows[0]["curve"]]
    if not epsilons or epsilons[0] != 0.0:
        raise ValueError("curve must start with the native epsilon=0 baseline")
    if any([float(point["epsilon"]) for point in row["curve"]] != epsilons for row in rows):
        raise ValueError("all rows must use the same epsilon curve")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["file_name"]), []).append(row)
    groups = [grouped[key] for key in sorted(grouped)]

    def points(sample: list[list[dict]], index: int) -> list[dict]:
        return [row["curve"][index] for group in sample for row in group]

    baseline = _balanced(points(groups, 0))
    rng = np.random.default_rng(seed)
    bootstrap_gains = [[] for _ in epsilons]
    for _ in range(int(replicates)):
        sample = [groups[int(index)] for index in rng.integers(0, len(groups), size=len(groups))]
        sampled_baseline = _balanced(points(sample, 0))
        for index in range(len(epsilons)):
            bootstrap_gains[index].append(_balanced(points(sample, index)) - sampled_baseline)

    curve = []
    for index, epsilon in enumerate(epsilons):
        current = points(groups, index)
        support = _balanced(current)
        changed = np.asarray([float(point["changed_fraction"]) for point in current], dtype=float)
        native = np.asarray([float(point["native_endpoint_support"]) for point in current], dtype=float)
        rescued = np.asarray([float(point["endpoint_support"]) for point in current], dtype=float) > native
        draws = np.asarray(bootstrap_gains[index], dtype=float)
        curve.append({
            "epsilon": epsilon,
            "predicate_balanced_endpoint_support": support,
            "gain_vs_native": support - baseline,
            "gain_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "changed_fraction": {
                "mean": float(changed.mean()),
                "p90": float(np.quantile(changed, 0.9)),
                "max": float(changed.max()),
            },
            "images_with_support_gain": int(rescued.sum()),
            "images_with_pixel_change": int(np.count_nonzero(changed)),
            "budget_violations": int(np.count_nonzero(changed > epsilon + 1e-12)),
        })
    return {
        "schema_version": 1,
        "contract": "paired physical-file bootstrap of predicate-balanced minimum-change gain",
        "images": len(rows),
        "bootstrap": {"unit": "physical_file_name", "groups": len(groups),
                      "replicates": int(replicates), "seed": int(seed)},
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.input.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    document = analyze(rows, args.seed, args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
