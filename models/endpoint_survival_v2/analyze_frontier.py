#!/usr/bin/env python3
"""Paired physical-file bootstrap and stage-rescue complementarity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def grouped_counts(path: Path, num_predicates: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["file_name"])
        gt = np.asarray(row["gt_per_predicate"], dtype=np.int64)
        hit = np.asarray(row["hit_per_predicate"], dtype=np.int64)
        if len(gt) != num_predicates or len(hit) != num_predicates:
            raise ValueError(f"predicate dimension mismatch in {path}")
        if key not in grouped:
            grouped[key] = (np.zeros(num_predicates, dtype=np.int64), np.zeros(num_predicates, dtype=np.int64))
        grouped[key][0][:] += gt
        grouped[key][1][:] += hit
    keys = sorted(grouped)
    return keys, np.stack([grouped[key][0] for key in keys]), np.stack([grouped[key][1] for key in keys])


def balanced_batch(gt: np.ndarray, hit: np.ndarray) -> np.ndarray:
    valid = gt > 0
    support = np.divide(hit, gt, out=np.zeros_like(hit, dtype=float), where=valid)
    return np.divide((support * valid).sum(axis=-1), valid.sum(axis=-1),
                     out=np.zeros(support.shape[:-1], dtype=float), where=valid.sum(axis=-1) > 0)


def analyze(baseline: Path, points: list[Path], semantic: Path, competition: Path,
            num_predicates: int, bootstraps: int, seed: int) -> dict:
    keys, base_gt, base_hit = grouped_counts(baseline, num_predicates)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=bootstraps)
    boot_gt = draws @ base_gt
    boot_base_hit = draws @ base_hit
    boot_base = balanced_batch(boot_gt, boot_base_hit)
    observed_base = float(balanced_batch(base_gt.sum(0)[None], base_hit.sum(0)[None])[0])
    point_results = []
    point_arrays: dict[Path, np.ndarray] = {}
    for point in points:
        point_keys, point_gt, point_hit = grouped_counts(point, num_predicates)
        if point_keys != keys or not np.array_equal(point_gt, base_gt):
            raise ValueError(f"paired population mismatch for {point}")
        point_arrays[point] = point_hit
        observed = float(balanced_batch(point_gt.sum(0)[None], point_hit.sum(0)[None])[0])
        boot_point = balanced_batch(boot_gt, draws @ point_hit)
        deltas = boot_point - boot_base
        point_results.append({
            "source": str(point), "observed_support": observed,
            "observed_delta": observed - observed_base,
            "paired_bootstrap_delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "probability_delta_at_least_0p03": float(np.mean(deltas >= 0.03)),
        })

    semantic_hit = point_arrays[semantic]
    competition_hit = point_arrays[competition]
    global_gt = base_gt.sum(0)
    weights = np.divide(1.0, global_gt, out=np.zeros(num_predicates, dtype=float), where=global_gt > 0)
    semantic_gain = ((semantic_hit - base_hit) * weights).sum(axis=1)
    competition_gain = ((competition_hit - base_hit) * weights).sum(axis=1)
    semantic_positive, competition_positive = semantic_gain > 0, competition_gain > 0
    both = int(np.count_nonzero(semantic_positive & competition_positive))
    union = int(np.count_nonzero(semantic_positive | competition_positive))
    return {
        "schema_version": 1,
        "contract": "paired bootstrap unit is physical file_name; duplicate annotation rows stay within group",
        "physical_file_groups": len(keys), "bootstrap_replicates": bootstraps, "seed": seed,
        "baseline_support": observed_base, "points": point_results,
        "frozen_stage_points": {"semantic": str(semantic), "competition": str(competition)},
        "rescue_complementarity": {
            "definition": "positive predicate-balanced hit-mass change per physical-file group",
            "semantic_positive_groups": int(semantic_positive.sum()),
            "competition_positive_groups": int(competition_positive.sum()),
            "both_positive_groups": both,
            "semantic_only_groups": int(np.count_nonzero(semantic_positive & ~competition_positive)),
            "competition_only_groups": int(np.count_nonzero(~semantic_positive & competition_positive)),
            "union_positive_groups": union,
            "positive_group_jaccard": both / union if union else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--points", nargs="+", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--competition", type=Path, required=True)
    parser.add_argument("--num-predicates", type=int, default=56)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.baseline, args.points, args.semantic, args.competition,
                     args.num_predicates, args.bootstraps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
