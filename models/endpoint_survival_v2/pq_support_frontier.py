#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_frontier(baseline_file: Path, input_files: list[Path]) -> dict:
    baseline_document = json.loads(baseline_file.read_text())
    if baseline_document.get("mode") != "native" or float(baseline_document.get("margin", -1)) != 0:
        raise ValueError("--baseline must be an explicit native, margin-zero run")
    baseline = baseline_document["metrics"]
    points = []
    for path in input_files:
        document = json.loads(path.read_text())
        metrics = document["metrics"]
        point = {
            "mode": document["mode"], "margin": document["margin"], "source": str(path),
            **{key: metrics[key] for key in ("endpoint_support", "predicate_balanced_endpoint_support", "pq", "pq_th", "pq_st", "changed_pixel_fraction", "changed_queries", "semantic_eligibility_flips", "competition_winner_flips", "native_segment_count_change")},
        }
        point["delta_endpoint_support"] = point["endpoint_support"] - baseline["endpoint_support"]
        point["delta_predicate_balanced_support"] = point["predicate_balanced_endpoint_support"] - baseline["predicate_balanced_endpoint_support"]
        point["delta_pq"] = point["pq"] - baseline["pq"]
        epsilon = 1e-12
        point["support_gate_passed"] = point["delta_predicate_balanced_support"] >= 0.03 - epsilon
        point["pq_gate_passed"] = point["delta_pq"] >= -0.01 - epsilon
        points.append(point)
    eligible = [point for point in points if point["support_gate_passed"] and point["pq_gate_passed"]]
    best = max(eligible, key=lambda point: (point["delta_predicate_balanced_support"], -point["changed_pixel_fraction"]), default=None)
    return {
        "schema_version": 2,
        "contract": "native-relative official-PQ versus full-population balanced endpoint-support frontier",
        "baseline_source": str(baseline_file),
        "baseline": {key: baseline[key] for key in ("endpoint_support", "predicate_balanced_endpoint_support", "pq", "pq_th", "pq_st")},
        "points": points,
        "best_qualified_point": best,
        "gates": {"endpoint_support_delta_min": 0.03, "pq_delta_min": -0.01, "structured_oracle_passed": bool(eligible)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_frontier(args.baseline, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
