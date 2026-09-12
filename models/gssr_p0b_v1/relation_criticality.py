#!/usr/bin/env python3
"""Summarize and plot relation-criticality records from a P0B audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def degree_bin(degree: int) -> str:
    return str(degree) if degree < 4 else "4+"


def selection_category(row: dict) -> str:
    score = bool(row["score_selected"])
    oracle = bool(row["balanced_oracle_selected"])
    if score and oracle:
        return "both"
    if score:
        return "score_topk_only"
    if oracle:
        return "balanced_oracle_only"
    return "neither"


def grouped_rescue(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    output = {}
    for name, group in sorted(groups.items()):
        dropped = [row for row in group if not row["score_selected"]]
        rescued = [row for row in dropped if row["balanced_oracle_selected"]]
        output[name] = {
            "entities": len(group),
            "score_dropped": len(dropped),
            "balanced_oracle_rescued": len(rescued),
            "rescue_rate_among_score_dropped": len(rescued) / len(dropped) if dropped else 0.0,
        }
    return output


def summarize(rows: list[dict]) -> dict:
    available = [row for row in rows if row["best_candidate_score"] is not None]
    scores = np.asarray([row["best_candidate_score"] for row in available], dtype=float)
    degrees = np.asarray([row["relation_degree"] for row in available], dtype=float)
    correlation = spearmanr(scores, degrees) if len(scores) > 1 else None
    categories = defaultdict(int)
    for row in rows:
        categories[selection_category(row)] += 1
    return {
        "entities": len(rows),
        "entities_with_supplied_candidate": len(available),
        "selection_categories": dict(sorted(categories.items())),
        "score_degree_spearman": {
            "rho": float(correlation.statistic) if correlation is not None else None,
            "pvalue": float(correlation.pvalue) if correlation is not None else None,
        },
        "rescue_by_relation_degree": grouped_rescue(
            rows, lambda row: degree_bin(int(row["relation_degree"]))
        ),
        "rescue_by_thing_or_stuff": grouped_rescue(rows, lambda row: row["thing_or_stuff"]),
        "rescue_by_rare_predicate": grouped_rescue(
            rows, lambda row: "rare" if row["has_rare_predicate"] else "not_rare"
        ),
    }


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "score_topk_only": "#E69F00",
        "balanced_oracle_only": "#0072B2",
        "both": "#009E73",
        "neither": "#999999",
    }
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for category in colors:
        selected = [
            row for row in rows
            if row["best_candidate_score"] is not None and selection_category(row) == category
        ]
        if not selected:
            continue
        axis.scatter(
            [row["best_candidate_score"] for row in selected],
            [row["relation_degree"] for row in selected],
            s=12,
            alpha=0.35,
            linewidths=0,
            color=colors[category],
            label=category.replace("_", " "),
        )
    axis.set_xlabel("Best matched candidate score")
    axis.set_ylabel("GT entity relation degree")
    axis.legend(frameon=False, markerscale=1.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.plot and args.plot.exists()):
        raise FileExistsError("refusing to overwrite an existing criticality output")
    with args.input.open() as stream:
        rows = [json.loads(line) for line in stream]
    document = summarize(rows)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    if args.plot:
        plot(rows, args.plot)
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()

