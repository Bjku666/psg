#!/usr/bin/env python3
"""Export CMUD utility labels from fit-only carrier predictions.

This runner intentionally accepts only a registered non-test partition.  It
does not alter the relation model and never reads official test rows for
training targets.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.legal_oracle import baseline_selection
from models.relation_decision_regret_v1.marginal_utility import marginal_slot_utilities, removal_utilities
from models.relation_decision_regret_v1.official_metric_adapter import mapped_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--partition", choices=("fit", "dev", "confirm"), default="fit")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--depth", default="all")
    args = parser.parse_args()
    if args.partition == "confirm":
        raise ValueError("confirm is reserved for one locked final evaluation; export fit/dev teacher only")
    annotation = json.loads(args.psg.read_text())
    split = json.loads(args.split_manifest.read_text())
    allowed = {str(value) for value in split[args.partition]}
    test_ids = {str(value) for value in annotation.get("test_image_ids", [])}
    # A bounded inference subset is materialized by ``make_annotation_subset``
    # with its selected fit/dev ids temporarily placed in ``test_image_ids``.
    # Those ids are local inference bookkeeping, not the official test split.
    # Accept that explicit contract while retaining the overlap guard for all
    # ordinary annotations (and never for confirm).
    local_subset = annotation.get("subset_contract", {}).get("partition") in {"fit", "dev"}
    if allowed & test_ids and not local_subset:
        raise ValueError("split manifest overlaps official test")
    if local_subset and args.partition == "confirm":
        raise ValueError("local fit/dev subset cannot be used for confirm")
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    by_id = {str(item["img_id"]): item for item in predictions}
    depth = "all" if str(args.depth) == "all" else int(args.depth)
    rows = []
    missing = []
    for entry in annotation["data"]:
        image_id = str(entry["image_id"])
        if image_id not in allowed or not entry.get("relations"):
            continue
        item = by_id.get(image_id)
        if item is None:
            missing.append(image_id)
            continue
        candidates, _ = mapped_candidates(entry, item, args.gt_seg_root, include_unmapped=True)
        for budget in args.budgets:
            baseline = baseline_selection(candidates, budget)
            insertion = marginal_slot_utilities(candidates, entry["relations"], baseline,
                                                budget, depth, len(annotation["predicate_classes"]))
            removal = removal_utilities(entry["relations"], baseline, len(annotation["predicate_classes"]))
            rows.append({"image_id": image_id, "file_name": entry["file_name"],
                         "bootstrap_group": entry["file_name"], "budget": int(budget),
                         "depth": depth, "insertion": insertion, "removal": removal})
    document = {"schema_version": 1, "contract": "fit/dev-only exact marginal utility teacher",
                "source_psg": str(args.psg), "partition": args.partition,
                "budgets": list(args.budgets), "depth": depth,
                "images_with_predictions": len(rows), "missing_predictions": missing, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n")
    print(json.dumps({key: document[key] for key in ("partition", "budgets", "images_with_predictions", "missing_predictions")}, indent=2))


if __name__ == "__main__":
    main()
