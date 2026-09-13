"""Deterministic physical-image grouped fit/dev/confirm split for P1.

The splitter operates only on ``train_image_ids`` and never reads or emits test
rows.  Grouping by ``file_name`` prevents duplicate relation rows for one
physical image from crossing a split boundary.
"""
from __future__ import annotations
import hashlib
from typing import Any, Mapping

def make_split(psg: Mapping[str, Any], *, seed: int = 0,
               fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[str]]:
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive and sum to one")
    declared = psg.get("train_image_ids")
    if declared:
        train_ids = {str(x) for x in declared}
    else:
        # OpenPSG's canonical psg.json declares test IDs only; all remaining
        # rows are the train universe for this split utility.
        test_ids = {str(x) for x in psg.get("test_image_ids", [])}
        train_ids = {str(row.get("image_id")) for row in psg.get("data", [])
                     if str(row.get("image_id")) not in test_ids}
    groups: dict[str, set[str]] = {}
    for row in psg.get("data", []):
        image_id = str(row.get("image_id"))
        if image_id not in train_ids:
            continue
        filename = str(row.get("file_name", image_id))
        groups.setdefault(filename, set()).add(image_id)
    ordered = sorted(groups, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    n = len(ordered)
    cut1, cut2 = round(n * fractions[0]), round(n * (fractions[0] + fractions[1]))
    buckets = [ordered[:cut1], ordered[cut1:cut2], ordered[cut2:]]
    names = ("fit", "dev", "confirm")
    return {name: [image_id for filename in bucket for image_id in sorted(groups[filename])]
            for name, bucket in zip(names, buckets)}
