"""Grouped fit/dev/confirm split with predicate coverage guarantees.

The historical splitter is intentionally left unchanged.  This module is the
v2 protocol for learner development: physical files remain atomic, while the
dev and confirm partitions are seeded with at least one physical group for
every predicate that occurs in the training population (when enough groups
exist).  Remaining groups are assigned by a deterministic hash order.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping


def make_stratified_split(
    psg: Mapping[str, Any], *, seed: int = 0,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[str]]:
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive and sum to one")
    test_ids = {str(x) for x in psg.get("test_image_ids", [])}
    declared = psg.get("train_image_ids")
    train_ids = ({str(x) for x in declared} if declared else {
        str(row.get("image_id")) for row in psg.get("data", [])
        if str(row.get("image_id")) not in test_ids
    })

    groups: dict[str, set[str]] = defaultdict(set)
    group_predicates: dict[str, set[int]] = defaultdict(set)
    for row in psg.get("data", []):
        image_id = str(row.get("image_id"))
        if image_id not in train_ids:
            continue
        filename = str(row.get("file_name", image_id))
        groups[filename].add(image_id)
        for relation in row.get("relations", []):
            if len(relation) >= 3:
                group_predicates[filename].add(int(relation[2]))
    names = sorted(groups, key=lambda name: hashlib.sha256(
        f"{seed}:{name}".encode()).hexdigest())
    n = len(names)
    targets = [round(n * fractions[0]), round(n * fractions[1]), 0]
    targets[2] = n - targets[0] - targets[1]
    buckets: list[list[str]] = [[], [], []]
    remaining = set(names)

    # Seed dev and confirm with rare predicates first.  A group is never split.
    predicate_groups: dict[int, list[str]] = defaultdict(list)
    for name in names:
        for predicate in group_predicates[name]:
            predicate_groups[predicate].append(name)
    for predicate in sorted(predicate_groups, key=lambda p: (len(predicate_groups[p]), p)):
        choices = [name for name in predicate_groups[predicate] if name in remaining]
        choices.sort(key=lambda name: hashlib.sha256(
            f"{seed}:seed:{predicate}:{name}".encode()).hexdigest())
        for bucket_index in (1, 2):
            if len(buckets[bucket_index]) >= targets[bucket_index]:
                continue
            candidate = next((name for name in choices if name in remaining), None)
            if candidate is None:
                break
            buckets[bucket_index].append(candidate)
            remaining.remove(candidate)
            choices.remove(candidate)

    # Fill exact group quotas in stable hash order.
    for name in names:
        if name not in remaining:
            continue
        eligible = [i for i in range(3) if len(buckets[i]) < targets[i]]
        if not eligible:
            eligible = [2]
        # Prefer the bucket furthest below its target; ties use fit/dev/confirm.
        index = max(eligible, key=lambda i: (targets[i] - len(buckets[i]), -i))
        buckets[index].append(name)

    result = {
        partition: [image_id for filename in bucket for image_id in sorted(groups[filename])]
        for partition, bucket in zip(("fit", "dev", "confirm"), buckets)
    }
    if set().union(*(set(values) for values in result.values())) & test_ids:
        raise AssertionError("stratified split overlaps official test")
    return result


def predicate_coverage(psg: Mapping[str, Any], image_ids: list[str]) -> list[int]:
    """Return sorted predicate ids represented by a partition."""
    allowed = {str(value) for value in image_ids}
    found = {int(relation[2]) for row in psg.get("data", [])
             if str(row.get("image_id")) in allowed
             for relation in row.get("relations", []) if len(relation) >= 3}
    return sorted(found)
