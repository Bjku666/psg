"""Correctness transition accounting for native-preserving repairs."""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Mapping, Sequence


def _truth(relations: Sequence[Sequence[int]]) -> dict[tuple[int, int], set[int]]:
    out = defaultdict(set)
    for s, o, r in relations:
        out[(int(s), int(o))].add(int(r))
    return out


def account_flips(relations: Sequence[Sequence[int]], native: Sequence[Mapping],
                  repaired: Sequence[Mapping]) -> dict:
    """Count correct→wrong, wrong→correct and preserve transitions by pair."""
    truth = _truth(relations)
    by_row = {int(row["row"]): row for row in repaired}
    counts = Counter()
    strata = defaultdict(Counter)
    for old in native:
        new = by_row.get(int(old["row"]), old)
        pair = tuple(map(int, old.get("gt_pair", old["pair"])))
        valid = truth.get(pair, set())
        old_ok = int(old["pred"]) in valid
        new_ok = int(new["pred"]) in valid
        state = ("correct" if old_ok else "wrong") + "_to_" + ("correct" if new_ok else "wrong")
        counts[state] += 1
        rank = int(old.get("predicate_rank", 0))
        strata[f"native_rank_{rank}"][state] += 1
    return {"counts": dict(counts), "total": int(sum(counts.values())),
            "strata": {key: dict(value) for key, value in strata.items()}}
