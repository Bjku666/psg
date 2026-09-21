"""Applied intervention and outcome accounting for predicate surgery."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence


def _truth(relations: Sequence[Sequence[int]]) -> dict[tuple[int, int], set[int]]:
    truth: dict[tuple[int, int], set[int]] = defaultdict(set)
    for subject, obj, predicate in relations:
        truth[(int(subject), int(obj))].add(int(predicate))
    return truth


def account_repairs(
    relations: Sequence[Sequence[int]],
    native: Sequence[Mapping],
    repaired: Sequence[Mapping],
) -> dict:
    """Count edits and correctness transitions by comparing emitted labels."""
    by_row = {int(row["row"]): row for row in repaired}
    truth = _truth(relations)
    transitions = Counter()
    changed = 0
    for old in native:
        new = by_row.get(int(old["row"]), old)
        old_predicate, new_predicate = int(old["pred"]), int(new["pred"])
        changed += int(old_predicate != new_predicate)
        mapped = old.get("gt_pair")
        valid = set() if mapped is None else truth.get(tuple(map(int, mapped)), set())
        old_ok = old_predicate in valid
        new_ok = new_predicate in valid
        transitions[
            ("correct" if old_ok else "wrong")
            + "_to_"
            + ("correct" if new_ok else "wrong")
        ] += 1
    total = len(native)
    return {
        "rows": int(total),
        "applied_swaps": int(changed),
        "intervention_rate": float(changed / total) if total else 0.0,
        "transitions": dict(transitions),
    }


def merge_repair_accounts(accounts: Sequence[Mapping]) -> dict:
    rows = sum(int(item.get("rows", 0)) for item in accounts)
    swaps = sum(int(item.get("applied_swaps", 0)) for item in accounts)
    transitions = Counter()
    for item in accounts:
        transitions.update(item.get("transitions", {}))
    return {
        "rows": int(rows),
        "applied_swaps": int(swaps),
        "intervention_rate": float(swaps / rows) if rows else 0.0,
        "transitions": dict(transitions),
    }

