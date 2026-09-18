"""Legal fixed-budget candidate oracle and baseline selection helpers."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


def _predicate_order(row: Mapping) -> np.ndarray:
    scores = np.asarray(row["pred_scores"], dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("candidate pred_scores must be a non-empty vector")
    # Predicate IDs are zero based and column zero (NONE) is not a relation.
    return np.argsort(-scores, kind="stable")


def evidence_candidates(rows: Sequence[Mapping], depth: int | str) -> list[dict]:
    """Attach the best legal predicate under an evidence-depth budget.

    ``depth`` limits which predicate columns can be selected, while retaining
    exactly one candidate per directed pair.  The carrier pair support is
    never expanded.
    """
    out = []
    for row in rows:
        order = _predicate_order(row)
        relation_order = [int(index) for index in order if int(index) > 0]
        if depth != "all":
            relation_order = relation_order[: int(depth)]
        if not relation_order:
            continue
        predicate = relation_order[0] - 1
        scores = np.asarray(row["pred_scores"], dtype=float)
        current = dict(row)
        current.update({
            "pred": int(predicate),
            "pred_score": float(scores[predicate + 1]),
            "predicate_rank": int(next(i for i, value in enumerate(order) if int(value) == predicate + 1)),
            "evidence_depth": depth,
        })
        out.append(current)
    return out


def _predicate_options(row: Mapping, depth: int | str) -> list[int]:
    order = [int(index) for index in _predicate_order(row) if int(index) > 0]
    if depth != "all":
        order = order[: int(depth)]
    return [index - 1 for index in order]


def baseline_selection(rows: Sequence[Mapping], budget: int) -> list[dict]:
    """D0: carrier ranking followed by predicate argmax."""
    candidates = evidence_candidates(rows, "all")
    # Fair PSG selects ``rel_rank.argsort()[-k:]``.  Preserve that exact
    # index-set contract (including its handling of tied scores).
    order = np.argsort(np.asarray([float(row["score"]) for row in candidates]))[-int(budget):]
    return [dict(candidates[int(index)]) for index in order]


def legal_oracle_selection(rows: Sequence[Mapping], relations: Sequence[Sequence[int]],
                           budget: int, depth: int | str) -> list[dict]:
    """Select a legal <=K set maximizing GT-aware image recall.

    The oracle sees GT only to label the already-supported candidate rows.  It
    cannot add a pair, alter a mask match, or choose a predicate outside the
    top-``depth`` evidence columns.  Weighted hit ordering is equivalent to
    maximizing image-wise mean recall when every pair has at most one GT
    predicate; it remains deterministic for ties and is conservative for
    malformed multi-label inputs.
    """
    gt = [(int(s), int(o), int(r)) for s, o, r in relations if int(s) != int(o)]
    counts = Counter(int(r) for _, _, r in gt)
    positives, negatives = [], []
    for source_row in rows:
        options = _predicate_options(source_row, depth)
        if not options:
            continue
        mapped_pair = source_row.get("gt_pair")
        pair = tuple(map(int, mapped_pair)) if mapped_pair is not None else (-1, -1)
        # Oracle may choose any predicate exposed by the evidence depth, but
        # still emits exactly one predicate for this physical pair.
        matching = [predicate for predicate in options
                    if any((pair[0], pair[1], predicate) == value for value in gt)]
        predicate = matching[0] if matching else options[0]
        row = dict(source_row)
        scores = np.asarray(source_row["pred_scores"], dtype=float)
        order = _predicate_order(source_row)
        row.update({
            "pred": int(predicate),
            "pred_score": float(scores[predicate + 1]),
            "predicate_rank": int(next(i for i, value in enumerate(order) if int(value) == predicate + 1)),
            "evidence_depth": depth,
        })
        predicate = int(row["pred"])
        is_positive = any((pair[0], pair[1], predicate) == value for value in gt)
        # Rare predicates have larger mR marginal value.  Score is the first
        # tie-break, followed by the original carrier row index.
        weight = 1.0 / max(1, counts[predicate]) if is_positive else 0.0
        key = (-weight, -float(row["pred_score"]), -float(row["score"]), int(row["row"]))
        (positives if is_positive else negatives).append((key, row))
    positives.sort(key=lambda x: x[0])
    negatives.sort(key=lambda x: x[0])
    chosen = [dict(row) for _, row in positives[: int(budget)]]
    if len(chosen) < int(budget):
        chosen.extend(dict(row) for _, row in negatives[: int(budget) - len(chosen)])
    return chosen


def decision_regret(baseline: Mapping, oracle: Mapping, metric: str = "mR") -> float:
    """Return oracle minus baseline for a metric key, in fractional units."""
    return float(oracle[metric]) - float(baseline[metric])
