"""Observable row and candidate features for DASP."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np


def _observable(row: Mapping) -> dict:
    logits = np.asarray(row["pred_scores"], dtype=np.float32).reshape(-1)
    if logits.size != 56:
        raise ValueError(f"expected 56 predicate scores, got {logits.size}")
    probs = np.clip(logits, 1e-6, 1.0 - 1e-6)
    logit = np.log(probs / (1.0 - probs)).astype(np.float32)
    order = np.sort(logit)[::-1]
    margins = [float(order[0] - order[k]) for k in (1, 2) if len(order) > k]
    entropy = float(-(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs)).mean())
    scalar = np.asarray([
        float(row.get("pair_score", 0.0)),
        float(margins[0] if margins else 0.0),
        float(margins[1] if len(margins) > 1 else 0.0),
        entropy,
        float(logit[int(row["native"])]),
    ], dtype=np.float32)
    return {"hidden": np.asarray(row["pair_features"], dtype=np.float32),
            "logits": logit, "scalar": scalar, "native": int(row["native"]),
            "candidates": np.asarray(row["candidates"], dtype=np.int64),
            "candidate_delta_logits": logit[np.asarray(row["candidates"], dtype=np.int64)] - logit[int(row["native"])],
            "image_index": int(row["image_index"]), "row_id": int(row["row_id"]),
            "bootstrap_group": str(row["bootstrap_group"]),
            "pair_score": float(row.get("pair_score", 0.0))}


def row_features(rows: Sequence[Mapping]) -> list[dict]:
    """Convert teachers to inference features, dropping all GT fields."""
    output = [_observable(row) for row in rows]
    for row in output:
        if any(key in row for key in ("relations", "gt_pair", "utilities", "edit_label", "best_candidate")):
            raise AssertionError("ground-truth field leaked into model features")
    return output

