"""Closest-work score controls over one frozen candidate support."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .legal_oracle import evidence_candidates


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float) / float(temperature)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.maximum(exp.sum(), np.finfo(float).tiny)


def _features(row: Mapping, predicate_counts: Mapping[int, int] | None = None) -> dict:
    scores = np.asarray(row["pred_scores"], dtype=float)[1:]
    probs = _softmax(scores)
    order = np.argsort(-probs, kind="stable")
    top = float(probs[order[0]]) if len(probs) else 0.0
    second = float(probs[order[1]]) if len(probs) > 1 else 0.0
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    predicate = int(order[0]) if len(order) else 0
    count = float((predicate_counts or {}).get(predicate, 1))
    return {
        "pair": float(row["score"]),
        "joint": float(row["score"]) * top,
        "top": top,
        "margin": top - second,
        "entropy": entropy,
        "frequency": np.log1p(count),
        "predicate": predicate,
        "probs": probs,
    }


def decode_rows(rows: Sequence[Mapping], budget: int, mode: str = "D0",
                predicate_counts: Mapping[int, int] | None = None,
                temperature: float = 1.0) -> list[dict]:
    """Decode <=K rows for D0-D7 while preserving unique directed pairs."""
    mode = str(mode).upper()
    if mode not in {f"D{i}" for i in range(8)}:
        raise ValueError(f"unknown simple control {mode}")
    candidates = evidence_candidates(rows, "all")
    scored = []
    for row in candidates:
        current = dict(row)
        feat = _features(row, predicate_counts)
        if mode == "D0":
            value = float(row["score"])
        elif mode == "D1":
            value = feat["joint"]
        elif mode == "D2":
            probs = _softmax(np.asarray(row["pred_scores"], dtype=float)[1:], temperature)
            current["pred"] = int(np.argmax(probs))
            current["pred_score"] = float(probs[current["pred"]])
            value = feat["pair"] * current["pred_score"]
        elif mode == "D3":
            # Inverse-frequency logit adjustment is deliberately exposed as a
            # control, not presented as a tuned method.
            value = feat["joint"] - 0.25 * feat["frequency"]
        elif mode == "D4":
            value = feat["joint"] + 0.25 / max(feat["frequency"], 1e-6)
        elif mode == "D5":
            value = feat["joint"] - 0.05 * feat["entropy"]
        elif mode == "D6":
            value = feat["pair"] * feat["margin"]
        else:  # D7: fixed, untrained linear diagnostic over controls.
            value = 0.45 * feat["joint"] + 0.25 * feat["margin"] - 0.05 * feat["entropy"] + 0.10 / max(feat["frequency"], 1e-6)
        current["decoder_score"] = float(value)
        scored.append(current)
    if mode == "D0":
        order = np.argsort(np.asarray([float(row["decoder_score"]) for row in scored]))[-int(budget):]
        return [scored[int(index)] for index in order]
    scored.sort(key=lambda x: (-x["decoder_score"], x["row"]))
    return scored[: int(budget)]


def predicate_histogram(relations: Sequence[Sequence[int]]) -> Counter:
    return Counter(int(relation[2]) for relation in relations)
