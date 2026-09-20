"""P1.3 factorization and Top-L control utilities.

The module keeps the carrier support fixed and makes the two decisions in the
qualification experiment explicit: which physical pairs occupy the budget and
which predicate hypothesis is emitted for each selected pair.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .legal_oracle import _predicate_options, _predicate_order, baseline_selection, legal_oracle_selection
from .official_metric_adapter import evaluate_population


def fixed_pair_predicate_oracle(rows: Sequence[Mapping], relations: Sequence[Sequence[int]],
                                budget: int, depth: int) -> list[dict]:
    """Keep baseline Top-K physical pairs and arbitrate predicate within Top-L."""
    gt = {(int(s), int(o), int(r)) for s, o, r in relations if int(s) != int(o)}
    selected = baseline_selection(rows, budget)
    out = []
    for source in selected:
        row = dict(source)
        options = _predicate_options(source, depth)
        mapped_pair = source.get("gt_pair")
        pair = tuple(map(int, mapped_pair)) if mapped_pair is not None else (-1, -1)
        matching = [p for p in options if (pair[0], pair[1], p) in gt]
        predicate = matching[0] if matching else options[0]
        scores = np.asarray(source["pred_scores"], dtype=float)
        order = _predicate_order(source)
        row.update({
            "pred": int(predicate),
            "pred_score": float(scores[predicate + 1]),
            "predicate_rank": int(next(i for i, value in enumerate(order) if int(value) == predicate + 1)),
            "evidence_depth": int(depth),
        })
        out.append(row)
    return out


def _top_l_hypotheses(rows: Sequence[Mapping], depth: int) -> list[dict]:
    out = []
    for source in rows:
        scores = np.asarray(source["pred_scores"], dtype=float)
        order = _predicate_options(source, depth)
        full_order = _predicate_order(source)
        for predicate in order:
            row = dict(source)
            row.update({
                "pred": int(predicate),
                "pred_score": float(scores[predicate + 1]),
                "predicate_rank": int(next(i for i, value in enumerate(full_order) if int(value) == predicate + 1)),
                "evidence_depth": int(depth),
            })
            out.append(row)
    return out


def decode_top_l(rows: Sequence[Mapping], relations: Sequence[Sequence[int]], budget: int,
                 depth: int, mode: str = "L0", predicate_counts: Mapping[int, int] | None = None,
                 alpha: float = 0.5, beta: float = 1.0, gamma: float = 0.0) -> list[dict]:
    """Run a fit-free Top-L decoder proxy.

    These deterministic modes are implementation/protocol smoke tests.  They
    are not the fit-trained L0--L4 controls in the P1.3 experiment plan.
    ``predicate_counts`` must come from a training population; evaluation GT
    relations are deliberately ignored to prevent label leakage.
    """
    mode = str(mode).upper()
    if mode not in {f"L{i}" for i in range(5)}:
        raise ValueError(f"unknown Top-L control {mode}")
    counts = Counter(predicate_counts or {})
    hypotheses = _top_l_hypotheses(rows, depth)
    if not hypotheses:
        return []
    # One legal hypothesis per directed pair.  Stable ordering makes ties
    # reproducible and preserves the carrier row as final tie-break.
    best: dict[tuple[int, int], dict] = {}
    priors = np.asarray([float(counts.get(p, 1)) for p in range(56)], dtype=float)
    priors /= max(priors.sum(), 1.0)
    for row in hypotheses:
        scores = np.asarray(row["pred_scores"], dtype=float)[1:]
        pred = int(row["pred"])
        logits = scores.copy()
        if mode == "L0":
            value = float(row["score"])
        elif mode == "L1":
            temperature = 1.0 + 0.5 * np.log1p(float(counts.get(pred, 1)))
            value = float(row["score"]) + float(logits[pred] / temperature)
        elif mode == "L2":
            value = float(row["score"]) + float(logits[pred] - alpha * np.log(max(priors[pred], 1e-12)))
        elif mode == "L3":
            value = float(row["score"]) + beta * float(logits[pred]) + gamma * float(np.log(max(1.0 / priors[pred], 1e-12)))
        else:
            # Fit-free multinomial calibration proxy: pair evidence and
            # predicate evidence are jointly normalized within each pair.
            value = float(row["score"]) * (1.0 / (1.0 + np.exp(-float(logits[pred]))))
        row = dict(row)
        row["decoder_score"] = value
        key = tuple(map(int, row["pair"]))
        old = best.get(key)
        if old is None or (value, -int(row["row"]), -int(row["predicate_rank"])) > (old["decoder_score"], -int(old["row"]), -int(old["predicate_rank"])):
            best[key] = row
    ordered = sorted(best.values(), key=lambda x: (-float(x["decoder_score"]), int(x["row"]), int(x["pred"])))
    return ordered[: int(budget)]


def population_action_weights(image_records: Sequence[Mapping], num_predicates: int) -> list[dict]:
    """Attach exact additive mR/R weights under the pinned population metric."""
    n_images = len(image_records)
    valid_image_counts = np.zeros(num_predicates, dtype=int)
    for image in image_records:
        counts = Counter(int(r[2]) for r in image.get("relations", ()))
        for predicate in counts:
            if 0 <= predicate < num_predicates:
                valid_image_counts[predicate] += 1
    out = []
    for image in image_records:
        gt = [(int(s), int(o), int(r)) for s, o, r in image.get("relations", ())]
        gt_counts = Counter(r for _, _, r in gt)
        total = len(gt)
        for candidate in image.get("candidates", ()):
            mapped_pair = candidate.get("gt_pair")
            pair = tuple(map(int, mapped_pair)) if mapped_pair is not None else (-1, -1)
            predicate = int(candidate["pred"])
            hits = sum(1 for s, o, r in gt if (s, o, r) == (pair[0], pair[1], predicate))
            mr_weight = (hits / gt_counts[predicate] / valid_image_counts[predicate] / num_predicates
                          if hits and gt_counts[predicate] and valid_image_counts[predicate] else 0.0)
            r_weight = hits / total / n_images if hits and total and n_images else 0.0
            row = dict(candidate)
            row.update({"image_id": str(image.get("image_id", "")), "mR_weight": float(mr_weight), "R_weight": float(r_weight)})
            out.append(row)
    return out


def additivity_check(image_records: Sequence[Mapping], num_predicates: int,
                     budget: int = 20, trials: int = 128, seed: int = 0) -> dict:
    """Compare exact evaluator deltas with analytic candidate-weight deltas."""
    rng = np.random.default_rng(seed)
    residuals_mr, residuals_r = [], []
    for _ in range(int(trials)):
        image = image_records[int(rng.integers(0, len(image_records)))]
        candidates = []
        for candidate in image.get("candidates", ()):
            row = dict(candidate)
            if "pred" not in row:
                options = _predicate_options(row, 1)
                if not options:
                    continue
                row["pred"] = int(options[0])
            candidates.append(row)
        if len(candidates) < 2:
            continue
        k = min(int(budget), len(candidates))
        base = [dict(x) for x in candidates[:k]]
        replacement = candidates[int(rng.integers(0, len(candidates)))]
        if replacement in base:
            continue
        changed = base[:-1] + [dict(replacement)]
        base_record = {"image_id": image["image_id"], "relations": image["relations"], "selected": base, "budget": k}
        new_record = {"image_id": image["image_id"], "relations": image["relations"], "selected": changed, "budget": k}
        # Evaluate just this image; scale by the same population factors used
        # in population_action_weights.
        old_eval = evaluate_population([base_record], num_predicates)
        new_eval = evaluate_population([new_record], num_predicates)
        def local_mr(eval_result):
            return float(np.nanmean(eval_result["per_image"][0]["per_predicate_recall"]))
        gt = [(int(s), int(o), int(r)) for s, o, r in image.get("relations", ())]
        gt_counts = Counter(r for _, _, r in gt)
        n_valid = max(1, len(gt_counts))
        def local_weight(candidate):
            mapped_pair = candidate.get("gt_pair")
            pair = tuple(map(int, mapped_pair)) if mapped_pair is not None else (-1, -1)
            predicate = int(candidate["pred"])
            hits = sum(1 for s, o, r in gt if (s, o, r) == (pair[0], pair[1], predicate))
            return (hits / gt_counts[predicate] / n_valid if hits and gt_counts[predicate] else 0.0,
                    hits / len(gt) if hits and gt else 0.0)
        old_mr = sum(local_weight(x)[0] for x in base); new_mr = sum(local_weight(x)[0] for x in changed)
        old_r = sum(local_weight(x)[1] for x in base); new_r = sum(local_weight(x)[1] for x in changed)
        residuals_mr.append((local_mr(new_eval) - local_mr(old_eval)) - (new_mr - old_mr))
        residuals_r.append((float(new_eval["R@%d" % k]) - float(old_eval["R@%d" % k])) - (new_r - old_r))
    return {"trials": len(residuals_mr), "max_abs_epsilon_mR": float(np.max(np.abs(residuals_mr))) if residuals_mr else None,
            "max_abs_epsilon_R": float(np.max(np.abs(residuals_r))) if residuals_r else None,
            "mean_abs_epsilon_mR": float(np.mean(np.abs(residuals_mr))) if residuals_mr else None,
            "mean_abs_epsilon_R": float(np.mean(np.abs(residuals_r))) if residuals_r else None}
