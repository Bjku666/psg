from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping
from models.gssr_p0c_v1.native_admission import pre_admission_state, upsampled_query_probabilities
from models.gssr_p0c_v1.raw_query_schema import decode_binary_mask


def select_items(psg: dict, population: Path | None = None, max_images: int | None = None) -> list[dict]:
    test = {str(x) for x in psg.get("test_image_ids", [])}
    items = [x for x in psg.get("data", []) if x.get("relations") and str(x["image_id"]) not in test]
    if population:
        d = json.loads(population.read_text())
        ids, files = {str(x) for x in d.get("image_ids", [])}, {str(x) for x in d.get("file_names", [])}
        items = [x for x in items if str(x["image_id"]) in ids or str(x["file_name"]) in files]
    items.sort(key=lambda x: str(x["image_id"]))
    return items[:max_images] if max_images else items


def load_records(manifest: Path) -> dict[str, dict]:
    out = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line); out[str(row["file_name"])] = row
    return out


def artifact_path(record: dict, manifest: Path) -> Path:
    p = Path(record["artifact_file"])
    return p if p.is_absolute() else manifest.parent / p


def raw_masks(record: dict, data: np.lib.npyio.NpzFile) -> np.ndarray:
    if record["queries"] and "mask_rle" in record["queries"][0]:
        return np.stack([decode_binary_mask(q["mask_rle"]) for q in record["queries"]])
    probs = upsampled_query_probabilities(torch.from_numpy(np.asarray(data["mask_logits"], dtype=np.float32)),
                                          (int(record["height"]), int(record["width"]))).numpy()
    return probs > 0.5


def relation_endpoints(relations: np.ndarray, n: int) -> set[int]:
    return {int(v) for s, o, _ in np.asarray(relations, dtype=int).reshape(-1, 3) for v in (s, o) if 0 <= int(v) < n}


def criticality(relations: np.ndarray, num_gt: int, frequencies: np.ndarray | None = None) -> np.ndarray:
    """Predicate-balanced relation mass per GT entity."""
    freq = np.ones(int(np.max(relations[:, 2]) + 1) if len(relations) else 1) if frequencies is None else np.asarray(frequencies)
    out = np.zeros(num_gt, dtype=float)
    for s, o, p in np.asarray(relations, dtype=int).reshape(-1, 3):
        w = 1.0 / max(float(freq[int(p)]), 1.0)
        if s != o:
            out[int(s)] += w; out[int(o)] += w
    return out


def biased_state(class_logits: np.ndarray, mask_logits: np.ndarray, target: tuple[int, int],
                 semantic_bias: np.ndarray | None = None, competition_bias: np.ndarray | None = None,
                 threshold: float = .5, mask_threshold: float = .5, overlap: float = .8,
                 device: str = "cpu") -> dict:
    """Native state with bounded, query-level interventions in log-score space."""
    cls = torch.as_tensor(np.asarray(class_logits, dtype=np.float32), device=device).clone()
    if semantic_bias is not None:
        bias = torch.as_tensor(np.asarray(semantic_bias, dtype=np.float32), device=device)
        # Structured oracle follows z_{q,y}+delta_s: raise only the query's
        # predicted foreground class, leaving competing foreground/no-object
        # logits unchanged.
        labels = cls[:, :-1].argmax(dim=-1)
        cls[torch.arange(len(cls)), labels] += bias
    mask_tensor = torch.as_tensor(np.asarray(mask_logits, dtype=np.float32), device=device)
    state = pre_admission_state(cls, mask_tensor, target,
                                threshold, mask_threshold, overlap)
    if competition_bias is None or not state["candidates"]:
        return state
    bias = np.asarray(competition_bias, dtype=float)
    probs = upsampled_query_probabilities(mask_tensor, target).cpu().numpy()
    scores, labels = torch.as_tensor(np.asarray(class_logits, dtype=np.float32), device=device).softmax(-1).max(-1)
    scores, labels = scores.cpu().numpy(), labels.cpu().numpy(); eligible = (labels != class_logits.shape[1] - 1) & (scores > threshold)
    ids = np.flatnonzero(eligible)
    # Use torch for exponentiation here; this environment's NumPy build can
    # raise while formatting ufunc operand errors after importing torch.
    scale = torch.exp(torch.as_tensor(bias, dtype=torch.float32)).numpy()
    weighted = probs * scores[:, None, None] * scale[:, None, None]
    winner = ids[weighted[ids].argmax(axis=0)].astype(np.int32) if len(ids) else np.full(target, -1, np.int32)
    candidates = []
    for q in ids:
        pixels = winner == q; won = sum(1 for x in pixels.ravel() if bool(x))
        if not won: continue
        original = sum(1 for x in weighted[q].ravel() if float(x) >= mask_threshold); ratio = won / original if original else 0.
        candidates.append({"query_id": int(q), "label_id": int(labels[q]), "score": float(scores[q]),
                           "won_area": won, "original_area": original, "area_ratio": ratio,
                           "native_keep": bool(ratio > overlap), "winning_mask": pixels})
    return {"winner_map": winner, "candidates": candidates}


def panoptic_quality(gt_mask: np.ndarray, gt_labels: np.ndarray, state: dict, threshold: float = .5) -> dict[str, float]:
    candidates = state["candidates"]; gt_areas = np.bincount(gt_mask[gt_mask >= 0], minlength=len(gt_labels))
    pred_masks = [c["winning_mask"] for c in candidates if c.get("native_keep", False)]
    pred_labels = [c["label_id"] for c in candidates if c.get("native_keep", False)]
    used = set(); tp = 0; sum_iou = 0.; fp = len(pred_masks); fn = len(gt_labels); class_stats = {"thing": [0, 0.], "stuff": [0, 0.]}
    for m, label in zip(pred_masks, pred_labels):
        if tuple(m.shape) != tuple(gt_mask.shape):
            raise ValueError(f"prediction/GT mask shape mismatch: {m.shape} vs {gt_mask.shape}")
        best, best_iou = -1, 0.
        for j, gl in enumerate(gt_labels):
            if j in used or int(gl) != int(label): continue
            tm = torch.as_tensor(m); tg = torch.as_tensor(gt_mask)
            inter = int(torch.logical_and(tm, tg == int(j)).sum().item()); union = int(tm.sum().item()) + int(gt_areas[j]) - inter; iou = inter / union if union else 0.
            if iou > best_iou: best, best_iou = j, iou
        if best >= 0 and best_iou > threshold:
            used.add(best); tp += 1; sum_iou += best_iou; fp -= 1; fn -= 1; bucket = "thing" if int(label) < 80 else "stuff"; class_stats[bucket][0] += 1; class_stats[bucket][1] += best_iou
    denom = tp + .5 * fp + .5 * fn; out = {"pq": (sum_iou / tp if tp else 0.) * (tp / denom if denom else 0.), "sq": sum_iou / tp if tp else 0., "rq": tp / denom if denom else 0., "tp": tp, "fp": fp, "fn": fn}
    for bucket, (btp, biou) in class_stats.items():
        n_pred = sum((int(l) < 80) == (bucket == "thing") for l in pred_labels); n_gt = sum((int(l) < 80) == (bucket == "thing") for l in gt_labels); bd = btp + .5*(n_pred-btp) + .5*(n_gt-btp)
        out[f"pq_{bucket[:2]}"] = (biou / btp if btp else 0.) * (btp / bd if bd else 0.)
    return out


def graph_proxy(relations: np.ndarray, mapping, selected: Iterable[int], num_predicates: int) -> dict[str, float]:
    selected = set(int(x) for x in selected); gt = set(int(x) for x in mapping.candidate_to_gt[list(selected)] if int(x) >= 0); rels = np.asarray(relations, dtype=int).reshape(-1, 3)
    supported = [(s, o, p) for s, o, p in rels if int(s) in gt and int(o) in gt and s != o]; total, hit = len(rels), len(supported); per = []
    for p in range(num_predicates):
        den = sum(int(r[2]) == p for r in rels); num = sum(int(r[2]) == p for r in supported)
        if den: per.append(num / den)
    return {"r@20": hit / total if total else 0., "r@50": hit / total if total else 0., "mr@20": float(np.mean(per)) if per else 0., "mr@50": float(np.mean(per)) if per else 0., "supported_relations": hit, "total_relations": total}
