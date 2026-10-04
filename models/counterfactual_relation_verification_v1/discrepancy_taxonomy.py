"""A3b taxonomy for paired OpenPSG references.

This is a CPU-only diagnostic.  It uses the actual panoptic masks to map the
second reference into the first reference's entity index space and never
deletes, deduplicates, or rewrites source annotations.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def panoptic_id(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0].astype(np.int64) + 256 * rgb[:, :, 1].astype(np.int64) + 256 * 256 * rgb[:, :, 2].astype(np.int64)


def mask_mapping(a: dict[str, Any], b: dict[str, Any], root: Path) -> dict[int, int]:
    pa, pb = root / a["pan_seg_file_name"], root / b["pan_seg_file_name"]
    if not pa.is_file() or not pb.is_file():
        return {}
    ma, mb = panoptic_id(np.asarray(Image.open(pa))), panoptic_id(np.asarray(Image.open(pb)))
    candidates: list[tuple[float, int, int]] = []
    for ia, aa in enumerate(a.get("annotations", [])):
        sid_a = int(a.get("segments_info", [])[ia]["id"])
        for ib, bb in enumerate(b.get("annotations", [])):
            if aa.get("category_id") != bb.get("category_id"):
                continue
            sid_b = int(b.get("segments_info", [])[ib]["id"])
            x, y = ma == sid_a, mb == sid_b
            union = int(np.logical_or(x, y).sum())
            score = int(np.logical_and(x, y).sum()) / union if union else 0.0
            if score > 0.5:
                candidates.append((score, ia, ib))
    out: dict[int, int] = {}
    used: set[int] = set()
    for score, ia, ib in sorted(candidates, reverse=True):
        if ia not in out and ib not in used:
            out[ia] = ib
            used.add(ib)
    return out


def relation_map(row: dict[str, Any], to_canonical: dict[int, int] | None = None) -> dict[tuple[int, int], set[int]]:
    out: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for s, o, p in row.get("relations", []):
        if to_canonical is not None:
            if s not in to_canonical or o not in to_canonical:
                continue
            s, o = to_canonical[s], to_canonical[o]
        out[(int(s), int(o))].add(int(p))
    return out


# Strict, high-confidence incompatibilities among predicates in OpenPSG.  T4
# is reported as a subset of T2, never as a replacement for the broader class.
CONFLICTS = {
    frozenset((3, 4)),   # on / in
    frozenset((3, 7)),   # on / on back of
    frozenset((14, 15)), # standing on / lying on
    frozenset((45, 46)), # driving / riding
    frozenset((52, 53)), # entering / exiting
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psg", type=Path, required=True)
    ap.add_argument("--panoptic-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = json.loads(args.psg.read_text())
    by_file: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in source["data"]:
        by_file[row["file_name"]].append(row)
    groups = {k: v for k, v in by_file.items() if len(v) == 2}
    counts = collections.Counter()
    examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    total_triples = 0
    total_pairs = 0
    mapped_groups = 0
    for file_name, (a, b) in sorted(groups.items()):
        ab = mask_mapping(a, b, args.panoptic_root)
        if len(ab) != len(a.get("annotations", [])) or len(ab) != len(b.get("annotations", [])):
            continue
        mapped_groups += 1
        # A is the canonical coordinate system; invert A -> B for reference B.
        ba = {v: k for k, v in ab.items()}
        ra, rb = relation_map(a), relation_map(b, ba)
        for pair in set(ra) | set(rb):
            pa, pb = ra.get(pair, set()), rb.get(pair, set())
            if pa == pb:
                continue
            total_pairs += 1
            if not pa or not pb:
                kind = "T1_missing_edge"
                counts[kind + "_pairs"] += 1
                counts[kind + "_triples"] += len(pa ^ pb)
            elif pa < pb or pb < pa:
                kind = "T3_multilabel_subset"
                counts[kind + "_pairs"] += 1
                counts[kind + "_triples"] += len(pa ^ pb)
            else:
                kind = "T2_same_pair_different_predicate"
                counts[kind + "_pairs"] += 1
                counts[kind + "_triples"] += len(pa ^ pb)
                if any(frozenset((x, y)) in CONFLICTS for x in pa for y in pb):
                    counts["T4_semantic_conflict_pairs"] += 1
                    counts["T4_semantic_conflict_triples"] += len(pa ^ pb)
                    kind = "T4_semantic_conflict"
            total_triples += len(pa ^ pb)
            if len(examples[kind]) < 10:
                examples[kind].append({"file_name": file_name, "pair": list(pair), "ref_a": sorted(pa), "ref_b": sorted(pb)})
    report = {
        "schema_version": 1,
        "name": "reference_ambiguity_audit_v1_A3b",
        "status": "cpu_complete",
        "source": {"psg": str(args.psg), "dataset_sha256": sha256(args.psg)},
        "population": {"paired_groups": len(groups), "fully_mapped_groups": mapped_groups},
        "totals": {"symmetric_delta_triples": total_triples, "affected_pairs": total_pairs},
        "taxonomy": dict(counts),
        "examples": dict(examples),
        "conflict_pairs": [sorted(x) for x in sorted(CONFLICTS, key=lambda x: tuple(sorted(x)))],
        "decision": "A4_required_before_method_training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
