#!/usr/bin/env python3
"""Replay native and counterfactual query admissions from a P0C artifact.

This is the executable P1A hook.  It reuses the exact frozen native admission
implementation and therefore applies confidence filtering, pixel competition,
and overlap filtering after restricting the decoder query IDs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from models.gssr_p0c_v1.native_admission import native_panoptic_admission
from models.gssr_p1_v1.panoptic_replay import replay_queries
from models.gssr_p1_v1.replay_metrics import validate_native_replay

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--record", required=True, type=Path, help="P0C image record JSON or manifest.jsonl")
    p.add_argument("--oracle-ids", required=True, type=Path, help="JSON list of raw query IDs")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--overlap-mask-area-threshold", type=float, default=0.8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw_record = args.record.read_text().splitlines()[0] if args.record.suffix == ".jsonl" else args.record.read_text()
    record = json.loads(raw_record)
    artifact = Path(record["artifact_file"])
    if not artifact.is_absolute(): artifact = args.record.parent / artifact
    with np.load(artifact) as data:
        # ``torch.from_numpy`` is unavailable in the lightweight audit env
        # when NumPy ABI versions differ; ``tensor`` performs a safe copy.
        class_logits = torch.tensor(data["class_logits"].astype(np.float32))
        mask_logits = torch.tensor(data["mask_logits"].astype(np.float32))
        stored = data.get("native_panoptic_segmentation")
    pool = record["queries"]
    native_ids = [q["query_id"] for q in pool if q.get("official_keep_flag")]
    oracle_ids = json.loads(args.oracle_ids.read_text())
    def backend(selected):
        ids = [int(q["query_id"]) for q in selected]
        idx = torch.tensor(ids, dtype=torch.long)
        segmentation, segments = native_panoptic_admission(class_logits[idx], mask_logits[idx],
            (int(record["height"]), int(record["width"])), args.threshold,
            args.mask_threshold, args.overlap_mask_area_threshold)
        for segment in segments:
            segment["query_id"] = ids[int(segment["query_id"])]
        return segmentation, segments
    native = replay_queries(pool, native_ids, backend)
    if stored is not None:
        official = type(native)(np.asarray(stored), native.segments, native.allowed_query_ids)
        validate_native_replay(official, native)
    oracle = replay_queries(pool, oracle_ids, backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "record": str(args.record), "native_ids": list(native.allowed_query_ids),
        "oracle_ids": list(oracle.allowed_query_ids),
        "native_segment_count": native.segment_count,
        "oracle_segment_count": oracle.segment_count,
        "native_segments": list(native.segments), "oracle_segments": list(oracle.segments),
        "native_segmentation": native.segmentation.tolist(),
        "oracle_segmentation": oracle.segmentation.tolist()
    }, indent=2) + "\n")

if __name__ == "__main__":
    main()
