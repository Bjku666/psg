#!/usr/bin/env python3
"""Check whether a frozen carrier contains deployable EVA evidence.

EVA needs candidate-conditioned access to subject/object/context visual tokens.
The p1.4 compact contract intentionally retains only pair features and logits,
so this audit prevents silently calling a score-only reranker an EVA model.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


VISUAL_KEYS = {
    "visual_map", "feature_map", "fmap", "visual_tokens", "roi_features",
    "subject_features", "object_features", "context_features",
}


def inspect(path: Path) -> dict:
    with path.open("rb") as stream:
        carrier = pickle.load(stream)
    images = carrier.get("images", [])
    observed = set()
    for image in images[: min(32, len(images))]:
        for row in image.get("candidates", []):
            observed.update(row.keys())
    visual = sorted(observed & VISUAL_KEYS)
    return {
        "path": str(path),
        "images": len(images),
        "observed_candidate_keys": sorted(observed),
        "visual_evidence_keys": visual,
        "eva_ready": bool(visual),
        "reason": ("candidate-specific visual evidence is present"
                   if visual else "compact carrier stores logits/pair features only"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema_version": 1, "contract": "EVA visual-evidence preflight",
              "carriers": [inspect(path) for path in args.carrier],
              "status": "authorized_to_implement" if all(inspect(path)["eva_ready"] for path in args.carrier)
              else "blocked_missing_visual_evidence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
