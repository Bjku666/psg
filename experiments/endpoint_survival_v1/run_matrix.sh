#!/usr/bin/env bash
set -euo pipefail

# Qualification-only matrix over frozen artifacts. Set CUDA_VISIBLE_DEVICES per
# invocation when a GPU is available; no trainable model is loaded here.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PSG="${PSG:?set PSG to the OpenPSG psg.json}"
MANIFEST="${MANIFEST:?set MANIFEST to the merged P0C manifest.jsonl}"
GT="${GT:?set GT to the panoptic GT annotations root}"
POP="${POP:-$ROOT/experiments/pregraph_attrition_v1/fit500_seed0.json}"
OUT="${OUT:-$ROOT/results/endpoint_survival_v1/fit500_seed0}"
mkdir -p "$OUT"

python "$ROOT/models/endpoint_survival_v1/e0_visual_difficulty_control.py" \
  --psg "$PSG" --manifest "$MANIFEST" --gt-seg-root "$GT" \
  --stage-audit "$ROOT/results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3/per_image.jsonl" \
  --population-manifest "$POP" --output "$OUT/e0_visual_control.json"

for mode in semantic competition; do
  for margin in 0.1 0.25 0.5 1.0 2.0; do
    tag="${mode}_$(echo "$margin" | tr . p)"
    python "$ROOT/models/endpoint_survival_v1/run_margin_oracle.py" \
      --psg "$PSG" --manifest "$MANIFEST" --gt-seg-root "$GT" \
      --population-manifest "$POP" --mode "$mode" --margin "$margin" \
      --device "${DEVICE:-cpu}" \
      --output "$OUT/$tag"
  done
done

python "$ROOT/models/endpoint_survival_v1/pq_support_frontier.py" \
  --inputs "$OUT"/semantic_*/summary.json "$OUT"/competition_*/summary.json \
  --output "$OUT/pq_support_frontier.json"
