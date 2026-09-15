#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PSG="${PSG:?set PSG to OpenPSG psg.json}"
MANIFEST="${MANIFEST:?set MANIFEST to the frozen fit500 raw-query manifest}"
GT="${GT:?set GT to the panoptic GT annotations root}"
POP="${POP:-$ROOT/experiments/pregraph_attrition_v1/fit500_seed0.json}"
OUT="${OUT:-$ROOT/results/endpoint_survival_v2/fit500_seed0}"
DEVICE="${DEVICE:-cpu}"
mkdir -p "$OUT/logs"

python "$ROOT/models/endpoint_survival_v2/e0_visual_difficulty_control.py" --psg "$PSG" --manifest "$MANIFEST" --gt-seg-root "$GT" \
  --stage-audit "$ROOT/results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3/per_image.jsonl" \
  --population-manifest "$POP" --output "$OUT/e0_visual_control.json" >"$OUT/logs/e0_visual_control.log" 2>&1

python "$ROOT/models/endpoint_survival_v2/run_margin_oracle.py" --psg "$PSG" --manifest "$MANIFEST" \
  --gt-seg-root "$GT" --population-manifest "$POP" --mode native --margin 0 --device "$DEVICE" \
  --output "$OUT/native_0" >"$OUT/logs/native_0.log" 2>&1

for mode in semantic competition; do
  for margin in 0.1 0.25 0.5 1.0 2.0; do
    tag="${mode}_$(printf '%s' "$margin" | tr . p)"
    python "$ROOT/models/endpoint_survival_v2/run_margin_oracle.py" --psg "$PSG" --manifest "$MANIFEST" \
      --gt-seg-root "$GT" --population-manifest "$POP" --mode "$mode" --margin "$margin" \
      --device "$DEVICE" --output "$OUT/$tag" >"$OUT/logs/$tag.log" 2>&1
  done
done

python "$ROOT/models/endpoint_survival_v2/pq_support_frontier.py" --baseline "$OUT/native_0/summary.json" \
  --inputs "$OUT"/semantic_*/summary.json "$OUT"/competition_*/summary.json \
  --output "$OUT/pq_support_frontier.json" >"$OUT/logs/frontier.log" 2>&1
