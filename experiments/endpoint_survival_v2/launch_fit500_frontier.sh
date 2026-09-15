#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PSG="/data2/liuhaoran/psg_data/openpsg/psg/psg.json"
MANIFEST="/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-endpoint-survival-v2-fit500-float32-merged/manifest.jsonl"
GT="/data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations"
POP="$ROOT/experiments/pregraph_attrition_v1/fit500_seed0.json"
OUT="$ROOT/results/endpoint_survival_v2/fit500_seed0"
GPUS=(0 1 3 5 6 7)

if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite $OUT" >&2
  exit 1
fi
mkdir -p "$OUT/logs"

run_wave() {
  local specs=("$@")
  local pids=()
  local index=0
  for spec in "${specs[@]}"; do
    read -r mode margin tag <<<"$spec"
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" python "$ROOT/models/endpoint_survival_v2/run_margin_oracle.py" \
      --psg "$PSG" --manifest "$MANIFEST" --gt-seg-root "$GT" \
      --population-manifest "$POP" --mode "$mode" --margin "$margin" --device cuda \
      --output "$OUT/$tag" >"$OUT/logs/$tag.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

run_wave \
  "native 0 native_0" \
  "semantic 0.1 semantic_0p1" \
  "semantic 0.25 semantic_0p25" \
  "semantic 0.5 semantic_0p5" \
  "semantic 1.0 semantic_1p0" \
  "semantic 2.0 semantic_2p0"

run_wave \
  "competition 0.1 competition_0p1" \
  "competition 0.25 competition_0p25" \
  "competition 0.5 competition_0p5" \
  "competition 1.0 competition_1p0" \
  "competition 2.0 competition_2p0"

python "$ROOT/models/endpoint_survival_v2/pq_support_frontier.py" \
  --baseline "$OUT/native_0/summary.json" \
  --inputs "$OUT"/semantic_*/summary.json "$OUT"/competition_*/summary.json \
  --output "$OUT/pq_support_frontier.json" >"$OUT/logs/frontier.log" 2>&1
