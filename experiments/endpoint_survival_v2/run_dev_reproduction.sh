#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PSG="/data2/liuhaoran/psg_data/openpsg/psg/psg.json"
MANIFEST="/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-endpoint-survival-v2-dev-float32-merged/manifest.jsonl"
GT="/data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations"
POP="$ROOT/experiments/endpoint_survival_v2/dev_population.json"
OUT="$ROOT/results/endpoint_survival_v2/dev_frozen"
GPUS=(2 5 6)
SPECS=("native 0 native_0" "semantic 1.0 semantic_1p0" "competition 0.25 competition_0p25")
PIDS=()

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing merged dev float32 manifest: $MANIFEST" >&2
  exit 1
fi
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite $OUT" >&2
  exit 1
fi
mkdir -p "$OUT/logs"
for index in 0 1 2; do
  read -r mode margin tag <<<"${SPECS[$index]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" python "$ROOT/models/endpoint_survival_v2/run_margin_oracle.py" \
    --psg "$PSG" --manifest "$MANIFEST" --gt-seg-root "$GT" --population-manifest "$POP" \
    --mode "$mode" --margin "$margin" --device cuda --output "$OUT/$tag" \
    >"$OUT/logs/$tag.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

python "$ROOT/models/endpoint_survival_v2/pq_support_frontier.py" \
  --baseline "$OUT/native_0/summary.json" \
  --inputs "$OUT/semantic_1p0/summary.json" "$OUT/competition_0p25/summary.json" \
  --output "$OUT/pq_support_frontier.json" >"$OUT/logs/frontier.log" 2>&1

python "$ROOT/models/endpoint_survival_v2/analyze_frontier.py" \
  --baseline "$OUT/native_0/per_image.jsonl" \
  --points "$OUT/semantic_1p0/per_image.jsonl" "$OUT/competition_0p25/per_image.jsonl" \
  --semantic "$OUT/semantic_1p0/per_image.jsonl" \
  --competition "$OUT/competition_0p25/per_image.jsonl" \
  --output "$OUT/paired_bootstrap.json" >"$OUT/logs/bootstrap.log" 2>&1
