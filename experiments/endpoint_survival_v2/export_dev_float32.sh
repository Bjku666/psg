#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/data2/liuhaoran/venvs/fair_psg_p0/bin/python}"
IMAGES="/data2/iwin/datasets/ECCV/COCO/images/train2017"
PSG="/data2/liuhaoran/psg_data/openpsg/psg/psg.json"
MODEL="/data2/liuhaoran/psg_data/models/mask2former-swin-large-coco-panoptic"
FILES="$ROOT/experiments/endpoint_survival_v2/dev_population.files.txt"
POP="$ROOT/experiments/endpoint_survival_v2/dev_population.json"
BASE="/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large"
GPUS=(2 5 6 7)
PIDS=()

for shard in 0 1 2 3; do
  output="$BASE/psg-endpoint-survival-v2-dev-float32-shard${shard}of4"
  log="$ROOT/results/endpoint_survival_v2/logs/export_dev_float32_shard${shard}of4.log"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON" "$ROOT/models/gssr_p0c_v1/export_raw_mask_queries.py" \
    --images "$IMAGES" --psg "$PSG" --model "$MODEL" --output "$output" \
    --split train --file-list "$FILES" --num-shards 4 --shard-index "$shard" \
    --progress-every 50 --artifact-profile endpoint_survival_v2 --device cuda \
    >"$log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

"$PYTHON" "$ROOT/models/gssr_p0c_v1/merge_raw_query_shards.py" --psg "$PSG" \
  --shards "$BASE"/psg-endpoint-survival-v2-dev-float32-shard{0,1,2,3}of4 \
  --output "$BASE/psg-endpoint-survival-v2-dev-float32-merged" --split train \
  --population-manifest "$POP" >"$ROOT/results/endpoint_survival_v2/logs/export_dev_float32_merge.log" 2>&1
