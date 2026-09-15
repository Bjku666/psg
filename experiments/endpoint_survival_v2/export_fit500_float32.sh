#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/data2/liuhaoran/venvs/fair_psg_p0/bin/python}"
IMAGES="${IMAGES:-/data2/iwin/datasets/ECCV/COCO/images/train2017}"
PSG="${PSG:-/data2/liuhaoran/psg_data/openpsg/psg/psg.json}"
MODEL="${MODEL:-/data2/liuhaoran/psg_data/models/mask2former-swin-large-coco-panoptic}"
POP_FILES="${POP_FILES:-$ROOT/experiments/pregraph_attrition_v1/fit500_seed0.files.txt}"
BASE="${BASE:-/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large}"
GPUS=(0 1 3 5 6 7)
PIDS=()

for shard in 0 1 2 3 4 5; do
  output="$BASE/psg-endpoint-survival-v2-fit500-float32-shard${shard}of6"
  log="$ROOT/results/endpoint_survival_v2/logs/export_float32_shard${shard}of6.log"
  mkdir -p "$ROOT/results/endpoint_survival_v2/logs"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON" "$ROOT/models/gssr_p0c_v1/export_raw_mask_queries.py" \
    --images "$IMAGES" --psg "$PSG" --model "$MODEL" --output "$output" \
    --split train --file-list "$POP_FILES" --num-shards 6 --shard-index "$shard" \
    --progress-every 10 --artifact-profile endpoint_survival_v2 --device cuda \
    >"$log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

"$PYTHON" "$ROOT/models/gssr_p0c_v1/merge_raw_query_shards.py" --psg "$PSG" \
  --shards "$BASE"/psg-endpoint-survival-v2-fit500-float32-shard{0,1,2,3,4,5}of6 \
  --output "$BASE/psg-endpoint-survival-v2-fit500-float32-merged" --split train \
  --population-manifest "$ROOT/experiments/pregraph_attrition_v1/fit500_seed0.json" \
  >"$ROOT/results/endpoint_survival_v2/logs/export_float32_merge.log" 2>&1
