#!/usr/bin/env bash
set -euo pipefail

# Resume-safe worker for the registered full fit/dev carrier export.  Invoke
# twice with parity 0/1 on distinct GPUs.  Completed compact shards are never
# overwritten and confirm is deliberately absent from the partition list.
ROOT=/data2/liuhaoran/project/cv/psg
CACHE=/data2/liuhaoran/psg_data/cache/relation_decision_regret_v1/p14
GPU_ID=${1:?usage: run_p14_full_remaining.sh GPU_ID PARITY}
PARITY=${2:?usage: run_p14_full_remaining.sh GPU_ID PARITY}

if [[ "$PARITY" != "0" && "$PARITY" != "1" ]]; then
  echo "PARITY must be 0 or 1" >&2
  exit 2
fi

run_partition() {
  local partition=$1
  local shards=$2
  local shard_id output
  for ((shard_id=PARITY; shard_id<shards; shard_id+=2)); do
    output="$CACHE/compact/$partition/${partition}_$(printf '%03d' "$shard_id").pkl"
    if [[ -f "$output" ]]; then
      echo "skip completed partition=$partition shard=$shard_id output=$output"
      continue
    fi
    echo "start partition=$partition shard=$shard_id gpu=$GPU_ID utc=$(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PARTITION="$partition" SHARD_ID="$shard_id" \
      bash "$ROOT/experiments/relation_decision_regret_v1/export_p14_carrier.sh"
  done
}

run_partition fit 66
run_partition dev 14
echo "worker complete gpu=$GPU_ID parity=$PARITY utc=$(date -u +%FT%TZ)"
