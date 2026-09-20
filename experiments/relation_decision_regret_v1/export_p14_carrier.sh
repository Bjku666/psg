#!/usr/bin/env bash
set -euo pipefail

# P1.4 full fit/dev export. Run one shard per GPU or invoke this script with
# explicit GPU/partition/shard bounds; it never touches confirm.
ROOT=/data2/liuhaoran/project/cv/psg
PY=/data2/liuhaoran/venvs/fair_psg_p0/bin/python
FAIR="$ROOT/third_party/fair_psg"
ANNO=/data2/liuhaoran/psg_data/openpsg/psg/psg.json
IMG=/data2/liuhaoran/psg_data/openpsg/coco
SEG=/data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations
MODEL=/data2/liuhaoran/psg_data/checkpoints/dsformer_relation_decomp_v1_seed0
MANIFEST="$ROOT/results/relation_decision_regret_v1/manifests/official_train_grouped_split_v2_seed0.json"
CACHE=/data2/liuhaoran/psg_data/cache/relation_decision_regret_v1/p14
PARTITION=${PARTITION:-fit}
SHARD_SIZE=${SHARD_SIZE:-500}
SHARD_ID=${SHARD_ID:-0}

mkdir -p "$CACHE/annotations/$PARTITION" "$CACHE/raw/$PARTITION" "$CACHE/compact/$PARTITION"
if [[ ! -f "$CACHE/annotations/$PARTITION/${PARTITION}_index.json" ]]; then
  "$PY" "$ROOT/models/relation_decision_regret_v1/make_partition_shards.py" \
    --psg "$ANNO" --split-manifest "$MANIFEST" --partition "$PARTITION" \
    --shard-size "$SHARD_SIZE" --output-dir "$CACHE/annotations/$PARTITION"
fi

SHARD_JSON="$CACHE/annotations/$PARTITION/${PARTITION}_$(printf '%03d' "$SHARD_ID").json"
RAW="$CACHE/raw/$PARTITION/${PARTITION}_$(printf '%03d' "$SHARD_ID").pkl"
OUT="$CACHE/compact/$PARTITION/${PARTITION}_$(printf '%03d' "$SHARD_ID").pkl"
test -f "$SHARD_JSON"
test -f "$MODEL/best_state.pth"
test ! -e "$OUT"

cd "$FAIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" PYTHONUNBUFFERED=1 PYTHONPATH="$FAIR:$ROOT" \
  "$PY" -m fair_psgg.tasks.inference "$SHARD_JSON" "$IMG" "$SEG" "$MODEL" "$RAW" \
  --bs 32 --workers 4 --split test --export-pair-features
cd "$ROOT"
"$PY" models/relation_decision_regret_v1/compact_carrier.py \
  --annotation "$SHARD_JSON" --predictions "$RAW" --gt-seg-root "$SEG" \
  --output "$OUT" --top-pairs 50

# Raw shard contains full masks and all directed pairs; the compact artifact is
# the registered continuation input. Remove only this exact completed shard.
rm -f "$RAW"
printf 'completed partition=%s shard=%s output=%s\n' "$PARTITION" "$SHARD_ID" "$OUT"
