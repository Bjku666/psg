#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
FAIR=/data1/liuhaoran/psg/third_party/fair_psg_git_retry
PY=/home/user5/anaconda3/envs/adframe/bin/python
ANNO=/data1/liuhaoran/psg/datasets/psg.json
IMG=/data1/liuhaoran/psg/datasets/openpsg/coco
SEG=/data1/liuhaoran/psg/datasets/openpsg/coco_panoptic_gt/annotations
MODE=${1:?mode: uniform, mask, or predicate}
GPU=${2:?CUDA device}
EPOCHS=${3:-1}
RUN_TAG=${RUN_TAG:-smoke1}
SEED=${SEED:-0}
MAX_BATCHES=${MAX_BATCHES:-2}
WORKERS=${WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
REL_CHUNK=${REL_CHUNK:-32}
SKIP_EVAL=${SKIP_EVAL:-1}
MODEL_STATE=${MODEL_STATE:-}
OUT=/data1/liuhaoran/psg/checkpoints/ambiguity_conditioned_relation_v1/${MODE}_seed${SEED}_${RUN_TAG}
LOG="$ROOT/experiments/ambiguity_conditioned_relation_v1/${MODE}_seed${SEED}_${RUN_TAG}.log"

case "$MODE" in uniform|mask|predicate) ;; *) echo "invalid mode: $MODE" >&2; exit 2 ;; esac
test -f "$ANNO"
test -d "$IMG/train2017"
test -d "$IMG/val2017"
test -d "$SEG/panoptic_train2017"
test ! -e "$OUT/done.txt"
mkdir -p "$(dirname "$OUT")"

STATE_ARGS=()
if [[ -n "$MODEL_STATE" ]]; then
  test -f "$MODEL_STATE"
  STATE_ARGS+=(--model-state "$MODEL_STATE")
fi

cd "$FAIR"
CUDA_VISIBLE_DEVICES="$GPU" ACRD_MODE="$MODE" ACRD_SEED="$SEED" ACRD_TOP_K=16 ACRD_NUM_CANDIDATES=3 \
FAIR_PSG_AMP=1 FAIR_PSG_FUSED=1 FAIR_PSG_MAX_TRAIN_BATCHES="$MAX_BATCHES" FAIR_PSG_SKIP_EVAL="$SKIP_EVAL" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONPATH="$FAIR:$ROOT" \
"$PY" "$ROOT/models/ambiguity_conditioned_relation_v1/train_seeded_acrd.py" \
  "$ROOT/models/ambiguity_conditioned_relation_v1/config_acrd.json" "$OUT" \
  --anno "$ANNO" --img "$IMG" --seg "$SEG" --epochs "$EPOCHS" --workers "$WORKERS" --no-bpbar \
  "${STATE_ARGS[@]}" \
  --cfg "rels_per_batch=$REL_CHUNK" "batch_size=$BATCH_SIZE" 2>&1 | tee "$LOG"
date -u +%FT%TZ > "$OUT/done.txt"
