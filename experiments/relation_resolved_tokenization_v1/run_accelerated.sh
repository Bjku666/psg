#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
FAIR=/data1/liuhaoran/psg/third_party/fair_psg_git_retry
PY=/home/user5/anaconda3/envs/adframe/bin/python
ANNO=/data1/liuhaoran/psg/datasets/psg.json
IMG=/data1/liuhaoran/psg/datasets/openpsg/coco
SEG=/data1/liuhaoran/psg/datasets/openpsg/coco_panoptic_gt/annotations
ID=${1:?control id (C0_feature_all or C1_patch4)}
GPU=${2:?CUDA device}
REL_MICROBATCH=${3:?relation micro-batch}
EPOCHS=${4:-40}
MODEL_STATE=${5:-}
SUFFIX=${6:-amp_seed0}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUMULATE=${GRAD_ACCUMULATE:-1}
case "$ID" in
  C0_feature_all) CFG="$ROOT/experiments/relation_resolved_tokenization_v1/config_feature_all.json" ;;
  C1_patch4) CFG="$ROOT/experiments/relation_resolved_tokenization_v1/config_patch4.json" ;;
  *) echo "unknown control: $ID" >&2; exit 2 ;;
esac
OUT=/data1/liuhaoran/psg/checkpoints/relation_resolved_tokenization_v1/${ID}_${SUFFIX}
LOG="$ROOT/experiments/relation_resolved_tokenization_v1/${ID}_${SUFFIX}.log"
mkdir -p "$(dirname "$OUT")"
test -f "$ANNO"; test -d "$IMG/train2017"; test -d "$SEG/panoptic_train2017"
test ! -e "$OUT/done.txt"
STATE_ARGS=()
if [[ -n "$MODEL_STATE" ]]; then
  test -f "$MODEL_STATE"
  STATE_ARGS+=(--model-state "$MODEL_STATE")
fi
cd "$FAIR"
CUDA_VISIBLE_DEVICES="$GPU" FAIR_PSG_AMP=1 FAIR_PSG_FUSED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONUNBUFFERED=1 PYTHONPATH="$FAIR:$ROOT" "$PY" \
  "$ROOT/models/relation_failure_decomp_v1/train_seeded.py" "$CFG" "$OUT" \
  --anno "$ANNO" --img "$IMG" --seg "$SEG" --epochs "$EPOCHS" --workers 4 --no-bpbar \
  "${STATE_ARGS[@]}" --cfg "rels_per_batch=$REL_MICROBATCH" "batch_size=$BATCH_SIZE" "grad_accumulate=$GRAD_ACCUMULATE" 2>&1 | tee "$LOG"
date -u +%FT%TZ > "$OUT/done.txt"
