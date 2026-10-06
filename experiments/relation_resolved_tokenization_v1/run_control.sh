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
REL_MICROBATCH=${REL_MICROBATCH:-32}
case "$ID" in
  C0_feature_all) CFG="$ROOT/experiments/relation_resolved_tokenization_v1/config_feature_all.json" ;;
  C1_patch4) CFG="$ROOT/experiments/relation_resolved_tokenization_v1/config_patch4.json" ;;
  *) echo "unknown control: $ID" >&2; exit 2 ;;
esac

OUT=/data1/liuhaoran/psg/checkpoints/relation_resolved_tokenization_v1/${ID}_seed0
LOG="$ROOT/experiments/relation_resolved_tokenization_v1/${ID}.log"
mkdir -p "$(dirname "$OUT")"
test -f "$ANNO"
test -d "$IMG/train2017"
test -d "$IMG/val2017"
test -d "$SEG/panoptic_train2017"
test -d "$SEG/panoptic_val2017"
test ! -e "$OUT/done.txt"

cd "$FAIR"
CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
PYTHONPATH="$FAIR:$ROOT" "$PY" \
  "$ROOT/models/relation_failure_decomp_v1/train_seeded.py" \
  "$CFG" "$OUT" --anno "$ANNO" --img "$IMG" --seg "$SEG" \
  --epochs 40 --workers 4 --no-bpbar \
  --cfg "rels_per_batch=$REL_MICROBATCH" 2>&1 | tee "$LOG"

date -u +%FT%TZ > "$OUT/done.txt"
