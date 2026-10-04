#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
FAIR=/data1/liuhaoran/psg/third_party/fair_psg_git_retry
PY=/home/user5/anaconda3/envs/adframe/bin/python
ANNO=/data1/liuhaoran/psg/datasets/psg.json
IMG=/data1/liuhaoran/psg/datasets/openpsg/coco
SEG=/data1/liuhaoran/psg/datasets/openpsg/coco_panoptic_gt/annotations
OUT=/data1/liuhaoran/psg/checkpoints/dsformer_relation_decomp_v1_seed0

test -f "$ANNO"
test -d "$IMG/train2017"
test -d "$IMG/val2017"
test -d "$SEG/panoptic_train2017"
test -d "$SEG/panoptic_val2017"
test ! -e "$OUT"
mkdir -p "$(dirname "$OUT")"

cd "$FAIR"
CUDA_VISIBLE_DEVICES=7 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
PYTHONPATH="$FAIR:$ROOT" "$PY" -m fair_psgg \
  configs/table2/masks-loc-sem.json "$OUT" \
  --anno "$ANNO" --img "$IMG" --seg "$SEG" \
  --epochs 40 --workers 4 --no-bpbar
