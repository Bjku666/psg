#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
FAIR=/data1/liuhaoran/psg/third_party/fair_psg_git_retry
PY=/home/user5/anaconda3/envs/adframe/bin/python
ANNO=/data1/liuhaoran/psg/datasets/psg.json
IMG=/data1/liuhaoran/psg/datasets/openpsg/coco
SEG=/data1/liuhaoran/psg/datasets/openpsg/coco_panoptic_gt/annotations
MODE=${1:?uniform, mask, or predicate}
GPU=${2:?gpu}
CHECKPOINT=${3:?checkpoint}
OUT=${4:?output json}
IDS=${5:?development image-id JSON}

cd "$FAIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$FAIR:$ROOT" "$PY" "$ROOT/models/ambiguity_conditioned_relation_v1/eval_seeded_acrd.py" \
  --config "$ROOT/models/ambiguity_conditioned_relation_v1/config_acrd.json" \
  --checkpoint "$CHECKPOINT" --anno "$ANNO" --img "$IMG" --seg "$SEG" \
  --output "$OUT" --mode "$MODE" --gpu "$GPU" --workers 8 \
  --batch-size 112 --rel-chunk 224 --image-ids "$IDS"
