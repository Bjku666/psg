#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
FAIR=/data1/liuhaoran/psg/third_party/fair_psg_git_retry
PY=/home/user5/anaconda3/envs/adframe/bin/python
ANNO=/data1/liuhaoran/psg/datasets/psg.json
IMG=/data1/liuhaoran/psg/datasets/openpsg/coco
SEG=/data1/liuhaoran/psg/datasets/openpsg/coco_panoptic_gt/annotations
ID=${1:?control id}
GPU=${2:-0}
MODEL=/data1/liuhaoran/psg/checkpoints/relation_resolved_tokenization_v1/${ID}_seed0
OUT=/data1/liuhaoran/psg/results/relation_resolved_tokenization_v1
PRED="$OUT/${ID}_test.pkl"
METRICS="$OUT/${ID}_singlempo.csv"
mkdir -p "$OUT"
test -f "$MODEL/best_state.pth"
test -f "$ANNO"

cd "$FAIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTHONPATH="$FAIR:$ROOT" "$PY" \
  -m fair_psgg.tasks.inference "$ANNO" "$IMG" "$SEG" "$MODEL" "$PRED" \
  --bs 32 --workers 4 --split test
"$PY" scripts/evaluate.py "$ANNO" "$PRED" "$METRICS" --seg "$SEG" --dedup fail
