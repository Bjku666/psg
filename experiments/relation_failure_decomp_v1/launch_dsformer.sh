#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
FAIR="$ROOT/third_party/fair_psg"
PY=/data2/liuhaoran/venvs/fair_psg_p0/bin/python
ANNO=/data2/liuhaoran/psg_data/openpsg/psg/psg.json
IMG=/data2/liuhaoran/psg_data/openpsg/coco
SEG=/data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations
OUT=/data2/liuhaoran/psg_data/checkpoints/dsformer_relation_decomp_v1_seed0
PRED_ANNO=/data2/liuhaoran/psg_data/inferred_masks/mask2former_swin_large/psg-test-mask2former-swin-large-finalpanoptic-fulltest-seed0-r4/matched_relation_decomp_v1.json
PRED_SEG=/data2/liuhaoran/psg_data/inferred_masks/mask2former_swin_large
CARRIER_DIR=/data2/liuhaoran/psg_data/relation_failure_decomp_v1
CARRIER="$CARRIER_DIR/dsformer_sgdet_full_pairs.pkl"
METRICS="$ROOT/results/relation_failure_decomp_v1/dsformer_sgdet_singlempo.csv"
DECOMP="$ROOT/results/relation_failure_decomp_v1/decomposition.json"

test -f "$ANNO"
test -d "$IMG/train2017"
test -d "$IMG/val2017"
test -d "$SEG/panoptic_train2017"
test -d "$SEG/panoptic_val2017"
test -f "$PRED_ANNO"
test ! -e "$OUT"
mkdir -p "$CARRIER_DIR"

cd "$FAIR"
CUDA_VISIBLE_DEVICES=7 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
PYTHONPATH="$FAIR:$ROOT" "$PY" \
  "$ROOT/models/relation_failure_decomp_v1/train_seeded.py" \
  configs/table2/masks-loc-sem.json "$OUT" \
  --anno "$ANNO" --img "$IMG" --seg "$SEG" \
  --epochs 40 --workers 4 --no-bpbar

CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 "$PY" \
  -m fair_psgg.tasks.inference \
  "$PRED_ANNO" "$IMG" "$PRED_SEG" "$OUT" "$CARRIER" \
  --bs 32 --workers 4 --split test

"$PY" scripts/evaluate.py \
  "$ANNO" "$CARRIER" "$METRICS" --seg "$SEG" --dedup fail

cd "$ROOT"
"$PY" models/relation_failure_decomp_v1/run_decomposition.py \
  --psg "$ANNO" --predictions "$CARRIER" --gt-seg-root "$SEG" \
  --output "$DECOMP"
