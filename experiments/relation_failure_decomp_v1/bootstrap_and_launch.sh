#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
COCO=/data2/liuhaoran/psg_data/openpsg/coco
ZIP="$COCO/train2017.zip"
URL=https://images.cocodataset.org/zips/train2017.zip
EXPECTED_BYTES=19336861798

mkdir -p "$COCO"
wget --no-check-certificate -c --progress=dot:giga "$URL" -O "$ZIP"

actual_bytes=$(stat -c %s "$ZIP")
test "$actual_bytes" -eq "$EXPECTED_BYTES"
unzip -tq "$ZIP"
unzip -q -n "$ZIP" -d "$COCO"

test -f "$COCO/train2017/000000417720.jpg"

"$ROOT/experiments/relation_failure_decomp_v1/launch_dsformer.sh"
