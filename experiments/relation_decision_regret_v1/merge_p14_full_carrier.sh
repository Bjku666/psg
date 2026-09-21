#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
CACHE=/data2/liuhaoran/psg_data/cache/relation_decision_regret_v1/p14
PY=/data2/liuhaoran/venvs/fair_psg_p0/bin/python
OUT="$ROOT/results/relation_decision_regret_v1/p14_full_carrier"

expected_count() {
  local partition=$1
  if [[ "$partition" == fit ]]; then echo 66; else echo 14; fi
}

for partition in fit dev; do
  expected=$(expected_count "$partition")
  actual=$(find "$CACHE/compact/$partition" -maxdepth 1 -type f -name "${partition}_*.pkl" | wc -l)
  if [[ "$actual" -ne "$expected" ]]; then
    echo "incomplete partition=$partition actual=$actual expected=$expected" >&2
    exit 1
  fi
done

mkdir -p "$OUT"
"$PY" "$ROOT/models/relation_decision_regret_v1/merge_compact_carrier.py" \
  --input-dir "$CACHE/compact/fit" --partition fit \
  --output "$OUT/fit_full.pkl"
"$PY" "$ROOT/models/relation_decision_regret_v1/merge_compact_carrier.py" \
  --input-dir "$CACHE/compact/dev" --partition dev \
  --output "$OUT/dev_full.pkl"

sha256sum "$OUT/fit_full.pkl" "$OUT/dev_full.pkl" | tee "$OUT/SHA256SUMS"
python - <<'PY'
import pickle
from pathlib import Path
root = Path('/data2/liuhaoran/project/cv/psg/results/relation_decision_regret_v1/p14_full_carrier')
for name, expected in [('fit_full.pkl', 32590), ('dev_full.pkl', 6990)]:
    with (root / name).open('rb') as stream:
        document = pickle.load(stream)
    assert len(document['images']) == expected, (name, len(document['images']))
    assert len({str(item['image_id']) for item in document['images']}) == expected
    print(name, 'images=', expected, 'predicates=', len(document['predicate_classes']))
PY
