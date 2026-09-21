#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/liuhaoran/project/cv/psg
CACHE=/data2/liuhaoran/psg_data/cache/relation_decision_regret_v1/p14
PY=/data2/liuhaoran/venvs/fair_psg_p0/bin/python
RESULTS="$ROOT/results/relation_decision_regret_v2"
FULL="$ROOT/results/relation_decision_regret_v1/p14_full_carrier"
LOG="$RESULTS/p17_after_export.log"
mkdir -p "$RESULTS"
exec > >(tee -a "$LOG") 2>&1

echo "waiting for registered 66 fit + 14 dev compact shards"
while true; do
  fit=$(find "$CACHE/compact/fit" -maxdepth 1 -type f -name 'fit_*.pkl' | wc -l)
  dev=$(find "$CACHE/compact/dev" -maxdepth 1 -type f -name 'dev_*.pkl' | wc -l)
  echo "$(date '+%F %T %Z') fit=$fit/66 dev=$dev/14"
  [[ "$fit" -eq 66 && "$dev" -eq 14 ]] && break
  sleep 60
done

bash "$ROOT/experiments/relation_decision_regret_v1/merge_p14_full_carrier.sh"

echo "Gate 0 + full-dev phenomenon"
"$PY" "$ROOT/models/relation_decision_regret_v2/run_p17_validity.py" \
  --carrier "$FULL/dev_full.pkl" \
  --output "$RESULTS/p17_gate0_full_dev.json" \
  --budget 50 --depth 3 --parity-actions 256 --trials-per-size 16 --seed 0

gate0=$(
  "$PY" -c "import json; print(json.load(open('$RESULTS/p17_gate0_full_dev.json'))['gate0']['pass'])"
)
headroom=$(
  "$PY" -c "import json; print(json.load(open('$RESULTS/p17_gate0_full_dev.json'))['results']['delta_mR_pp'])"
)
if [[ "$gate0" != "True" ]]; then
  echo "STOP: Gate 0 failed; metric-exact formulation is not authorized"
  exit 0
fi
if "$PY" -c "import sys; sys.exit(0 if float('$headroom') < 3.0 else 1)"; then
  echo "STOP: full-dev surgery headroom ${headroom}pp < 3pp"
  exit 0
fi

echo "fit-only S0-S7 controls"
"$PY" "$ROOT/models/relation_decision_regret_v2/run_p17_controls.py" \
  --fit-carrier "$FULL/fit_full.pkl" --dev-carrier "$FULL/dev_full.pkl" \
  --output "$RESULTS/p17_controls_full_fit_dev.json" \
  --decisions-output "$RESULTS/p17_controls_full_decisions.pkl" \
  --budget 50 --depth 3 --seed 0 --learned-group-cap 200000

decision=$(
  "$PY" -c "import json; print(json.load(open('$RESULTS/p17_controls_full_fit_dev.json'))['gate2']['decision'])"
)
if [[ "$decision" != "full MEAR authorized" ]]; then
  echo "STOP: Gate 2 decision=$decision"
  exit 0
fi

echo "MEAR seed0 on full fit/dev"
CUDA_VISIBLE_DEVICES=6 "$PY" "$ROOT/models/relation_decision_regret_v2/run_p17_mear.py" \
  --fit-carrier "$FULL/fit_full.pkl" --dev-carrier "$FULL/dev_full.pkl" \
  --simple-results "$RESULTS/p17_controls_full_fit_dev.json" \
  --output "$RESULTS/p17_mear_full_seed0.json" \
  --decisions-output "$RESULTS/p17_mear_full_seed0_decisions.pkl" \
  --simple-decisions "$RESULTS/p17_controls_full_decisions.pkl" \
  --checkpoint "$RESULTS/p17_mear_full_seed0.pt" \
  --budget 50 --depth 3 --epochs 5 --batch-size 4096 --learning-rate 3e-4 --seed 0 --device cuda

mear_pass=$(
  "$PY" -c "import json; print(json.load(open('$RESULTS/p17_mear_full_seed0.json'))['gate3']['pass'])"
)
if [[ "$mear_pass" != "True" ]]; then
  echo "STOP: Gate 3 seed0 failed; no DARS/visual evidence/extra seeds"
  exit 0
fi

for seed in 1 2; do
  echo "MEAR seed$seed on full fit/dev"
  CUDA_VISIBLE_DEVICES=6 "$PY" "$ROOT/models/relation_decision_regret_v2/run_p17_mear.py" \
    --fit-carrier "$FULL/fit_full.pkl" --dev-carrier "$FULL/dev_full.pkl" \
    --simple-results "$RESULTS/p17_controls_full_fit_dev.json" \
    --simple-decisions "$RESULTS/p17_controls_full_decisions.pkl" \
    --output "$RESULTS/p17_mear_full_seed${seed}.json" \
    --decisions-output "$RESULTS/p17_mear_full_seed${seed}_decisions.pkl" \
    --checkpoint "$RESULTS/p17_mear_full_seed${seed}.pt" \
    --budget 50 --depth 3 --epochs 5 --batch-size 4096 --learning-rate 3e-4 --seed "$seed" --device cuda
done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/data2/liuhaoran/project/cv/psg/results/relation_decision_regret_v2')
rows = [json.loads((root / f'p17_mear_full_seed{seed}.json').read_text()) for seed in (0, 1, 2)]
deltas = [float(row['gate3']['gain_vs_simple_mR_pp']) for row in rows]
lower = min(float(row['bootstrap_vs_simple']['ci95_pp'][0]) for row in rows)
mean_delta = sum(deltas) / len(deltas)
print({'seed_deltas_pp': deltas, 'mean_delta_pp': mean_delta, 'min_bootstrap_lower_pp': lower})
if not (mean_delta >= 1.5 and all(delta > 0 for delta in deltas) and lower > 0):
    print('STOP: Gate 6 failed; no DARS/visual evidence/second carrier')
    raise SystemExit(0)
print('PROMOTE: Gate 6 passed; inspect DARS trigger before any new module')
PY
