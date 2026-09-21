#!/usr/bin/env bash
set -euo pipefail

cd /data2/liuhaoran/project/cv/psg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

python models/selective_predicate_surgery_v2/run_ccuv_controls.py \
  --fit-carrier results/relation_decision_regret_v1/p14_full_carrier/fit_full.pkl \
  --dev-carrier results/relation_decision_regret_v1/p14_full_carrier/dev_full.pkl \
  --output results/selective_predicate_surgery_v2/ccuv_controls_full_fit.json \
  --seed 0
