# Relation decision regret v1

This is the frozen qualification line for E0--E4: predicate evidence-depth
audit, a legal fixed-budget oracle on the frozen carrier support, D0--D7
score controls, and exact insertion/removal utility labels for a future CMUD
teacher. It does not create or train CMUD/URSD before the gates pass.

`models/relation_failure_decomp_v1/` and its results are immutable historical
evidence.

## Local resource policy

All resources downloaded specifically for this line must live below this
repository: use `third_party/relation_decision_regret_v1/` for papers or source
snapshots and `results/relation_decision_regret_v1/` for generated artifacts.
Do not write downloads into `$HOME`, `/tmp`, or an external cache. Large
OpenPSG data and model weights may remain at already registered absolute paths
documented by the root README; record those paths and checksums in each run.

The tools do not require a network download: the pinned evaluator is already
under `third_party/fair_psg/`. If a paper or stronger carrier checkpoint is
needed later, download it into a repository-local directory and record URL,
SHA256, and license before use.

## Commands

```bash
python models/relation_decision_regret_v1/predicate_rank_audit.py --help
python models/relation_decision_regret_v1/run_legal_decision_oracle.py --help
python models/relation_decision_regret_v1/run_simple_controls.py --help
python models/relation_decision_regret_v1/make_split_manifest.py --help
python models/relation_decision_regret_v1/export_utility_teacher.py --help
python models/relation_decision_regret_v1/utility_learnability_audit.py --help
python models/relation_decision_regret_v1/utility_transfer_audit.py --help
python -m pytest -q models/relation_decision_regret_v1/tests
```

No output from this line authorizes a learner automatically. The gate is
recorded after E0--E4 using `DIAGNOSTIC_PLAN.json`.

## Current CTPA/RAHA status

The p1.4 fit500→dev500 CTPA matrix is recorded in
`records/20260920/20260920_ctpa_raha_gate.md` and
`results/relation_decision_regret_v1/CTPA_GATE.json`. Uniform CTPA, RAHA
pairwise weighting, and the strict frozen-pair-feature recheck all regress
corrected mR despite improving ordinary R. This score-only arbitration line is
stopped as a promoted method; confirm remains locked. EVA is only authorized
after a new registered carrier export containing candidate-specific visual
tokens. The current compact carrier contains logits and optional 384-D pair
features but no visual map/token fields, as shown by
`results/relation_decision_regret_v1/EVA_PREFLIGHT_HIDDEN.json`.

The first bounded fit-only teacher smoke is stored under
`results/relation_decision_regret_v1/smoke_fit64/`. Its CPU-only ranking audit
improved R at K=20 on 16 held-out images but regressed at K=50, so a full CMUD
learner remains gated until a larger fit/dev teacher and locked holdout are
run.

The independent `fit500` → `dev500` transfer audit is stored under
`results/relation_decision_regret_v1/dev500/`. It trains only a score-feature
probe on fit labels; dev labels are evaluation-only. The positive-utility
classifier is the current CMUD prototype candidate, while confirm/test remains
locked.

## P1.3 factorization qualification

`ACTION_SPACE_LOCK.json` freezes the legal `(physical pair, predicate
hypothesis)` action space.  Run the zero-GPU decomposition with:

```bash
python models/relation_decision_regret_v1/run_p13.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --predictions /data2/liuhaoran/psg_data/relation_failure_decomp_v1/dsformer_sgdet_full_pairs.pkl \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --output results/relation_decision_regret_v1/p1_3/full_test.json \
  --bootstrap-replicates 100
```

The learner pilot is deliberately fit/dev-only:

```bash
python models/relation_decision_regret_v1/run_mpu_hpa_pilot.py --help
```

`P13_GATE.json` records the current gate.  The oracle evidence closes the
set/swap utility story and motivates the exploratory MPU/HPA pilot; formal
learner promotion remains blocked on fit-trained L0--L4 controls and a
56-predicate fit/dev carrier.
