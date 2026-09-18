# Relation decision regret v1 implementation

## Scope

The pasted strategy was translated into a qualification-only repository line:

- benchmark/carrier/literature/oracle/simple-control locks;
- strict SingleMPO mapped-candidate adapter;
- predicate evidence-depth audit (`M=1,2,3,5`);
- legal GT-aware fixed-K oracle (`K=20,50`, including `all` depth);
- D0--D7 controls over the identical frozen support;
- physical-file grouped bootstrap for oracle deltas;
- contract tests.

Historical `relation_failure_decomp_v1` files were not modified. CMUD and URSD
were intentionally not created: their gate is `DIAGNOSTIC_PLAN.json` after E0--E4.

## Verification

```text
python -m pytest -q models/relation_decision_regret_v1/tests \
  models/relation_failure_decomp_v1/tests \
  models/endpoint_survival_v2/tests
14 passed
```

No external download was needed. The pinned Fair PSG evaluator already exists
under `third_party/fair_psg/`. If later work needs papers/checkpoints, the
experiment README requires repository-local downloads with URL, SHA256, and
license records.

## Real-artifact smoke

Using the existing (external, already registered) DSFormer pickle and OpenPSG
panoptic GT with `--max-images 2`:

- predicate-rank audit: 9 mapped relations; rank ≤1/2/3/5 = 66.67/77.78/88.89/100%;
- legal oracle: R@20 baseline 0.3833, top-1/2/3/5/all 0.4250/0.4667/0.5083/0.5500/0.5500;
- D0--D7 controls completed without violating unique directed pairs.
- exact insertion/removal utility helpers are available for a future CMUD
  teacher, but no learner is enabled.

## Follow-up fit teacher smoke and storage action

The bounded fit-only teacher export was completed after correcting the local
subset contract in `export_utility_teacher.py`.  The 64-image inference
subset is explicitly marked `subset_contract.partition=fit`; its temporary
`test_image_ids` therefore do not bypass the official-test guard for ordinary
annotations.  The artifact is:

```text
results/relation_decision_regret_v1/smoke_fit64/utility_teacher.json
```

It contains 128 image-budget rows (64 images at K=20 and K=50), no missing
predictions, and non-zero exact insertion/removal labels.  The relevant
tests pass: `15 passed` across the relation-decision, decomposition, and
endpoint-survival contract suites.

The following already-extracted installer archives were recorded and removed
to recover space.  Their extracted directories remain in place; no historical
result, checkpoint, prediction, annotation, or repository artifact was
removed.

| Path | Bytes | SHA256 | Reason |
|---|---:|---|---|
| `/data2/liuhaoran/psg_data/openpsg/coco/train2017.zip` | 19336861798 | `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929` | COCO train2017 is already extracted under the same registered data root |
| `/data2/liuhaoran/psg_data/openpsg/panoptic_gt/panoptic_annotations_trainval2017.zip` | 860725834 | `c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370` | panoptic GT is already extracted and used by the recorded runs |

## Utility learnability audit

Using only the 64-image fit smoke teacher and a deterministic 48/16 grouped
split, the CPU-only ridge probe produced this held-out diagnostic:

| Budget | Baseline R | Ridge-probe R | Legal-oracle R | Probe AUC |
|---:|---:|---:|---:|---:|
| 20 | 0.34204 | 0.36644 (+2.44 pp) | 0.98958 | 0.9491 |
| 50 | 0.50821 | 0.48952 (-1.87 pp) | 0.98958 | 0.9502 |

The output is
`results/relation_decision_regret_v1/smoke_fit64/utility_learnability.json`.
This confirms the exact labels are not vacuous, but the budget-dependent
regression means CMUD is not promoted yet. The next run is a larger fit/dev
teacher with a locked holdout; any GPU launch must use only devices 4--7.

## fit500 teacher and nonlinear probe

The larger fit-only run used 500 grouped-fit images and the existing DSFormer
checkpoint. Inference ran on `CUDA_VISIBLE_DEVICES=4`; devices 0--3 were not
used. The prediction artifact has 500 unique image ids and no missing rows.
The exact teacher contains 1000 image-budget rows and is stored at
`results/relation_decision_regret_v1/fit500/utility_teacher.json`.

The locked 375/125 image split was evaluated with a CPU-only
`HistGradientBoostingRegressor` probe over frozen carrier score features:

| Budget | Baseline R | Ridge probe R | Tree probe R | Legal-oracle R | Tree AUC |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.35440 | 0.34740 | 0.36829 (+1.39 pp) | 0.97175 | 0.9118 |
| 50 | 0.44730 | 0.44673 | 0.45244 (+0.51 pp) | 0.97279 | 0.9060 |

The full diagnostic is
`results/relation_decision_regret_v1/fit500/utility_learnability_v3.json`.
The result authorizes a feature-based CMUD prototype on fit/dev only, but not
any confirm/test evaluation yet. The probe uses only frozen score features;
it is not the final method and no claim is made from its smoke mR (the subset
does not cover all predicate classes).

## Independent fit-to-dev transfer

A separate 500-image dev partition was inferred on `CUDA_VISIBLE_DEVICES=5`
and labelled offline.  The probe was trained only on the 500-image fit
teacher; dev labels were used only for the locked evaluation.  The artifacts
are `results/relation_decision_regret_v1/dev500/utility_teacher.json` and
`utility_transfer_v2.json`.

| Budget | Dev baseline R | Fit-tree probe R | Fit-positive classifier R | Legal-oracle R | Positive AUC |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.422778 | 0.422848 | 0.423564 (+0.079 pp) | 0.972737 | 0.9225 |
| 50 | 0.482940 | 0.484730 | 0.487821 (+0.488 pp) | 0.972998 | 0.9224 |

This supports a fit-trained binary “worth a slot” target as the next CMUD
prototype.  The gain remains small relative to the legal headroom, so no
confirm/test result is claimed and no GPU training has been launched.

These are smoke-only numbers (the subset does not contain all 56 predicates,
so its mR fields are intentionally not used for a gate). Full-population E0--E4
must be run before any learner is considered.
