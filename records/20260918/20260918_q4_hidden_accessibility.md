# Q4 hidden-accessibility implementation and smoke

## Protocol changes

- Added `stratified_split.py` and upgraded the relation-decision manifest to a
  physical-file grouped 70/15/15 fit/dev/confirm split.
- The generated v2 manifest has 32,590/6,990/6,983 image IDs; all three
  partitions contain all 56 predicates and have zero image overlap.
- Updated the benchmark lock: the official test diagnostics are historical
  hypothesis-generating evidence; no further test-driven design is allowed,
  and learned-method development is fit/dev only.

## E1 smoke artifact

The Fair PSG inference path now has an opt-in `--export-pair-features` flag.
It exports the final 384-dimensional relation token aligned with each pair,
without changing the historical three-output inference contract.  The strict
adapter carries this field through mapped candidates.

Artifact:

```text
results/relation_decision_regret_v1/e1_hidden_fit64/accessibility_audit.json
```

This is a 48/16 grouped fit-only smoke audit (the subset is not a full 56-class
metric gate).  F0 is the existing six score features; F1 appends the 384-D
relation token; F2 additionally appends endpoint geometry/classes.

| features | dims | model | utility AUC | R@20 | R@50 |
|---|---:|---|---:|---:|---:|
| F0 | 6 | ridge | 0.9602 | 0.3426 | 0.5119 |
| F0 | 6 | histogram | 0.9439 | 0.3159 | 0.4614 |
| F1 | 390 | ridge | 0.8467 | 0.2770 | 0.3738 |
| F1 | 390 | histogram | 0.9559 | 0.3390 | 0.5093 |
| F2 | 400 | ridge | 0.8534 | 0.2874 | 0.3775 |
| F2 | 400 | histogram | 0.9414 | 0.3468 | 0.4426 |

Interpretation: this bounded smoke does not promote hidden-feature DARR.  The
raw relation token is not automatically decision-accessible under the current
generic probes, and AUC does not translate into a reliable fixed-budget gain.
The next authorized run is the same audit on the complete v2 fit/dev teacher;
no confirm/test result is claimed.

## Verification

```text
python -m pytest -q models/relation_decision_regret_v1/tests \
  models/gssr_p1_v1/tests/test_split_train_dev.py \
  models/relation_failure_decomp_v1/tests
9 passed
```
