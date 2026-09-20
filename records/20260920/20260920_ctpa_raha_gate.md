# CTPA / RAHA fit500→dev500 gate

## Registration

This record continues `P14_FULL_DEV_GATE.json` using only the two existing
compact p1.4 carrier artifacts:

- fit: 500 grouped images, Top-50 physical-pair support;
- dev: 500 grouped images, Top-50 physical-pair support;
- 56 predicate classes, float32 predicate scores, float16 384-D pair features;
- no confirm carrier generated or read.

The runner is `models/relation_decision_regret_v1/ctpa.py`.  CTPA changes only
the predicate emitted for each already-supported pair.  Pair support, pair
ranking, masks, and SingleMPO mapping are fixed.  `gt_pair` is used to create
fit labels only and is not read during decode.  The Hidden-MLP is a
capacity-matched killer control.  RAHA is represented by the registered
inverse-frequency weighted pairwise objective.

## Results

| depth | K | method | mR | R | ΔmR vs baseline |
|---:|---:|---|---:|---:|---:|
| 2 | 20 | baseline | 46.241% | 42.278% | 0 |
| 2 | 20 | CTPA | 34.182% | 45.379% | -12.060 pp |
| 2 | 20 | Hidden-MLP | 36.329% | 46.924% | -9.912 pp |
| 3 | 50 | baseline | 53.544% | 48.294% | 0 |
| 3 | 50 | CTPA | 38.460% | 53.504% | -15.084 pp |
| 3 | 50 | Hidden-MLP | 37.997% | 55.090% | -15.547 pp |
| 2 | 50 | baseline | 53.544% | 48.294% | 0 |
| 2 | 50 | CTPA | 39.006% | 52.597% | -14.539 pp |
| 5 | 20 | baseline | 46.241% | 42.278% | 0 |
| 5 | 20 | CTPA | 30.890% | 48.193% | -15.351 pp |
| 5 | 50 | baseline | 53.544% | 48.294% | 0 |
| 5 | 50 | CTPA | 35.013% | 55.603% | -18.531 pp |
| 3 | 50 | RAHA-CTPA | 43.003% | 52.523% | -10.542 pp |

Strict hidden-feature recheck (the compact carrier built from the registered
`hidden_fit500_predictions.pkl` and `hidden_dev500_predictions.pkl`) gave:

| depth | K | method | mR | R | ΔmR vs baseline |
|---:|---:|---|---:|---:|---:|
| 3 | 50 | CTPA + 384-D pair features | 34.917% | 53.264% | -18.628 pp |
| 3 | 50 | Hidden-MLP + 384-D pair features | 38.781% | 54.768% | -14.763 pp |

This closes the possible objection that the earlier compact artifacts had been
score-only.  The strict hidden-feature run still fails the corrected mR gate.

The compact result JSON files are under
`results/relation_decision_regret_v1/ctpa_*.json`.

## Gate decision

The CTPA line is **stopped as a promoted method**.  It consistently improves
ordinary R while substantially reducing mR.  Increasing hypothesis depth from
2 to 5 worsens mR, so fixed-depth arbitration is not the explanation.  RAHA
recovers about 4.54 pp mR relative to uniform CTPA but remains 10.54 pp below
the frozen baseline.  CTPA beats Hidden-MLP once at L=3/K=50, but both fail the
primary mR gate; this is not a method win.

Interpretation: the frozen carrier contains recoverable predicate ambiguity,
but both score-only and frozen-pair-feature arbitration are biased toward
frequent predicates and cannot serve corrected mR.  This satisfies the
pre-registered trigger for the next candidate: candidate-specific visual
evidence re-acquisition (EVA), rather than another score-only reranker.  The
EVA preflight is recorded in `EVA_PREFLIGHT_HIDDEN.json` and is blocked because
the compact carrier has no visual map/token fields.  A new registered visual
carrier export would be required before EVA; no confirm run is authorized.

## Execution note

Four initial matrix jobs were stopped after 12 minutes because they redundantly
trained the expensive Hidden-MLP in parallel and caused severe CPU contention.
The final CTPA/RAHA matrix was rerun with the Hidden-MLP skipped where an
existing same-depth/budget killer result already existed.  This only changes
execution scheduling, not data, labels, support, or metrics.
