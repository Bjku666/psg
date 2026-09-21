# 2026-09-21 Cleanup and Experiment Termination

## Cleanup scope

The project-local large-file inventory was checked under `/data1` and
`/data2/liuhaoran/project/cv/psg`. No matching project directory was found
under `/data1`. The following completed-run decision caches were first moved
to the recoverable quarantine directory `.cleanup_quarantine/20260921/`,
verified, and then deleted as explicitly requested:

| File | Bytes | Reason |
|---|---:|---|
| `results/relation_decision_regret_v2/p17_mear_full_seed0_decisions.pkl` | 333,508,268 | Intermediate cache; corresponding JSON result and log are retained |
| `results/relation_decision_regret_v2/p17_controls_full_decisions.pkl` | 311,143,482 | Intermediate cache; corresponding JSON result and log are retained |

The v3 audit carrier `fit_full.pkl`, the historical `dev_full.pkl` carrier,
and the registered `fit500/utility_teacher.json` diagnostic were retained.

## Experiment termination

The locked v3 catastrophic-harm audit completed with `audit_killed`:

- select harm-veto capture: 14.78% (required >=20%)
- calib harm-veto capture: 13.71% (required >=20%)

The P0 gate therefore failed. G5, G6, dev, GPU, second-carrier, visual,
confirm, and official-test stages remain prohibited by
`experiments/selective_predicate_surgery_v3/HARM_AUDIT_LOCK.json`.

## Verification

- v3 machine-readable audit result remains present.
- Both protected full carriers remain present.
- JSON result artifacts remain parseable.
- Existing regression suite rerun after cleanup: `13 passed`.
