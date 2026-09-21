# Selective Predicate Surgery v3 — Catastrophic-Harm Audit

日期：2026-09-21（Asia/Shanghai）  
协议：`experiments/selective_predicate_surgery_v3/HARM_AUDIT_LOCK.json`  
实现：`models/selective_predicate_surgery_v3/catastrophic_harm_audit.py`  
Carrier：`results/relation_decision_regret_v1/p14_full_carrier/fit_full.pkl`

## Launch / resource checks

- Fit carrier and lock file existed before launch.
- CPU-only execution; no GPU was requested or used.
- Existing p1.9 carrier and source data were not modified.
- Regression tests: `PYTHONPATH=. pytest -q models/selective_predicate_surgery_v2/tests/test_ccuv.py models/relation_decision_regret_v2/tests/test_surgery.py` → **13 passed**.
- Command:

```bash
PYTHONPATH=. python -u models/selective_predicate_surgery_v3/catastrophic_harm_audit.py \
  --fit-carrier results/relation_decision_regret_v1/p14_full_carrier/fit_full.pkl \
  --output results/selective_predicate_surgery_v3/harm_audit_full_fit.json
```

The full fit-only audit completed after approximately 68 minutes of CPU time. It
did not load the dev carrier and did not run G5/G6 because the P0 audit gate was
evaluated first.

## P0 audit result

| population | learned ΔmR | oracle harm-veto ΔmR | harm-veto capture | positive-only proposal ceiling |
|---|---:|---:|---:|---:|
| select | +0.53304 pp | +3.64206 pp | **14.78%** | 69.53% |
| calib | −0.15888 pp | +3.79614 pp | **13.71%** | 65.82% |

The oracle harm veto removes all negative utility from the selected proposal
prefix, but it still captures less than the preregistered 20% of full oracle
gain on both select and calib. The positive-only ceiling remains high, which
indicates that the compact proposal representation contains positive actions,
but the current ranking cannot expose enough of them under the locked coverage
curve without an oracle.

## Decision

**KILL v3 compact-feature gate family at P0.** The audit gate requires both
select and calib harm-veto capture ≥20% and ΔmR>0; only the ΔmR clauses pass.
Do not run G5 analytic asymmetric reweighting, G6 factorized heads, one-shot
dev, visual evidence, second carrier, confirm, or official test from this line.

Machine-readable output:
`results/selective_predicate_surgery_v3/harm_audit_full_fit.json`

