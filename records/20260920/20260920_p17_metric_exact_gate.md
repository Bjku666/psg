# P1.7 metric-exact surgery gate

Implemented the protocol-validity repair requested after P1.6:

- population-exact mR/R action contribution using frozen per-predicate image
  denominators;
- independent single-action evaluator parity and distinct-row 2/3/5/10-edit
  additivity audit;
- actual applied-swap accounting with rescue/damage transitions;
- fit-only S0--S7 action controls;
- MEAR candidate-relative low-rank ranker with full 384-D pair token, full
  predicate logits, explicit native/candidate identities, metric-weighted
  ranking, sign loss, and a fixed zero KEEP anchor.

Qualification results on the registered hidden fit500/dev500 carrier:

| gate | result |
|---|---|
| Gate 0 single-action max residual | `1.03e-16` mR, `6.69e-17` R |
| Gate 0 distinct-row max residual | `1.09e-16` mR, `7.62e-17` R |
| exact oracle intervention rate | `664 / 20136 = 3.298%` |
| exact oracle headroom | `+23.662 mR pp`, `+22.840 R pp` |
| strongest simple control | S0 margin, `+0.636 mR pp` |
| simple/oracle captured fraction | `0.0269` |
| Gate 2 decision | full MEAR authorized |

The first controls attempt is retained as `p17_controls_fit500_dev500.json`
for audit but superseded by the corrected `*_v2.json`; its S1/S3 threshold
sign and learned-score KEEP calibration were implementation bugs and are not
used for any claim.

The full P1.4 fit/dev export is running on GPUs 6/7 with the frozen epoch-28
DSFormer carrier. Confirm and official test remain untouched.

An end-to-end one-epoch CPU MEAR smoke on the existing 500/500 qualification
carrier completed successfully. It is explicitly non-formal (`Gate 3` failed
with `-0.644 mR pp` versus S0), and exists only to validate the ranker/decode
path before the full-population run.

## Full fit/dev outcome (2026-09-21)

The registered 66 fit and 14 dev shards completed and merged successfully.
The evaluator excludes relation-free images from the action population, leaving
31,969 fit and 6,868 dev relation-bearing images (the manifest totals remain
32,590 and 6,990 respectively). The merged carrier checksums are recorded in
`results/relation_decision_regret_v1/p14_full_carrier/SHA256SUMS`.

Full-dev validity passed:

- single-action maximum residual: `1.284e-16` mR, `6.206e-17` R;
- distinct-row maximum residual: `1.280e-16` mR, `6.677e-17` R;
- population-exact oracle headroom: `+22.353 mR pp`, `+24.017 R pp`;
- actual intervention rate: `9,292 / 254,538 = 3.651%`.

The full fit-only S0--S7 control gate authorized MEAR because no simple control
captured meaningful oracle headroom. MEAR seed 0 then failed Gate 3:

- MEAR mR: `57.642` versus native/S0 `59.108`;
- gain versus strongest simple: `-1.466 mR pp`;
- R change versus native: `+2.156 pp`;
- grouped bootstrap versus S0: `[-3.928, +0.777]` mR pp;
- correct→wrong transitions: `3,332`.

The preregistered stop condition is therefore active: MEAR is killed for this
line, and seed 1/2, DARS, EoD-VR, DSGG, confirm, and official test are not run.
