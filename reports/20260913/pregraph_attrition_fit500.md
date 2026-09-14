# Pre-graph attrition fit-500 qualification

## Decision

`pregraph_attrition_v1` is closed as a minimum-change actionability failure.
Pixel competition is a real attrition stage, but the registered small-edit
qualification does not expose enough recoverable support to authorize EPQC or
another fixed-K rescue learner. The official test split was not used for this
decision.

## Frozen contract

- Population: 500 official-train fit rows, 499 physical filenames, selected by
  SHA-256 with seed 0.
- Carrier: frozen Mask2Former Swin-L raw-query artifacts.
- Metric: predicate-balanced fixed-K endpoint support with native per-image K.
- Matching: class-compatible mask IoU strictly greater than 0.5.
- Inference: 2,000 paired bootstrap replicates over physical filenames.
- Intervention: at most one GT-guided add/drop swap per image; epsilon is a
  per-image changed-pixel cap.

## Stage decomposition

| Stage | Balanced support | Loss from previous stage | 95% CI |
|---|---:|---:|---:|
| Raw query | 0.87605 | - | - |
| Semantic eligibility | 0.80448 | 0.07157 | [0.04706, 0.10094] |
| Pixel competition | 0.72479 | 0.07969 | [0.03035, 0.09618] |
| Native admission | 0.71941 | 0.00538 | [0.00236, 0.00996] |

The registered competition-loss threshold of 0.03 passes. Endpoint lifecycle
counts are 1,369 absent at raw supply, 517 lost at semantic eligibility, 328 at
competition, 45 at admission, and 3,514 survivors.

## Near-miss control

There are 209 relation-critical and 142 non-relation suppressed,
semantic-matched queries. Relation-critical queries have a median retained-area
ratio of 0.03648 and require a median 76.35% of their original area to cross the
native overlap rule. The non-relation controls are easier: 0.13792 retained and
66.24% deficit. The grouped-bootstrap relation-minus-control median deficit is
+10.11 points, 95% CI [+2.91, +30.61]. No relation-critical query is within
0.5% of its original-area threshold.

## Minimum-change curve

| Per-image epsilon | Balanced support | Gain vs native | 95% CI | Improved rows |
|---:|---:|---:|---:|---:|
| 0.10% | 0.72120 | 0.00179 | [0.00000, 0.00395] | 1/500 |
| 0.25% | 0.72391 | 0.00450 | [0.00005, 0.01223] | 4/500 |
| 0.50% | 0.73133 | 0.01192 | [0.00454, 0.02316] | 15/500 |
| 1.00% | 0.73309 | 0.01368 | [0.00626, 0.02521] | 25/500 |
| 2.00% | 0.75851 | 0.03910 | [0.01013, 0.05059] | 39/500 |

The registration used "material" without a numeric minimum. The proposal's
3--5 point example is therefore an interpretation target, not a retroactively
registered threshold. At the registered 0.5% cap, even the CI upper bound is
below 3 points and gains occur in only 3% of rows. Together with the control
result, this is insufficient to justify learner training.

## Artifacts

- `results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3/summary.json`
- `results/pregraph_attrition_v1/near_miss_fit500_seed0_v4.json`
- `results/pregraph_attrition_v1/min_edit_fit500_seed0_v2/analysis.json`

The earlier unweighted-overlap and bootstrap-mean summaries remain on disk for
provenance but are superseded by the versions above.
