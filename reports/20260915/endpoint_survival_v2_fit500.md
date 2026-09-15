# Endpoint survival v2: corrected fit-500 qualification

## Outcome

The structured-oracle gate passes after all five p0.7 evaluator-contract
corrections and an additional float32 native-replay correction. This authorizes
an independent dev reproduction, but not yet a learner or a joint margin grid.

| Mode | Margin | Balanced support delta | Official PQ delta | Changed pixels | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| semantic | 0.1 | +0.00090 | +0.00187 | 0.23% | fail support |
| semantic | 0.25 | +0.00246 | +0.00499 | 0.64% | fail support |
| semantic | 0.5 | +0.01086 | +0.00777 | 1.13% | fail support |
| semantic | 1.0 | +0.04172 | +0.01486 | 1.82% | pass |
| semantic | 2.0 | +0.04831 | +0.02216 | 2.90% | pass |
| competition | 0.1 | +0.02954 | +0.00877 | 1.41% | fail support |
| competition | 0.25 | +0.03456 | +0.01561 | 2.55% | pass |
| competition | 0.5 | +0.03411 | +0.01758 | 3.69% | pass |
| competition | 1.0 | +0.00919 | +0.01405 | 4.72% | fail support |
| competition | 2.0 | -0.04357 | -0.00894 | 6.13% | fail support |

The explicit native baseline is 0.71941 balanced endpoint support and 0.57409
official category-averaged PQ. The non-monotonic competition curve matters:
large competition margins eventually destroy both support and PQ, so the
result is a bounded frontier rather than evidence for unrestricted boosting.

## Paired uncertainty and frozen dev choice

The paired-bootstrap unit is physical `file_name`; duplicate annotation rows
remain together. Semantic 1.0 has a balanced-support delta of +0.04172 with
95% CI `[+0.01295,+0.05246]`. Competition 0.25 has +0.03456 with 95% CI
`[+0.00620,+0.04481]`. Their positive-gain physical-group Jaccard is 0.488
(26 semantic-only, 17 competition-only, and 41 shared positive groups).

Before looking at dev, the selected rule was frozen as the smallest passing
margin per stage: semantic 1.0 and competition 0.25. Full relation-bearing dev
contains 4565 rows in 4545 physical-file groups. It will be evaluated only at
native, semantic 1.0, and competition 0.25.

## Interpretation boundary

These are GT-informed structured oracles over endpoint survival. They establish
actionable pre-graph method space, not learned prediction and not PSG
R@K/mR@K. A learner is considered only after the frozen points reproduce on
independent dev.
