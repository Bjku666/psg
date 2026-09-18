# Literature lock: relation decision regret (v1)

This line is a qualification experiment, not a claim that the surrounding
PSG areas are empty.  The closest-work scan is frozen before any learner is
trained.  The claim under test is narrower:

> Given frozen object identities, a frozen legal candidate support, a fixed
> output budget, and corrected SingleMPO evaluation, can candidate-level
> marginal slot utility be recovered beyond ordinary score calibration?

| Cluster | Already solves | This line's exact delta |
| --- | --- | --- |
| Fair PSG / DSFormer | corrected evaluation and dense relation carrier | legal fixed-K candidate-set decision regret |
| Pair-Net / SAPPM | pair proposal / pair ranking | triplet utility after pair support is frozen |
| predicate/triplet learning, HRT | predicate or triplet representations | membership in a unique-pair fixed-budget set |
| conformal SGG | uncertainty/prediction sets | deterministic marginal utility under the evaluator contract |
| CMAT | graph-level counterfactual credit | exact candidate insertion/removal utility, without policy gradients |
| PSGTR/PSGFormer | set-style graph prediction | set prediction over existing physical object identities |

This is a record of the search performed for this repository, not an
absolute non-existence claim.  The literature gate kills the line only if a
paper is found with the same corrected SingleMPO/fixed-K support and the same
candidate-level marginal slot utility (or identity-preserving set decoder).
