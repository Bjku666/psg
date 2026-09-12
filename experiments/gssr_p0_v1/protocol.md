# GSSR P0 fixed-budget qualification protocol

Registered 2026-09-12 before any PSG candidate-pool result.

## Claim under test

At a fixed entity budget, entity-set composition removes a substantial fraction
of GT relations before pair proposal.  This is distinct from segmentation supply
and from selecting subject-object pairs among already admitted entities.

The primary P0 contrast is `gt_set_oracle - score_topk` on predicate-balanced
endpoint support (`mR_inf_K`).  This is a GT-assisted headroom diagnostic, not a
deployable method and not final PSG accuracy.

## Population and matching

- Dataset: official OpenPSG `psg.json`, official test IDs only.
- Carrier 1: `facebook/mask2former-swin-large-coco-panoptic` final panoptic
  candidates (smoke/diagnostic carrier).
- Carrier 2: OneFormer Swin-L final panoptic candidates after Carrier 1 is valid.
- Formal candidate pool later moves to raw MaskDINO queries; final masks alone
  are not sufficient for the paper's admission-budget claim.
- Matching: class-compatible box IoU > 0.5 for the initial Fair PSG sanity path;
  class-compatible mask IoU > 0.5 is required for formal results.
- Matching semantics reproduce SingleMPO: each prediction chooses its best GT,
  and only the best prediction for a GT is retained.

## Budget freeze

Do not invent absolute K values before seeing candidate counts.  Export count
quantiles first.  Then evaluate retention ratios 25%, 50%, and 75%, with per-image
`K=max(1, ceil(ratio*M))`.  A later compute-aligned absolute-K sweep must be
registered separately.  Every arm receives exactly the same per-image K.

## Experiment matrix

| Run | Single changed factor | Value | Fixed configuration | Expected outcome |
| --- | --- | --- | --- | --- |
| P0-S | selection | score_topk | pool, K, split, matching | strong local-quality baseline |
| P0-Q | selection | mask_quality_topk | same | may match score when quality is unavailable |
| P0-D | selection | diversity_topk | same | removes high-IoU duplicate candidates |
| P0-N | selection | gt_nodewise | same | independent GT degree utility wastes slots on duplicates |
| P0-O | selection | gt_set_oracle | same | exact fixed-K endpoint-support ceiling |

`gt_nodewise` and `gt_set_oracle` are oracle diagnostics, not baselines available
at inference.  A predicted relation-degree baseline is added only with frozen
DSFormer outputs.

## Metrics fixed before execution

Primary mechanism metric: predicate-balanced endpoint support `mR_inf_K`.

Secondary mechanism metrics: micro endpoint support, per-image macro endpoint
support, entity recall, candidate count, selected node count, and ordered pair
count `K(K-1)`.  Every summary includes per-predicate numerators/denominators and
per-image rows.  No downstream `mR@50` may be reported without frozen DSFormer
inference plus corrected `--dedup fail` SingleMPO evaluation.

## Exact oracle definition

Candidate-to-GT eligibility follows the declared matching rule.  The set oracle
solves a binary integer program per image: choose at most K supplied GT entities
and maximize the number of GT relation instances whose two endpoints are chosen.
One representative candidate is then selected for each chosen GT and unused
slots are deterministically padded, preserving exactly K candidates.  MILP
failure is a failed image, never silently replaced by greedy selection.

## Falsification gates

Continue beyond P0 only if both hold on a complete carrier:

1. `gt_set_oracle - score_topk >= 0.04` absolute predicate-balanced endpoint
   support at a predeclared mid budget (50% retention).
2. In the later node-oracle x pair-oracle 2x2 study, node-oracle gain exceeds
   pair-oracle gain or constitutes a clearly material fraction of recoverable
   loss.

After P0, a learner is eligible only if set repair beats both predicted
relation-degree Top-K and a local MLP, and yields at least +1.0 SingleMPO mR@50
on development data or a substantial pair-compute reduction at matched mR.

## Cost estimate

- Annotation-only audits: CPU, under 1 hour, below 1 GB outputs.
- Panoptic inference per carrier: one A100, estimated 1--4 GPU-hours pending
  smoke calibration; model cache roughly 1--3 GB.
- Frozen DSFormer inference: one A100, estimated below 2 GPU-hours after a valid
  checkpoint is identified.
- Full DSFormer retraining is not authorized by this P0 contract.
- API cost: zero.  Formal COCO/PSG assets are expected to require under 20 GB for
  the validation-only path.

## Analysis

Report curves over retention ratio with paired per-image bootstrap 95% confidence
intervals for oracle-minus-baseline.  Plot endpoint support against ordered pair
count.  Failed/missing images count as zero only when the carrier itself omitted
them; evaluator errors stop the run.  Seed is 0 for bootstrap and all tie breaks.

