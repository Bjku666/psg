# GSSR P0B formal supply/admission protocol

P0B preserves `gssr_p0_v1` as immutable first diagnostic evidence. It adds no
learner and makes no downstream PSG-accuracy claim.

The official evaluation population remains every relation-containing OpenPSG
test scene row. Sixteen pairs of rows share one COCO filename and identical
entity/panoptic annotations but carry distinct relation annotations. Point
estimates therefore retain all 2,177 official rows; uncertainty resamples the
2,161 physical-image `file_name` clusters so the duplicated images are not
treated as statistically independent.

## Exact objectives

`gt_set_oracle_micro` assigns unit weight to every non-self GT relation.
`gt_set_oracle_balanced` assigns relation instance `(s,r,o)` weight `1/N_r`,
where `N_r` is the predicate frequency over the fixed evaluation split. The
latter is the exact oracle for the primary predicate-balanced endpoint-support
metric. Both solve a binary MILP and then deterministically lift GT nodes to
exactly K candidate nodes.

## Matching and score contract

Formal evidence requires class-compatible mask IoU strictly greater than 0.5.
Box IoU is retained only for regression against P0. A mask-quality arm is
invalid unless the exporter explicitly supplies `mask_quality`; no score
fallback is permitted. Raw-query score definitions are frozen as:

- `class_score`: maximum foreground softmax probability;
- `mask_quality`: mean sigmoid mask probability over pixels above 0.5;
- `joint_score`: `class_score * mask_quality`.

## Supply decomposition

Ratios 0.25, 0.50, 0.75, and 1.00 use `K=max(1,ceil(ratio*M))`. The 100% arm is
the full candidate-pool ceiling `F(C)`. At each smaller budget the report must
verify:

`F(C)-F(score_K) = [F(C)-F(oracle_K)] + [F(oracle_K)-F(score_K)]`.

These terms are total budget loss, unavoidable capacity loss, and wrong-
composition loss. Entity criticality is emitted at 50% with degree, incident
predicate frequency, thing/stuff, area, best candidate score/IoU, and rescue
status.

## Gates

The formal balanced mask-IoU gap at 50% must be at least 0.04. P1 remains
forbidden until raw-query native admission and compute-equivalent node-vs-pair
audits also pass. DSFormer checkpoint availability does not block P0B.

