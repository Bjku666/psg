# Pre-graph attrition fit-500 execution record

The frozen fit population contains 500 rows and 499 physical-file groups. Eight
parallel A100 shards exported all 499 physical images to the external P0C full
artifact directory; the merged artifacts occupy 389 MB.

During analysis, two contract issues were corrected before the final decision:

1. Near-miss and min-edit original area now match Mask2Former's official
   class-weighted mask overlap test.
2. Stage loss `estimate` now reports the full-sample point difference; the
   bootstrap draw mean is retained separately as `bootstrap_mean`.

The strict relation/non-relation control and the minimum-change paired
bootstrap both use physical filename as the resampling unit. Final commands are
recorded in `experiments/pregraph_attrition_v1/protocol.md`. No learner was
launched after the small-edit gate failed, and no official test data influenced
the decision.
