# Conditional endpoint-survival learner protocol

This protocol was preregistered before reading the independent v2 dev result.
It did not authorize training. Each stage was to be authorized separately only if
its frozen v2 margin reproduces at least +3 pp balanced endpoint support with
official PQ delta no worse than -1 pp on dev.

The teacher performs a real singleton counterfactual through the frozen native
pipeline. It stores graph benefit and visual disruption as separate targets;
the endpoint-critical binary indicator is neither the target nor an input.
This prevents repeating the earlier error of asking a learner to imitate a
large but unactionable oracle.

L0 tests whether ordinary uncertainty is sufficient. L1 adds only local query
and mask features. L2 adds predicted pair/relation context. A relational claim
requires L2 to improve over L1 under the same intervention budget and margin.
L3 is conditional on both stages and the L2 comparison; it is not a default
two-dimensional margin search.

All first-stage training and cross-validation remain inside the frozen
fit500 physical-file groups. Dev is evaluated once after choices are frozen;
confirm stays locked until the complete learner is selected. Official PSG
R@K/mR@K are deferred until a frozen relation carrier is integrated.

The independent dev result subsequently failed for both frozen stage points.
This learner line is therefore closed without training; confirm and the frozen
relation carrier remain locked.
