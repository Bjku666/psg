# Relation-resolved tokenization v1

This line follows the registered `p3.3` closure.  It changes one representation
factor at a time on the pinned Fair-PSG DSFormer carrier and keeps the corrected
SingleMPO evaluator, PSG split, seed, optimizer, and 40-epoch schedule fixed.

The first authorized controls are:

* `C0_feature_all`: Faster-RCNN FPN stages 0--3 resized/concatenated before the
  existing 1x1 squasher (`feature_key=all`).
* `C1_patch4`: the original stage-0 feature with uniform 4x4 patch embedding.

No proposed R²Former/InterfaceFormer network is authorized by this directory.
Promotion requires both controls to finish and the mechanism/oracle gates in the
user-registered protocol to be evaluated.

The first official-memory launch OOMed: FPN concatenation (`C0`) and the 4x
token count (`C1`) exceed the available memory at `rels_per_batch=128`.  The
runner therefore exposes `REL_MICROBATCH` (default 32) for a feasibility pilot;
pilot metrics must not be used as the final simple-control gate.

An A100 acceleration path is available in `run_accelerated.sh`. It uses
bfloat16 autocast only when `FAIR_PSG_AMP=1`, with larger relation micro-batches
to reduce repeated backbone forwards. Its artifacts are suffixed `_amp` and
are reported separately from the original pilot.

The accelerated runner also sets `FAIR_PSG_FUSED=1`: the complete image batch
is encoded once and relation chunks are handled inside DaniFormer. This is an
execution optimization; the non-fused path remains the default.
