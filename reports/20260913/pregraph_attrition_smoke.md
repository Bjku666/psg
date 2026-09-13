# Pre-graph attrition implementation smoke

The new `pregraph_attrition_v1` tools were validated against one existing P0C
full-artifact test image and one direct Mask2Former one-pass forward.  This is
an implementation smoke only; the official test split is not used for model or
method selection.

Checks passed:

- raw, semantic-eligibility, full-pool competition, and native-admission
  mappings completed with fixed native `K`;
- endpoint lifecycle records used global decoder query IDs;
- near-miss records included area deficit, competition-candidate status, and
  lowest-margin rescue cost;
- the GT minimum-change oracle preserved native support and zero changed pixels
  at `epsilon=0`;
- 22 repository tests passed, including the new lifecycle, near-miss, and
  minimum-change tests.

The fit-500 evidence run is intentionally not started: this checkout contains
no `train2017` image files and the compact P1 artifacts omit raw masks.  Once
the fit images or a P0C full one-pass source are available, run
`run_stage_decomposition.py` with `--images ... --model ...` on the registered
fit partition, then run `near_miss_audit.py` and `min_edit_oracle.py` on the
same frozen population.
