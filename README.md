# Before-the-Graph PSG

This repository now studies **relation-endpoint survival during panoptic
formation**. The active qualification question is:

> Under frozen panoptic evidence, can graph-critical endpoint hypotheses be
> preserved through semantic filtering and pixel competition without a
> material loss in official panoptic quality?

The historical first diagnostic is `gssr_p0_v1`; qualification work is in
`gssr_p0b_v1` (formal matching/contracts) and `gssr_p0c_v1` (raw-query entity
admission). P0 is frozen. The original `gssr_p1_v1` restricted replay is
invalidated before evidence because it intervened before pixel competition.
The corrected `gssr_p1_v2` line is now closed as a post-competition
actionability failure: its dev fixed-competition oracle gained only 0.00999.
The `pregraph_attrition_v1` fit-500 qualification is complete. It locates
material losses at semantic eligibility and pixel competition, but its
per-image 0.5% minimum-change oracle gains only 0.01192 balanced endpoint
support. The small-edit gate failed, so this method line is closed without a
learner. `endpoint_survival_v1` is retained as smoke-only code. The active
`endpoint_survival_v2` line first corrects evaluator contracts and then runs a
fit-500 semantic/competition margin qualification. No learner is authorized
unless balanced endpoint support improves by at least 3 pp while official PQ
drops by no more than 1 pp.

The corrected v2 oracle passed this threshold on fit500 but did not reproduce
it on the independent grouped dev population: frozen semantic and competition
interventions gained only 1.82 pp and 1.67 pp balanced support, respectively.
The v2 method line and its conditional P1 learner are therefore closed; no
joint grid, confirm run, or relation-carrier claim is authorized.

The active line is now `relation_failure_decomp_v1`. It trains the pinned Fair
PSG DSFormer dense relation carrier and then attributes every GT relation to
endpoint, pair-retention, predicate, final-ranking, or success under the
corrected SingleMPO mask-identity contract. No pair/predicate/ranking learner
is authorized until the registered conditional-oracle gate is evaluated.

## Layout

- `models/gssr_p0_v1/`: evaluator, exact fixed-K oracle, tests, and launchers.
- `models/gssr_p0b_v1/`: balanced oracle, mask matcher, identity audit, supply
  decomposition, criticality export, and formal runner.
- `models/gssr_p0c_v1/`: raw-query exporter/schema, native admission audit, and
  compute-equivalent node-vs-pair oracle.
- `models/gssr_p1_v1/`: historical restricted-query replay adapter, metrics,
  and grouped split utilities; its pre-competition replay is not evidence.
- `models/gssr_p1_v2/`: fixed winning-mask entity-admission replay primitives.
- `models/pregraph_attrition_v1/`: raw-to-admission endpoint lifecycle,
  near-miss margin audit, and GT minimum-change rescue oracle.
- `models/endpoint_survival_v2/`: corrected population metrics, official COCO
  category-wise PQ, E0 controls, and structured-margin qualification.
- `models/relation_failure_decomp_v1/`: seeded carrier wrapper and four-stage
  identity-correct decomposition.
- `models/relation_decision_regret_v1/`: metric-locked predicate-rank audit,
  legal fixed-K decision oracle, and D0-D7 controls. This is qualification
  only; CMUD/URSD are gated and not yet implemented.
- `experiments/gssr_p0_v1/`: frozen hypothesis, run matrix, gates, and costs.
- `experiments/relation_failure_decomp_v1/`: frozen R0 plan and launch chain.
- `results/gssr_p0_v1/`: immutable run outputs and status metadata.
- `records/YYYYMMDD/`: commands, deviations, and decisions.
- `reports/YYYYMMDD/`: human-readable result summaries.
- `third_party/fair_psg/`: pinned official Fair PSG / DSFormer source.

Resources downloaded specifically for a new experiment must be stored inside
the repository (see `experiments/relation_decision_regret_v1/README.md`) and
recorded with URL, SHA256, and license. Existing dataset/checkpoint paths are
exceptions only when already registered in a run contract.

Large data and weights remain outside the source tree.  The default data root is
`/data2/liuhaoran/psg_data/openpsg` because the pre-existing
`/data2/liuhaoran/datasets` symlink is broken.

## Current entry points

```bash
cd /data2/liuhaoran/project/cv/psg
python -m pytest -q models/gssr_p0_v1/tests

python models/gssr_p0_v1/run_budget_audit.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --candidates /path/to/inferred_masks/val2017/anno.json \
  --split test --budgets 0.25 0.50 0.75 \
  --output results/gssr_p0_v1/<new-run-id>
```

See `experiments/gssr_p0_v1/protocol.md` before launching evidence runs.

P0B/P0C entry points:

```bash
python models/gssr_p0b_v1/audit_identity_contract.py --help
python models/gssr_p0b_v1/prepare_panoptic_gt.py --help
python models/gssr_p0b_v1/run_formal_audit.py --help
python models/gssr_p0c_v1/export_raw_mask_queries.py --help
python models/gssr_p0c_v1/raw_query_audit.py --help
python models/gssr_p0c_v1/node_pair_budget_oracle.py --help

python models/gssr_p1_v2/run_actionability_oracle.py --help
python models/pregraph_attrition_v1/run_stage_decomposition.py --help
python models/pregraph_attrition_v1/near_miss_audit.py --help
python models/pregraph_attrition_v1/min_edit_oracle.py --help
python models/pregraph_attrition_v1/analyze_min_edit.py --help

python -m pytest -q models/gssr_p1_v1/tests models/gssr_p1_v2/tests
```

P1 v2 evidence is closed and tracked by `results/gssr_p1_v2/RUN_STATUS.json`.
The completed pre-graph decision and artifact paths are in
`experiments/pregraph_attrition_v1/PREGRAPH_PLAN.json`; no pre-graph learner is
authorized.

Use `/data2/liuhaoran/venvs/fair_psg_p0/bin/python` for the raw-query exporter;
the base environment has an incompatible Transformers/Accelerate combination.
