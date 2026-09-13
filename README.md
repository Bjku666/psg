# RelSupport-PSG

This repository is the experiment root for **fixed-budget entity admission in
panoptic scene graph generation**.  The first qualification question is:

> Under the same candidate pool and the same node budget K, how much relation
> support is lost before pair proposal because the wrong entities are admitted?

The frozen first diagnostic is `gssr_p0_v1`; qualification work is in
`gssr_p0b_v1` (formal matching/contracts) and `gssr_p0c_v1` (raw-query entity
admission). P0 is frozen. The original `gssr_p1_v1` restricted replay is
invalidated before evidence because it intervened before pixel competition.
The corrected next stage is `gssr_p1_v2`: fixed-competition entity admission,
followed by P1B learnability ladders. The learner remains explicitly
unauthorized until replay gates pass.

## Layout

- `models/gssr_p0_v1/`: evaluator, exact fixed-K oracle, tests, and launchers.
- `models/gssr_p0b_v1/`: balanced oracle, mask matcher, identity audit, supply
  decomposition, criticality export, and formal runner.
- `models/gssr_p0c_v1/`: raw-query exporter/schema, native admission audit, and
  compute-equivalent node-vs-pair oracle.
- `models/gssr_p1_v1/`: historical restricted-query replay adapter, metrics,
  and grouped split utilities; its pre-competition replay is not evidence.
- `models/gssr_p1_v2/`: fixed winning-mask entity-admission replay primitives.
- `experiments/gssr_p0_v1/`: frozen hypothesis, run matrix, gates, and costs.
- `results/gssr_p0_v1/`: immutable run outputs and status metadata.
- `records/YYYYMMDD/`: commands, deviations, and decisions.
- `reports/YYYYMMDD/`: human-readable result summaries.
- `third_party/fair_psg/`: pinned official Fair PSG / DSFormer source.

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

python -m pytest -q models/gssr_p1_v1/tests models/gssr_p1_v2/tests

P1 v2 evidence is registered in `experiments/gssr_p1_v2/` and tracked by
`results/gssr_p1_v2/RUN_STATUS.json`; no learner is authorized yet.
```

Use `/data2/liuhaoran/venvs/fair_psg_p0/bin/python` for the raw-query exporter;
the base environment has an incompatible Transformers/Accelerate combination.
