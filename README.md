# RelSupport-PSG

This repository is the experiment root for **fixed-budget entity admission in
panoptic scene graph generation**.  The first qualification question is:

> Under the same candidate pool and the same node budget K, how much relation
> support is lost before pair proposal because the wrong entities are admitted?

The active line is `gssr_p0_v1`.  P0 is diagnostic only: no learned repair model
is allowed until the registered node-vs-pair and oracle-headroom gates pass.

## Layout

- `models/gssr_p0_v1/`: evaluator, exact fixed-K oracle, tests, and launchers.
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

