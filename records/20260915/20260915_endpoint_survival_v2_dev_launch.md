# endpoint_survival_v2 independent dev launch

The corrected fit500 frontier completed before dev selection. The rule frozen
without viewing dev outcomes is the smallest passing margin per stage:
semantic 1.0 and competition 0.25. A 2000-replicate physical-file paired
bootstrap and rescue-complementarity analysis were completed first.

The dev population was materialized verbatim from the existing grouped split:
4565 relation-bearing annotation rows in 4545 physical-file groups. Official
test rows remain excluded.

Pre-flight checks passed on 2026-09-15 14:51 CST:

- configuration differences are population and frozen-point changes only;
- annotation, train images, model, panoptic GT, split and population files exist;
- 14 P0C/v2 tests passed and all launch scripts pass `bash -n`;
- output paths did not exist;
- `/data2` had 136 GB free; projected float32 dev artifacts are about 30 GB;
- GPUs 2, 5, 6 and 7 were free and selected for export; existing jobs on
  GPUs 0, 1, 3 and 4 were left untouched;
- point-specific local logs provide monitoring;
- ETA is approximately 16:45 CST, with allowance to 17:00 for shared-storage
  contention.

The tmux session is
`endpoint-survival-v2-dev4565-frozen-stage-reproduction`. It chains the
four-shard float32 export, exact merge, three parallel frozen evaluations,
frontier construction, and paired bootstrap.
