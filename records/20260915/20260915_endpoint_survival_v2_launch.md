# endpoint_survival_v2 launch record

## Contract corrections

- `endpoint_survival_v1` is frozen as smoke-only.
- Balanced endpoint support is aggregated from predicate hit/denominator
  totals over the complete population, not averaged per image.
- PQ follows the category-wise COCO panopticapi contract pinned to commit
  `7bb4655548f98f3fedc07bf37e9040a992b054b0`.
- E0 exact-matches category, treats category categorically in its residual
  model, and uses stable descending joint-score rank.
- The frontier requires an explicit native margin-zero result and uses a
  normalized `-0.01` PQ gate.
- R@K/mR@K endpoint proxies were removed from qualification output.
- Intervention size includes changed-pixel fraction, changed queries,
  semantic eligibility flips, competition winner flips, and segment-count
  change.

## Additional native-fidelity gate

The old p0c_full artifacts stored mask logits as float16 and failed an exact
winner-map replay check. A new backward-compatible exporter profile stores
float32 mask logits. Six shards covering 499 physical files (500 registered
rows with one duplicate physical-file group) were exported and merged at:

```text
/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-endpoint-survival-v2-fit500-float32-merged
```

Each v2 point now hard-fails if the native winner map or native panoptic
assembly differs pixel-for-pixel from its stored verified output. A two-image
smoke passed this gate.

## Pre-flight and launch

- Related tests: 13 passed (P0C + endpoint_survival_v2).
- Inputs: annotation, population, float32 manifest, GT, and stage audit exist.
- GPUs: 0, 1, 3, 5, 6, 7; GPUs 2 and 4 excluded for existing work.
- Tracking: local point-specific logs under
  `results/endpoint_survival_v2/fit500_seed0/logs/`.
- Ten-image measured runtime: 38.6 seconds; two six-GPU waves are expected to
  finish in roughly one hour, subject to shared filesystem contention.
- Run name: `endpoint-survival-v2-fit500-seed0-oracle-frontier`.
- No learner and no semantic/competition joint grid is authorized.
