# Agent Contract

## Before work

1. Read the frozen P0 protocol plus active `gssr_p0b_v1` and `gssr_p0c_v1`
   protocols/plans.
2. Read each line's `RUN_STATUS.json` when present.
3. Check dataset, weight, source revision, disk, and GPU paths read-only.
4. Never overwrite a completed run directory or modify upstream datasets.

## Scientific constraints

- Keep the raw candidate pool and exact node budget fixed within a comparison.
- P0 changes only the entity selection rule; it does not train a learner.
- Use class-compatible IoU > 0.5 matching, matching Fair PSG SingleMPO.
- One predicted entity may map to at most one GT entity and each GT entity may
  contribute at most once.  Duplicate nodes must not inflate recall.
- Label exact MILP results by their objective: `gt_set_oracle_micro` or
  `gt_set_oracle_balanced`; never call a greedy result oracle.
- Keep official scene rows for point estimates and group bootstrap samples by
  physical-image `file_name`.
- Never run `mask_quality_topk` unless `mask_quality` is explicitly exported.
- `endpoint_support` is a mechanism metric, not downstream PSG performance.
- Only corrected SingleMPO is admissible for downstream DSFormer results.
- Do not start GSSR training until the P0 gates pass.

## Storage and lineage

- Source/config/tests live under `models/<line>_vN/`.
- Frozen protocols live under `experiments/<line>_vN/`.
- Each launch creates `results/<line>_vN/<run-id>/`; no in-place retries.
- Large datasets and checkpoints live outside this repository and are referenced
  by absolute path plus SHA256.
- Important commands and deviations are recorded under `records/YYYYMMDD/`.
