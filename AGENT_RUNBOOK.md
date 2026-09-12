# Agent Contract

## Before work

1. Read `experiments/gssr_p0_v1/protocol.md` and `P0_PLAN.json`.
2. Read `results/gssr_p0_v1/RUN_STATUS.json`.
3. Check dataset, weight, source revision, disk, and GPU paths read-only.
4. Never overwrite a completed run directory or modify upstream datasets.

## Scientific constraints

- Keep the raw candidate pool and exact node budget fixed within a comparison.
- P0 changes only the entity selection rule; it does not train a learner.
- Use class-compatible IoU > 0.5 matching, matching Fair PSG SingleMPO.
- One predicted entity may map to at most one GT entity and each GT entity may
  contribute at most once.  Duplicate nodes must not inflate recall.
- Label the exact MILP result `gt_set_oracle`; never call a greedy result oracle.
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

