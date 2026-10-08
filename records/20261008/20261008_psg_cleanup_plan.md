# PSG storage cleanup plan — 2026-10-08

This plan was generated after checking active processes, checkpoint metadata,
repository references, and file sizes. Deletion was confirmed by the user and completed via Linux Trash on 2026-10-08.

## Protected

- `/data1/liuhaoran/psg/datasets/` — dataset files required for reruns.
- `/data1/liuhaoran/psg/checkpoints/dsformer_relation_decomp_v1_seed0/` — pinned
  baseline referenced by historical result records.
- `/data1/liuhaoran/psg/checkpoints/ambiguity_conditioned_relation_v1/predicate_seed0_full20ep_complete/`
  — currently active corrected E8 run; do not touch until its process exits.
- `/data1/liuhaoran/psg/checkpoints/ambiguity_conditioned_relation_v1/predicate_seed0_seed0_5ep/`
  — warm-start source for the active E8 run; keep until E8 completes.
- `/data1/liuhaoran/psg/cache/counterfactual_relation_verification_v1/paired_final_dev.pkl`
  and `paired_final_fit.pkl` — fixed development and fit subsets.
- All code, configs, experiment plans, logs, result JSON, and status records in
  the repository under `/data2/liuhaoran/project/cv/psg/`.

## Proposed deletion

The following completed or stopped checkpoint directories contain large model
state files whose launch settings and outcomes remain documented in the
repository. They total approximately 14.436 GiB:

### ACRD smoke, throughput, and duplicate controls

- `predicate_seed0_smoke1` (0.914 GiB)
- `predicate_seed0_throughput_b64_r128` (0.914 GiB)
- `predicate_seed0_throughput_b128_r256` (0.914 GiB)
- `predicate_seed0_throughput_b160_r320` (0.914 GiB)
- `predicate_seed0_full20ep` (0.914 GiB; bounded smoke run with `MAX_BATCHES=2`)
- `predicate_seed0_epoch1_b112_r224` (0.914 GiB; profiling warm-start no longer needed)
- `uniform_seed0_smoke1` (0.913 GiB)
- `uniform_seed0_epoch1_b112_r224` (0.913 GiB)
- `mask_seed0_smoke1` (0.913 GiB)
- `mask_seed0_epoch1_b112_r224` (0.913 GiB)

### Stopped relation-resolved-tokenization controls

- `C0_feature_all_batch16_probe` (1.147 GiB)
- `C0_feature_all_ddp_smoke3` (1.147 GiB)
- `C0_feature_all_ddp_bs16_rel32_ga2_40ep` (1.152 GiB; killed run)
- `C0_feature_all_seed0` (1.154 GiB; stopped run)
- `C1_patch4_ddp_3gpu_smoke` (0.351 GiB)
- `C1_patch4_ddp_3gpu_bs2_rel8_ga11_smoke` (0.351 GiB)

## Deliberately retained for now

- `paired_final.pkl` and `paired_epoch2.pkl` (~1.8 GiB total), because
  historical audit JSON files reference them.
- The final ACRD checkpoint and warm-start checkpoint until the active E8 run
  finishes. A second cleanup can remove the warm-start and redundant
  `best_state.pth` after final dev selection.

## Reproducibility retained

The repository retains the ACRD implementation, configs, launch/evaluation
scripts, Gate 3 summary, dev metrics, launch record, and the relation control
matrix/status documents. Deleting the listed checkpoint directories removes
stored weights, but leaves commands and textual records needed to rerun them.

## Completion

Sixteen listed checkpoint directories were first moved to Linux Trash and then permanently deleted on user confirmation. Their exact trash entries and metadata were removed from `/data1/.Trash-1005`; other projects in that trash were preserved. The checkpoint root is approximately 3.0G. Protected E8, warm-start, baseline, datasets, dev/fit caches, code, and textual records were verified after cleanup.
