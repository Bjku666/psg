# Status

## 2026-10-06: resource-adapted multi-GPU training

The original `rels_per_batch=128` protocol exceeds the available 80 GB GPU
memory. The pinned Fair-PSG trainer now has a minimal NCCL DDP path with
distributed sampling, rank-0 validation/checkpointing, gradient accumulation,
and safe checkpoint unwrapping. BF16 autocast and fused backbone extraction
are enabled for the accelerated runs.

| Run | GPUs | Per-rank batch | Relation chunk | Accumulation | State |
|---|---|---:|---:|---:|---|
| `C0_feature_all_ddp_bs16_rel32_ga2_40ep` | 4,5 | 16 | 32 | 2 | running |
| `C1_patch4_ddp_3gpu_bs2_rel8_ga11_40ep_retry2` | 3,6,7 | 2 | 8 | 11 | running (tmux `psg_c1_patch4_retry2`) |

The C0 run has effective global batch 64. The C1 run has effective global
batch 66, the closest practical integer configuration to 64 with three GPUs.
Both are resource-adapted protocols and must be labeled as such; they are not
official-parity evidence.

Bounded DDP smoke checks passed for C0 and for the current C1 configuration.
The earlier C1 two-GPU batch-2/relation-16 run OOMed on a high-memory sample;
the batch-1 three-GPU attempt was safe but slower and was replaced by the
current relation-8 configuration. No final accuracy gate exists yet.

After each `done.txt` appears, run `evaluate_control.sh` and the corrected
SingleMPO evaluator. Smoke metrics are not final results.

## 2026-10-06: frozen evidence diagnostic

`results/relation_evidence_learning_v1/frozen_geometry_diagnostic.json`
completed on 500 grouped images (3,732 matched GT relations).  The geometry
baseline reached 24.84% balanced accuracy.  Adding full-resolution interface
geometry changed this by -1.71 pp; 2x and 4x downsampled interface changed it
by -0.69 pp and -1.97 pp; scene-context proxy changed it by -3.32 pp.  These
are mechanism diagnostics, not PSG mR/R gates, and no proposed architecture is
authorized by them.
