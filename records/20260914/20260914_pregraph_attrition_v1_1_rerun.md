# Pre-graph attrition v1.1 reproducibility rerun

The frozen `fit500_seed0.json` population and merged P0C full artifacts were
rerun in fresh output directories on 2026-09-14.  No completed result directory
was overwritten.

## Commands

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
/data2/liuhaoran/venvs/fair_psg_p0/bin/python \
models/pregraph_attrition_v1/run_stage_decomposition.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --manifest /data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-pregraph-fit500-seed0-p0c-full-merged/manifest.jsonl \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --split train --population-manifest experiments/pregraph_attrition_v1/fit500_seed0.json \
  --output results/pregraph_attrition_v1/stage_audit_fit500_seed0_v1_1

/data2/liuhaoran/venvs/fair_psg_p0/bin/python \
models/pregraph_attrition_v1/near_miss_audit.py \
  --input results/pregraph_attrition_v1/stage_audit_fit500_seed0_v1_1/per_image.jsonl \
  --output results/pregraph_attrition_v1/near_miss_fit500_seed0_v1_1.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
/data2/liuhaoran/venvs/fair_psg_p0/bin/python \
models/pregraph_attrition_v1/min_edit_oracle.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --manifest /data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-pregraph-fit500-seed0-p0c-full-merged/manifest.jsonl \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --split train --population-manifest experiments/pregraph_attrition_v1/fit500_seed0.json \
  --output results/pregraph_attrition_v1/min_edit_fit500_seed0_v1_1

/data2/liuhaoran/venvs/fair_psg_p0/bin/python \
models/pregraph_attrition_v1/analyze_min_edit.py \
  --input results/pregraph_attrition_v1/min_edit_fit500_seed0_v1_1/per_image.jsonl \
  --output results/pregraph_attrition_v1/min_edit_fit500_seed0_v1_1/analysis.json
```

## Results

The stage point estimates are identical to the registered run:

| stage | predicate-balanced fixed-K endpoint support |
|---|---:|
| raw | 0.8760475 |
| semantic | 0.8044821 |
| competition | 0.7247917 |
| native admission | 0.7194094 |

At the registered 0.5% changed-pixel budget, the minimum-change oracle reaches
0.7313304, for a +0.0119210 gain over native (95% paired physical-file
bootstrap CI [0.0045439, 0.0231645]); 15/500 images improve.  The near-miss
relation-minus-control medians are unchanged: area-ratio difference -0.10144,
deficit-fraction difference +0.10112, and rescue-cost difference +513.05.

The base Python environment failed once with a NumPy representation error;
the repository-recorded `fair_psg_p0` environment completed successfully with
27/27 unit tests passing under `PYTHONPATH=.`.  This is an environment note,
not a data or metric deviation.

The registered small-edit gate remains failed.  No EPQC/query-survival learner
was launched or authorized.
