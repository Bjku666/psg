# Endpoint survival qualification protocol

All experiments use the frozen P0C raw-query manifest and the fit-500 grouped
population. E0 compares relation-critical and non-critical GT entities within
image after matching area, thing/stuff, category, raw mask IoU, class score,
mask quality and decoder query rank. A logistic coefficient on
predicate-balanced graph criticality is the residual test.

E1 and E2 add a bounded query-level margin (0.1, 0.25, 0.5, 1, 2 logit units)
to relation-critical queries, then rerun the complete native Mask2Former
pipeline. E1 changes semantic class logits; E2 changes pixel competition
log-scores. Every point reports PQ, class-bucket PQ, endpoint support and
predicate-balanced graph proxies. These are qualification oracles only; they do
not authorize learner training unless endpoint-support gain is at least 3 pp
and PQ loss is no worse than 1 pp.

Example commands:

```bash
python models/endpoint_survival_v1/e0_visual_difficulty_control.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --manifest /data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-pregraph-fit500-seed0-p0c-full-merged/manifest.jsonl \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --stage-audit results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3/per_image.jsonl \
  --population-manifest experiments/pregraph_attrition_v1/fit500_seed0.json \
  --output results/endpoint_survival_v1/e0_fit500.json

python models/endpoint_survival_v1/run_margin_oracle.py --mode semantic --margin 0.5 ...
python models/endpoint_survival_v1/run_margin_oracle.py --mode competition --margin 0.5 ...
```
