# Pre-graph attrition qualification protocol

## Experiment matrix

| Run | Factor | Population | Fixed conditions | Decision |
|---|---|---|---|---|
| `stage_audit_fit500` | stage availability | grouped official-train fit, 500 images | frozen Mask2Former, IoU > 0.5, native per-image K, seed 0 | locate largest transition loss |
| `near_miss_fit500` | endpoint rescue cost | exact rows from stage audit | same model, masks, thresholds, and K | test narrow-loss hypothesis |
| `min_edit_fit500` | changed-pixel budget | exact rows from stage audit | GT-guided fixed-K swaps, eps = 0, .1, .25, .5, 1, 2% | require material gain at <= .5% |
| `fixed_k_rescue` | only if prior gates pass | frozen fit then dev | no learner, same downstream graph evaluator | decide whether to register a learned method |

The existing test artifact smoke is not a matrix level and cannot affect any
decision.  No official confirm partition is opened by this protocol.

## Execution

With P0C full artifacts:

```bash
python models/pregraph_attrition_v1/run_stage_decomposition.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --split train --split-manifest results/gssr_p1_v2/manifests/official_train_grouped_split_seed0.json \
  --partition fit --max-images 500 --output results/pregraph_attrition_v1/stage_audit_fit500
```

When raw artifacts are available, add `--manifest /path/to/p0c_full/manifest.jsonl`.
Otherwise use one-pass mode with `--images ... --model ...`; it computes and
discards tensors per image.  The grouped split manifest is still required for
partition selection.

Then run `near_miss_audit.py` and `min_edit_oracle.py` on the same image list.
Do not run a learner between these steps.

## Resource estimate and analysis

One-pass storage is O(number of JSON lifecycle rows), with no full-mask cache.
GPU time is approximately 500 forward passes plus CPU matching/MILP; start with
one shard and stop if filesystem headroom falls below 20 GB.  The primary
analysis is the predicate-balanced fixed-K endpoint-support difference between
raw, semantic, competition, and admission stages, followed by the endpoint
death-stage and changed-pixel curves.  The competition line is eligible for a
new method only when its loss is at least 0.03 and the minimum-change oracle
shows a material gain at no more than 0.5% changed pixels.
