# Pre-graph attrition qualification protocol

## Experiment matrix

| Run | Factor | Population | Fixed conditions | Decision |
|---|---|---|---|---|
| `stage_audit_fit500_seed0_v3` | stage availability | `fit500_seed0.json`, 500 rows / 499 physical-file groups | frozen Mask2Former, IoU > 0.5, native per-image K, seed 0 | complete: semantic and competition losses are material |
| `near_miss_fit500_seed0_v4` | endpoint rescue cost | exact rows from stage audit | official class-weighted overlap-area rule; relation vs non-relation control | complete: narrow-loss hypothesis not supported |
| `min_edit_fit500_seed0_v2` | changed-pixel budget | exact frozen population | GT-guided fixed-K swaps, eps = 0, .1, .25, .5, 1, 2% | complete: small-edit materiality gate failed |
| `fixed_k_rescue` | only if prior gates pass | frozen fit then dev | no learner, same downstream graph evaluator | not authorized |

The existing test artifact smoke is not a matrix level and cannot affect any
decision.  No official confirm partition is opened by this protocol.

## Execution

The population is frozen by SHA-256 sampling over physical filenames, not by
image-id order:

```bash
python models/pregraph_attrition_v1/make_population_manifest.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --split-manifest results/gssr_p1_v2/manifests/official_train_grouped_split_seed0.json \
  --partition fit --max-images 500 --seed 0 \
  --output experiments/pregraph_attrition_v1/fit500_seed0.json
```

With P0C full artifacts:

```bash
python models/pregraph_attrition_v1/run_stage_decomposition.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --manifest /data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-pregraph-fit500-seed0-p0c-full-merged/manifest.jsonl \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --split train --population-manifest experiments/pregraph_attrition_v1/fit500_seed0.json \
  --output results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3
```

When raw artifacts are available, add `--manifest /path/to/p0c_full/manifest.jsonl`.
Otherwise use one-pass mode with `--images ... --model ...`; it computes and
discards tensors per image.  The grouped split manifest is still required for
partition selection.

The completed follow-up commands are:

```bash
python models/pregraph_attrition_v1/near_miss_audit.py \
  --input results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3/per_image.jsonl \
  --output results/pregraph_attrition_v1/near_miss_fit500_seed0_v4.json

python models/pregraph_attrition_v1/min_edit_oracle.py \
  --psg /data2/liuhaoran/psg_data/openpsg/psg/psg.json \
  --manifest /data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-pregraph-fit500-seed0-p0c-full-merged/manifest.jsonl \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --split train --population-manifest experiments/pregraph_attrition_v1/fit500_seed0.json \
  --output results/pregraph_attrition_v1/min_edit_fit500_seed0_v2

python models/pregraph_attrition_v1/analyze_min_edit.py \
  --input results/pregraph_attrition_v1/min_edit_fit500_seed0_v2/per_image.jsonl \
  --output results/pregraph_attrition_v1/min_edit_fit500_seed0_v2/analysis.json
```

No learner may be run after the failed minimum-change gate.

## Resource estimate and analysis

The 8-way export took about 30 seconds wall time (about 0.07 aggregate A100 GPU
hours) and the merged lossless artifacts occupy 389 MB. Repository result
files occupy about 5 MB. Stage analysis took about two minutes and the
minimum-change CPU oracle about six minutes. No API cost is involved. The primary
analysis is the predicate-balanced fixed-K endpoint-support difference between
raw, semantic, competition, and admission stages, followed by the endpoint
death-stage and changed-pixel curves. Adjacent-stage deltas use 2,000 paired
bootstrap replicates with physical filename as the sampling unit. The
competition line is eligible for a
new method only when its loss is at least 0.03 and the minimum-change oracle
shows a material gain at no more than 0.5% changed pixels.

## Qualification result

The full-sample stage supports are 0.87605, 0.80448, 0.72479, and 0.71941.
The competition point loss is 0.07969 with grouped-bootstrap 95% CI
[0.03035, 0.09618], so the first gate passes. At epsilon 0.005, however, the
minimum-change gain is only 0.01192 (95% CI [0.00454, 0.02316]) and only 15 of
500 rows improve. The registration said "material" without assigning a numeric
minimum; the proposal's 3--5 point example is therefore treated as an
interpretive target, not retroactively as a preregistered threshold. The CI
upper bound remains below 3 points, and the strict control audit shows that
relation-critical suppressed queries require larger, not smaller, repairs.
The learner is not authorized and this method line is closed.
