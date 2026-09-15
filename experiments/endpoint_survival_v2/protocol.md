# Endpoint survival v2 qualification protocol

`endpoint_survival_v1` is frozen as a smoke implementation. V2 corrects the
population aggregation, PQ evaluator, E0 controls, baseline contract, and PQ
gate before any fit-500 evidence is produced.

V2 also requires the `endpoint_survival_v2` artifact profile, which stores
mask logits as float32. The earlier p0c_full profile stored them as float16 and
cannot exactly replay its frozen winner map. Every v2 point hard-fails unless
the float32 logits reproduce the stored native winner map and official
panoptic assembly pixel-for-pixel.

The primary support metric is computed once over the complete frozen
population: for each observed predicate, sum supported and total relations
over all images, divide those totals, then average across predicates. PQ uses
the official COCO category-wise matching and aggregation contract from
`cocodataset/panopticapi@7bb4655548f98f3fedc07bf37e9040a992b054b0`,
including its strict IoU threshold, declared GT areas, void removal, and crowd
handling. Values are normalized to `[0,1]`; therefore a one
percentage-point degradation is `-0.01`.

The first matrix contains one explicit native run plus separate semantic and
competition margins `0.1, 0.25, 0.5, 1, 2`. It does not include a joint grid.
Only Endpoint Support, Balanced Endpoint Support, official PQ/PQTh/PQSt, and
intervention-size metrics are reported. No endpoint proxy is named R@K or
mR@K.

E0 matches controls within image and exact category, uses category as a
categorical fixed effect, and defines query rank by descending joint score.

A learner is authorized only if at least one structured-oracle point obtains
at least `+0.03` balanced support relative to the explicit native baseline and
official PQ changes by at least `-0.01`.

```bash
python -m pytest -q models/endpoint_survival_v2/tests

PSG=/data2/liuhaoran/psg_data/openpsg/psg/psg.json \
MANIFEST=/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/psg-endpoint-survival-v2-fit500-float32-merged/manifest.jsonl \
GT=/data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
DEVICE=cuda:0 bash experiments/endpoint_survival_v2/run_matrix.sh
```

On the registered eight-A100 host, the formal run uses the hardware-specific
two-wave launcher (GPUs 2 and 4 are intentionally left to existing jobs):

```bash
bash experiments/endpoint_survival_v2/launch_fit500_frontier.sh
```

After the fit500 frontier and bootstrap are complete, freeze the smallest
passing margin for each stage in `FROZEN_DEV_PLAN.json`. The independent full
dev reproduction is one native-relative comparison, not another search:

```bash
bash experiments/endpoint_survival_v2/run_dev_pipeline.sh
```
