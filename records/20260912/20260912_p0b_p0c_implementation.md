---
date: 2026-09-12
experiment: gssr_p0b_v1_and_gssr_p0c_v1
status: p0b_p0c_formal_gates_complete_learner_pending
---

# P0B/P0C implementation record

The frozen `gssr_p0_v1` source and results were not changed. P0B separates the
micro and predicate-balanced MILP objectives, rejects missing mask-quality
fallbacks, groups bootstrap samples by physical-image filename, adds a 100%
candidate-pool ceiling, decomposes budget loss, and exports entity criticality.
P0C adds a raw Mask2Former query schema/exporter, query-aware reproduction of
native panoptic admission, a raw-query supply/admission audit, and a compute-
equivalent node-vs-pair oracle.

## Identity contract

The relation-containing test population has 2,177 official scene rows but
2,161 unique COCO filenames. There are exactly 16 duplicate filename groups;
each has identical entity and panoptic annotations but distinct relations.
Candidate filenames are unique and cover all 2,161 physical images. Point
estimates retain all scene rows. Bootstrap sampling uses the 2,161 filename
clusters.

## Full box regression

This is still a final-panoptic, box-IoU diagnostic and is not the formal mask
claim. At 50% retention:

| arm | predicate-balanced endpoint support |
| --- | ---: |
| score Top-K | 0.3987 |
| micro set oracle | 0.5462 |
| balanced set oracle | 0.5916 |
| full pool (100%) | 0.6346 |

The balanced-oracle gap is +0.1929 with filename-cluster paired-bootstrap 95%
CI [0.1685, 0.2204] (1,000 resamples). The 0.2359 total budget loss decomposes
into 0.0430 unavoidable capacity loss and 0.1929 wrong-composition loss.

The criticality audit includes all 24,880 GT entities. Among score-dropped
entities, balanced-oracle rescue rate rises monotonically with relation degree:
11.1% (degree 0), 22.9% (1), 31.0% (2), 39.6% (3), and 50.8% (4+). The score-
degree Spearman correlation among supplied entities is 0.3935. A diagnostic
score-vs-degree plot is stored with the run.

## Compute-equivalent node-vs-pair preliminary

At the 50% final-mask box diagnostic, the balanced entity oracle reaches
0.5916 and the balanced pair oracle reaches 0.6223 under the same average
K(K-1) pair budget. Pair minus entity is +0.0307, 95% CI [0.0269, 0.0355].
Entity selection therefore captures most of this preliminary pair-oracle
ceiling, but only the raw-query mask-matched run may decide the gate.

## Raw-query smoke

One image was exported with 200 raw queries. Each query has independent class,
mask-quality, and joint scores, an RLE mask, box, decoder feature, pixel-pooled
feature, and native keep/segment identity. The artifact is 3.2 MB; 28/200
queries survived native admission. The query-aware native reconstruction was
checked pixel-for-pixel and metadata-for-metadata against the official
Transformers postprocessor.

The smoke lives outside Git at:

`/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/smoke1_r2`

## Formal P0B mask-IoU result

The COCO panoptic archive was verified and extracted to
`/data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations`. The formal
P0B mask-IoU audit passed its 50% gate: score Top-K 0.3980, micro oracle
0.5421, balanced oracle 0.5912, and balanced gap +0.1932 (95% CI
[0.1695, 0.2199]). The full-pool ceiling is 0.6356; total loss 0.2376 splits
into 0.0444 capacity loss and 0.1932 composition loss.

## Raw-query P0C result

The four raw-query shards were merged into
`/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/fulltest_mask2former_swin_large_raw_queries_merged_v1`
with 2,170 unique candidate filenames. The mask-IoU audit covered 2,177
relation-containing scene rows (2,161 physical-image bootstrap groups).
At the native admission budget (mean K=9.37), native admission reaches
predicate-balanced endpoint support 0.6426, while the balanced oracle at the
same K reaches 0.7947: gap +0.1521, bootstrap 95% CI [0.1296, 0.1737]. This
passes the material-gap gate. The compute-equivalent pair oracle reaches
0.7989, only +0.0042 over the entity oracle (95% CI [0.0027, 0.0060]), so
pair selection is non-material on the raw-query mask-matched contract.
Failure decomposition at native admission is 20.0% supply failure,
15.7% entity-admission failure (predicate-balanced fractions), and 64.3%
surviving entity admission.

## Remaining blocker

No data or audit blocker remains. Learner and DSFormer training are still
unauthorized pending an explicit decision on the next experiment phase.
