# P0B/P0C qualification summary

## Decision

Both formal mask contracts are now complete. P0B and the raw-query P0C
material-gap gates pass; the raw-query pair-vs-entity contrast is small and
does not justify a pair-first learner. These are qualification results only:
no GSSR or DSFormer learner has been trained or authorized.

P0B's corrected box regression strengthens the mechanism signal: at 50%
retention, score Top-K is 0.3987, the micro oracle is 0.5462, and the exact
predicate-balanced oracle is 0.5916. The balanced gap is +0.1929 (95% CI
[0.1685, 0.2204]). The full-pool ceiling is 0.6346, so wrong composition
accounts for 0.1929 of the total 0.2359 budget loss.

The population discrepancy is resolved: 2,177 official scene rows correspond
to 2,161 physical images because 16 filenames occur twice with distinct
relation annotations. Point estimates use rows; bootstrap uses filename
clusters.

Relation-criticality data show a monotonic rescue trend from 11.1% for degree-0
score-dropped nodes to 50.8% for degree-4+ nodes. The preliminary final-mask
box node-vs-pair result favors the pair oracle by only 3.07 pp at 50%.

The raw-query exporter passed native-admission reconstruction, and the full
merged export contains 2,170 unique candidate filenames. At native K (mean
9.37), native admission obtains 0.6426 predicate-balanced endpoint support;
the balanced oracle at the same K obtains 0.7947, a +0.1521 gap (95% CI
[0.1296, 0.1737]). The pair oracle at the compute-equivalent K(K-1) budget is
0.7989, only +0.0042 above the entity oracle (95% CI [0.0027, 0.0060]).
The native failure decomposition is 20.0% supply, 15.7% entity-admission,
and 64.3% surviving admission (predicate-balanced fractions).

The COCO panoptic GT archive passed CRC validation and was extracted before
these audits. The next permitted step is therefore an explicitly approved
P1 learner design; this run itself does not start training.
