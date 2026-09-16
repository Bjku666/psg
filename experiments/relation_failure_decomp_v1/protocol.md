# Relation failure decomposition v1

Registered before DSFormer training on 2026-09-15. This line measures where
ground-truth relations disappear under an identity-correct SingleMPO pipeline.
It does not assume that pair selection is the bottleneck and authorizes no
learner before the conditional-oracle gate is evaluated.

The frozen carrier is Fair PSG DSFormer at revision
`2497be40bd8bd0ca6c2ad25be1522ebf42e6623b`, trained with the official
`masks-loc-sem.json` recipe for 40 epochs. Its dense inference evaluates every
directed non-self pair. Pair budgets 20, 50, 100, and all are applied only for
the diagnostic pair-retention curve using the frozen no-relation score
`1 - P(no-relation | pair)`.

For each GT relation, map predicted masks to unique GT objects with
class-compatible mask IoU strictly greater than 0.5 and SingleMPO one-to-one
deduplication. Define nested events E (both endpoints survive and uniquely
match), P (directed pair is retained), C (correct predicate is recognized), and
T_K (correct triplet survives final top-K). Report
`P(E) * P(P|E) * P(C|E,P) * P(T_K|E,P,C)` and mutually exclusive buckets:
endpoint failure, pair failure, predicate failure, ranking failure, success.

Every result includes micro and predicate-balanced metrics, official scene-row
point estimates, physical-file grouped bootstrap intervals, per-image rows,
and the corrected Fair PSG evaluator (`--dedup fail`) for downstream R/mR.
No method is trained or selected until a single-stage conditional oracle gains
at least 3 percentage points balanced mR, with pair selection additionally
requiring at least 30% of balanced misses to be pair failures.
