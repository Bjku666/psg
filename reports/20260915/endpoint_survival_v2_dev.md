# Endpoint survival v2: independent dev decision

## Decision

The v2 method line is closed after the preregistered independent dev gate
failed for both frozen stage interventions. No learner, joint semantic/competition
grid, confirm partition, or downstream PSG carrier run is authorized.

The result is a useful transferability negative result: the structured oracle
was positive on fit500, but its frozen margins did not reproduce the required
balanced endpoint-support gain on an independent grouped dev population.

| Frozen point | Fit500 balanced Δ | Dev balanced Δ | Dev official PQ Δ | Dev gate |
| --- | ---: | ---: | ---: | --- |
| semantic margin 1.0 | +4.17 pp | +1.82 pp | +1.43 pp | fail support |
| competition margin 0.25 | +3.46 pp | +1.67 pp | +1.60 pp | fail support |

Both dev runs passed pixel-exact native replay for all 4565 rows. The native
dev baseline is balanced endpoint support `0.7357164052` and official PQ
`0.5930194660`. Semantic 1.0 reaches `0.7538966188`; competition 0.25 reaches
`0.7523674472`. Their micro endpoint-support gains are larger (`+3.35 pp` and
`+3.05 pp`), but the registered primary metric is predicate-balanced support,
so the gate correctly remains closed.

## Uncertainty and mechanism

The physical-file paired bootstrap gives semantic 1.0 a 95% CI of
`[+1.33,+2.70] pp` and competition 0.25 `[+0.42,+3.14] pp`; the probability of
reaching the required +3 pp is `0.0035` and `0.0385`, respectively. Thus the
failure is not a numerical aggregation accident.

At the predicate level, semantic intervention gains are concentrated in a
subset of predicates (for example predicate 27: +8.3 pp; predicate 47: +5.1
pp), while many predicates are unchanged. Competition has similar gains but
also regressions, including predicate 33 (`-20 pp`, only 5 dev relations) and
predicate 28 (`-8.3 pp`). This explains why micro support can pass a rough
threshold while balanced support does not.

The rescue populations remain partly complementary: semantic-positive and
competition-positive groups have Jaccard `0.4560` on dev (337 shared, 210
semantic-only, 192 competition-only), but complementarity alone is not enough
to authorize a learner when neither fixed action reproduces the primary gate.

## What is retained

The source code, evaluator tests, contracts, fit500 summaries, dev summaries,
frontiers, bootstrap outputs, and regeneration scripts remain. The large dev
float32 raw-query artifacts may be removed after this report is recorded; they
are reproducible from the pinned exporter, checkpoint, population manifest and
split manifest.
