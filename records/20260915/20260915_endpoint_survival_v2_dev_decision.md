# Endpoint survival v2 dev decision

The independent grouped dev run completed with native pixel-exact sanity on
all 4565 rows. Both frozen stage interventions improved official PQ but failed
the preregistered +3 pp predicate-balanced endpoint-support gate:

- semantic 1.0: +1.818 pp balanced support, +1.434 pp PQ;
- competition 0.25: +1.665 pp balanced support, +1.602 pp PQ.

The paired bootstrap and predicate-level decomposition are in
`results/endpoint_survival_v2/dev_frozen/paired_bootstrap.json` and
`reports/20260915/endpoint_survival_v2_dev.md`. Learner training, joint grid,
confirm evaluation and the frozen relation carrier remain unauthorized.

After preserving the summaries and this decision record, the dev float32 raw
query shards are safe to delete as reproducible intermediates. They were
deleted on 2026-09-15, releasing approximately 29 GB; `/data2` free space
returned from 107 GB to 136 GB. The deletion is not recoverable from trash but
the artifacts can be regenerated with
`experiments/endpoint_survival_v2/run_dev_pipeline.sh`.
