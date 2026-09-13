# P1A corrected actionability result

## Decision

The corrected fixed-competition P1A dev gate failed. No P1B learner or GSSR
training is authorized, and the confirm partition is not opened.

## Population and contract

- official OpenPSG train only; test rows were excluded;
- grouped fit/dev/confirm manifest, with the dev partition selected before
  evaluation;
- 4,565 relation-bearing dev rows (4,655 registered dev IDs, including rows
  without relations);
- intervention after full-pool semantic eligibility and pixel competition,
  before entity admission;
- fixed-K selection over mutually exclusive winning-mask candidates.

## Results

| quantity | value |
|---|---:|
| native balanced endpoint support | 0.7357164052 |
| actionable fixed-competition oracle | 0.7457041158 |
| actionable delta | +0.0099877106 |
| required delta | +0.0600000000 |
| native assembly sanity | 4,565 / 4,565 exact |
| P1B authorization | false |

The candidate-supply ceiling (admitting every post-competition candidate) was
0.7458772376, only +0.0101608324 over native. This is an informational ceiling,
not the 200-raw-query denominator.

The compact P1 v2 artifacts intentionally omit raw-mask RLE and full mask
logits, so the same-partition 200-raw-query gap was not available. The
actionable fraction is therefore reported as null rather than using the old
test gap or relabeling the candidate ceiling.

## Reproducibility

The eight source shard summaries are under
`results/gssr_p1_v2/dev-fixed-competition-balanced-oracle-shard*`; the merged
result is `results/gssr_p1_v2/dev-fixed-competition-balanced-oracle-merged`.
The runner and merger both use the corrected fixed-competition contract, and
all repository P1 tests pass.
