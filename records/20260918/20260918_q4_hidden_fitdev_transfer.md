# Q4 hidden fit-to-dev transfer

## Run contract

- Carrier: frozen DSFormer checkpoint `dsformer_relation_decomp_v1_seed0`.
- Training labels: exact fit500 marginal-utility teacher only.
- Evaluation labels: dev500 only; no confirm/test rows were read.
- Features: F0 = six score features; F1 = F0 + exported 384-D final
  relation token.
- Selection: fixed legal directed-pair support, budgets K=20 and K=50.

Pair-token exports:

```text
results/relation_decision_regret_v1/hidden_fit500/predictions.pkl
results/relation_decision_regret_v1/hidden_dev500/predictions.pkl
```

Transfer outputs:

```text
results/relation_decision_regret_v1/hidden_transfer_f0.json
results/relation_decision_regret_v1/hidden_transfer_f1.json
```

## Results

The subset does not cover all 56 predicates, so these are R guardrail results,
not an mR gate.

| budget | feature/model | dev utility AUC | dev positive AUC | dev R | gain vs baseline |
|---:|---|---:|---:|---:|---:|
| 20 | F0 tree | 0.9147 | 0.9222 | 0.42148 | -0.129 pp |
| 20 | F0 classifier | — | 0.9222 | 0.42289 | +0.012 pp |
| 20 | F1 tree | 0.9189 | 0.9359 | 0.42807 | +0.529 pp |
| 20 | F1 classifier | — | 0.9359 | 0.43015 | +0.737 pp |
| 50 | F0 tree | 0.9146 | 0.9224 | 0.48460 | +0.166 pp |
| 50 | F0 classifier | — | 0.9224 | 0.48682 | +0.388 pp |
| 50 | F1 tree | 0.9197 | 0.9362 | 0.48679 | +0.385 pp |
| 50 | F1 classifier | — | 0.9362 | 0.48997 | +0.703 pp |

## Gate decision

F1 is directionally better than F0 on both budgets and improves positive-label
transfer AUC by about 1.4 pp.  However, the fixed-K R gain remains below the
registered 1.0 pp promotion threshold (and no full-population mR gate exists on
these 500-image subsets).  Therefore DARR is **not promoted** yet.  The result
supports a bounded BURL/boundary-aware prototype on the complete v2 fit/dev
population; no confirm/test evaluation is authorized.
