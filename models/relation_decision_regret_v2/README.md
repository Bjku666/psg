# Relation decision regret v2: metric-exact predicate surgery

This package implements the P1.6 follow-up from the p1.5 analysis. It is
downstream of the frozen v1 carrier and never changes object masks, physical
pairs, ranking, or output budget.

```bash
python models/relation_decision_regret_v2/run_p16.py \
  --psg results/relation_decision_regret_v1/fit500/psg_subset.json \
  --fit-predictions /data2/liuhaoran/psg_data/cache/relation_decision_regret_v1/hidden_fit500_predictions.pkl \
  --dev-psg results/relation_decision_regret_v1/dev500/psg_subset.json \
  --dev-predictions /data2/liuhaoran/psg_data/cache/relation_decision_regret_v1/hidden_dev500_predictions.pkl \
  --gt-seg-root /data2/liuhaoran/psg_data/openpsg/coco_panoptic_gt/annotations \
  --output results/relation_decision_regret_v2/p16_fit500_dev500.json
```

`predicate_substitution_regret.py` is the exact teacher, `selective_repair_oracle.py`
computes KEEP/Top-L ceilings, `flip_accounting.py` decomposes rescue/damage,
and `mers.py` is an intentionally lightweight fit-only utility learner.
