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

`population_metric_utility.py` derives the exact population-mR/R contribution
of one legal edit; `additivity_audit_v2.py` checks independent distinct-row
edits against full evaluator recomputation; and `repair_accounting.py` counts
only actually emitted predicate changes. `simple_action_controls.py` provides
the preregistered S0--S7 controls. `metric_exact_action_ranker.py` implements
MEAR: a candidate-relative low-rank scorer trained with metric-weighted ranking
and sign losses, with KEEP fixed at score zero. `mers.py` remains the original
P1.6 Ridge probe and is not overwritten.

P1.7 Gate-0 and Gate-2 outputs are stored under
`results/relation_decision_regret_v2/`; full-population MEAR is authorized only
after the registered fit/dev carrier export and the full simple-control gate.
