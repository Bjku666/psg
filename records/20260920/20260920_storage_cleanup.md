# 2026-09-20 repository storage cleanup

## Scope

The cleanup was limited to rebuildable, ignored artifacts. Source code,
checkpoints, annotations, experiment protocols, reports, manifests, logs,
compact result summaries, and the registered external prediction cache were
left in place. The pre-existing working-tree modification in
`third_party/fair_psg` was not touched.

## Removed

- 183,491,154 bytes across 201 files of ignored per-image/audit outputs from
  closed P0/P1 exploratory runs (`per_image.jsonl`, `entity_criticality.jsonl`,
  `rows.jsonl`, and derived PNG plots), excluding the canonical
  `results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3/per_image.jsonl`
  and `results/pregraph_attrition_v1/min_edit_fit500_seed0_v2/per_image.jsonl`.
- Python bytecode caches (`*.pyc`, `*.pyo`) under the repository.
- Empty `__pycache__` directories left by the bytecode cleanup.

The compact `summary.json`/`analysis.json` artifacts and the corresponding
experiment records remain the authoritative historical record. The external
prediction cache at
`/data2/liuhaoran/psg_data/cache/relation_decision_regret_v1/` was deliberately
retained because the full fit/dev carrier export has not yet completed.

## Verification

After cleanup, verify that the protected files and canonical intermediates are
present with:

```bash
test -f README.md
test -f AGENT_RUNBOOK.md
test -f experiments/relation_decision_regret_v1/P13_GATE.json
test -f results/relation_decision_regret_v1/p1_3/full_test.json
test -f results/pregraph_attrition_v1/stage_audit_fit500_seed0_v3/summary.json
test -f results/pregraph_attrition_v1/min_edit_fit500_seed0_v2/analysis.json
```

