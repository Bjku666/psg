# Reference ambiguity A3b/A4 run

## Outcome

The CPU-only A3b discrepancy taxonomy completed. A4 was aborted before any
GPU work because the pinned DSFormer checkpoint and immutable action carrier
artifact were unavailable.

## Command

```bash
python -m models.counterfactual_relation_verification_v1.discrepancy_taxonomy \
  --psg /data1/liuhaoran/psg/datasets/psg.json \
  --panoptic-root /data1/liuhaoran/psg/datasets/openpsg/coco_panoptic_gt/annotations \
  --output results/reference_ambiguity_audit_v1/A3B_REPORT.json
```

## A3b result

- 241/241 paired physical-image groups were fully mapped using panoptic masks.
- The 2,124 relation symmetric-difference triples were decomposed as:
  1,798 missing-edge, 289 same-pair/different-predicate, and 37 multilabel
  subset triples.
- The strict curated semantic-conflict subset contained 12 triples.

## A4 stop condition

`/data1/liuhaoran/psg/checkpoints/dsformer_relation_decomp_v1_seed0/best_state.pth`
and `/data1/liuhaoran/psg/cache/counterfactual_relation_verification_v1`
were both absent. Therefore action-level signed utility could not be computed,
the sign-instability gate remained undefined, and no method training was
started.

Result: `results/reference_ambiguity_audit_v1/A4_ACTION_SIGN_GATE.json`.
