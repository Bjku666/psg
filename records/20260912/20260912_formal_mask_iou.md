---
date: 2026-09-12
experiment: gssr_p0b_v1
stage: formal_mask_iou
status: gate_passed
---

# Formal Mask2Former mask-IoU qualification

The official COCO panoptic archive completed at its declared 860,725,834-byte
size. Both the outer archive and nested `panoptic_val2017.zip` passed CRC checks;
exactly 5,000 validation PNGs were safely extracted.

The full audit used 2,177 relation-containing OpenPSG scene rows, 2,161
physical-image bootstrap groups, class-compatible mask IoU strictly greater
than 0.5, and 1,000 paired bootstrap resamples.

At 50% retention, score Top-K reached 0.39797 predicate-balanced endpoint
support and the exact balanced oracle reached 0.59122. The gap is +0.19324,
95% CI [0.16948, 0.21989], exceeding the registered +0.04 gate.

The full-pool ceiling is 0.63559. Total budget loss 0.23761 decomposes into
0.04437 unavoidable capacity loss and 0.19324 wrong-composition loss.

Under the matched `K(K-1)` pair-compute contract, the balanced entity oracle is
0.59122 and pair oracle is 0.62270 at 50%; pair minus entity is +0.03149,
95% CI [0.02757, 0.03623]. This remains a final-panoptic carrier result; the
decisive raw-query audit follows.

No learner or DSFormer training was started.

