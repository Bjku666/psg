# Formal mask-IoU result

The P0B formal mask gate passes decisively. At 50% entity retention,
predicate-balanced endpoint support is 0.39797 for score Top-K and 0.59122 for
the exact balanced oracle: +0.19324 with filename-cluster paired-bootstrap 95%
CI [0.16948, 0.21989].

The 100% full-pool ceiling is 0.63559. Of the 0.23761 total budget loss, 0.04437
is capacity loss and 0.19324 is wrong-composition loss. Rescue among score-
dropped nodes increases from 10.5% at relation degree 0 to 47.1% at degree 4+.

The compute-equivalent pair oracle exceeds the entity oracle by only 0.03149 at
the 50% budget. Raw-query export and mask-matched admission analysis are now the
remaining qualification stage.

