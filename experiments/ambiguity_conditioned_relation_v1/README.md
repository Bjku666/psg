# Ambiguity-conditioned relation computation v1

This line implements the first E3--E5 carrier from the registered
ambiguity-conditioned evidence-acquisition proposal.  It keeps the OpenPSG
annotation, Fair-PSG loader, corrected SingleMPO evaluator, seed, and
train/dev/test protocol fixed.  The backbone and patch memory are computed
once per image; each relation pair performs K-token cross-attention.

The three profiling modes are matched controls:

* `uniform`: image-shared static top-K retrieval (sparse-evidence control)
* `mask`: subject/object mask-overlap top-K retrieval (DSFlash-style spatial control)
* `predicate`: coarse top-M predicate hypotheses select different visual tokens

The initial launch is bounded to two train batches per mode.  No accuracy
claim or method promotion is allowed until all three modes pass shape,
peak-memory, and throughput checks, followed by a registered one-epoch run.
