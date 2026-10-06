# Relation evidence learning v1

This directory registers the frozen diagnostics requested before opening a new
relation-token architecture.  It does not authorize R²Former, InterfaceFormer,
or evidence-dropout training.

The diagnostic uses the existing 500-image frozen Fair-PSG carrier cache and
the official PSG annotations.  For GT relations whose two endpoints are
uniquely matched by the carrier, it compares held-out predicate recognition
from:

* endpoint and box geometry;
* the same features plus mask-interface geometry;
* the same features plus low-resolution (2x/4x) interface geometry;
* the same features plus scene-level context statistics.

Image IDs are kept intact across the grouped train/test split.  The output is
mechanism evidence only; it is not an official PSG mR/R result and cannot
replace the registered C0/C1 simple-control gate.

