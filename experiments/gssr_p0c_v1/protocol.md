# GSSR P0C raw-query admission protocol

P0C moves the mechanism test upstream of final panoptic entities. The pool is
all raw Mask2Former decoder queries. Native admission reproduces the official
postprocessor's no-object, class-confidence, pixel-competition, and overlap-
area checks and records which raw query generated each final segment.

The raw exporter stores logits and features in per-image compressed NumPy
artifacts and writes query-level scores, RLE masks, boxes, and native keep IDs
to a versioned JSONL manifest. The raw-query audit uses mask IoU only and
compares native admission with a predicate-balanced exact oracle at the same
per-image native K, plus fixed-ratio class, mask-quality, joint, and oracle
curves.

The first decomposition classifies every GT relation as supply failure, entity-
admission failure, or surviving entity admission. Pair-admission and predicate
failure are deliberately left unavailable until a frozen relation carrier
provides pair proposals and predicate outputs.

The compute-equivalent oracle diagnostic compares K admitted nodes with all
K(K-1) ordered pairs against all M supplied nodes with at most K(K-1) oracle-
selected ordered pairs. Final-panoptic box results are preliminary only; the
gate is decided on raw-query, mask-matched results.

