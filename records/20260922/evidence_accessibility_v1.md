# Evidence accessibility v1 — 2026-09-22

## Decision

**KILL / 中止 learned-surgery 主线。** No verifier training, second carrier,
dev/test access, or candidate-support changes are authorized.

## Registered sequence

The fit-only cheap probe was completed before this visual preflight:

- E0 compact representation: failed the pre-registered 20% positive-utility
  capture gate on both select and calib; best observed select capture was about
  4.45% and calib about 10.52%.
- E1 endpoint class/box geometry: failed; no stable improvement over E0.
- E2 predicted-mask geometry: failed; no stable improvement over E0.
- Many operating points had `delta_R < -0.5 pp`.

These results are recorded in
`results/evidence_accessibility_v1/strict_fit500_probe.json`.

## Visual preflight: root cause and controls

The frozen visual export initially failed for three independently reproduced,
non-model reasons:

1. CUDA backbone features were multiplied by CPU segmentation weights. The
   minimal fix moves the read-only segmentation tensor to `feat.device`.
2. The inference result accumulator omitted the three visual-evidence lists,
   producing `KeyError: 'subject_features'`. The lists were initialized in the
   default result contract.
3. Torch-created NumPy views were a same-name but non-identical `numpy.ndarray`
   class under the installed Torch/NumPy ABI; direct pickle failed. Export
   arrays are canonicalized with `np.array(..., copy=True)`, validated by a
   standalone pickle smoke.

After these fixes, the 500-image `fit_000` visual preflight completed all 63
batches successfully and produced:

`results/evidence_accessibility_v1/visual_preflight/fit_000_visual.pkl`

The artifact contains 500 images and 220,648 candidate pairs. Subject,
object, and union evidence are finite 256-D float32 arrays; pair features are
384-D. File size is approximately 2.1 GB.

## Final falsification / contract check

The visual artifact cannot be joined to the registered frozen carrier:

- carrier/raw predictions: 500 image IDs
- visual `fit_000` shard: 500 image IDs
- exact intersection: 409 IDs
- 91 carrier IDs and 91 visual IDs are disjoint

The mismatch is at the image-ID level (not row ordering), so positional or
partial joining would invalidate the utility teacher and violate the frozen
evidence contract. Re-exporting another unaligned 500-image shard would not
produce a falsifiable probe and would consume substantial disk (the successful
shard is already ~2.1 GB with only ~64 GB available).

Therefore the visual line is **blocked/killed at evidence accessibility
contract**, before any learned visual verifier is allowed. The overall
learned-surgery line is killed by the earlier E0/E1/E2 failures and this
visual contract failure.

## Code/environment notes

The export changes are limited to frozen inference evidence plumbing and NumPy
ABI compatibility; no model weights, labels, candidate support, or metrics
were changed. Compilation and the synthetic visual-output smoke passed before
the real export.

