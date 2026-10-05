# BAP-PSG qualification abort — 2026-10-05

The corrected P3.1 identity contract was registered and audited:

- point estimate keeps scene rows;
- `file_name` is the physical-image grouping/bootstrap key;
- grouped fit/dev/confirm have zero physical-group overlap;
- official test was not used for selection.

B0 was then run on the grouped fit population with the frozen DSFormer
carrier.  The carrier support and SingleMPO mapping were held fixed.  Local
evidence was acquired only from tight or context pair crops; no learner was
trained.

Results:

| action | K=20 ΔmR | K=50 ΔmR |
|---|---:|---:|
| tight-crop GT-gated acquisition oracle | +8.29 pp | +8.78 pp |
| context-crop GT-gated acquisition oracle | +0.92 pp | +1.29 pp |
| tight local-argmax update (no learning) | +2.23 pp | +2.44 pp |
| tight random-half reacquisition | −0.96 pp | +1.95 pp |
| tight fixed-margin trigger, best tested | +0.42 pp | +0.51 pp |

The tight oracle clears the exploratory +3 pp headroom gate, but the
deployable no-learning policy captures only 26.9% / 27.8% of that headroom,
below the registered 0.35 actionability threshold.  Context reacquisition
itself fails the +3 pp acquisition gate.  Therefore B1/formal model
development is not authorized.

Status: **ABORTED at B1 actionability gate**.

No router training, confirm/test tuning, new carrier launch, or claim of a
deployable BAP-PSG method is permitted from these artifacts.
