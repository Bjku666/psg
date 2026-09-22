# Evidence accessibility v1

This is a fit-only, zero-GPU probe of the registered p2 hypothesis: useful
fixed proposals exist, but the compact carrier may omit evidence needed to
rank them.  The probe keeps the frozen candidate support and exact insertion /
removal utility teacher fixed.  It compares E0 (score statistics), E1 (E0 plus
384-D pair features), and E2 (E1 plus endpoint class and box geometry), using
only Ridge and HistGradientBoosting controls.

The qualification gate is evaluated on grouped train/select/calib partitions:
positive utility capture must reach 20% in both select and calib, with positive
`delta_mR`, `delta_R >= -0.5 pp`, and edit rate <=5%.  No dev or official test
data is used for tuning.  A failure authorizes the next visual-accessibility
preflight only; it does not authorize a visual verifier by itself.

