# RECAP-PSG diagnostic abort — 2026-10-04

Followed the registered sequence from the RECAP-PSG proposal. The existing
strict p2.9/evidence-accessibility rerun was checked first and is closed at the
accessibility gate: E1/E2 did not satisfy the joint select/calibration gate
(`select capture=0.201`, `calib capture=0.0843`; selected `delta_mR` was
negative).

The next B0 source-identity audit was executed against the verified OpenPSG
annotation. It exited with status 2: 48,749 rows, 48,508 unique physical file
names, 241 collision groups / 482 rows, so the registered `dedup=fail`
contract is not met. The detailed artifact is
`results/counterfactual_relation_verification_v1/source_identity_audit_20261004.json`.

No crop oracle, router training, dev/test tuning, or new carrier launch was
authorized. The experiment is stopped pending a re-registered source identity
contract that resolves the collisions.
