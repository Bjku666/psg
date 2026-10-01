"""Run the zero-GPU preflight diagnostic from immutable historical outputs.

This report is explicitly retrospective: it cannot promote a new method or
open confirm/test. It exists to make the existing Gate x Chooser and exact
oracle evidence visible under the new protocol before a new aligned carrier
is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate-chooser",
        type=Path,
        default=Path("results/selective_predicate_surgery_v1/gate_chooser_full_dev.json"),
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("results/relation_decision_regret_v2/p17_gate0_full_dev.json"),
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=Path("results/relation_decision_regret_v2/p17_controls_full_fit_dev.json"),
    )
    parser.add_argument(
        "--mear",
        type=Path,
        default=Path("results/relation_decision_regret_v2/p17_mear_full_seed0.json"),
    )
    args = parser.parse_args()
    gate = json.loads(args.gate_chooser.read_text())
    oracle = json.loads(args.oracle.read_text())
    controls = json.loads(args.controls.read_text())
    mear = json.loads(args.mear.read_text())
    partial = gate["partial_oracles"]
    native = oracle["results"]["baseline"]
    exact = oracle["results"]["population_exact_oracle"]
    simple = controls["gate2"]
    report = {
        "schema_version": 1,
        "name": "counterfactual_relation_verification_v1_retrospective_diagnostic",
        "status": "retrospective_only_blocked_new_carrier",
        "sources": {
            str(path): {"sha256": sha256(path)}
            for path in (args.gate_chooser, args.oracle, args.controls, args.mear)
        },
        "D0_gate_chooser": {
            "baseline_mR": native["mR"],
            "exact_oracle_delta_mR_pp": oracle["results"]["delta_mR_pp"],
            "learned_gate_exact_chooser_delta_mR_pp": partial["learned_gate_oracle_chooser"]["delta_mR_pp"],
            "exact_gate_learned_chooser_delta_mR_pp": partial["oracle_gate_learned_chooser"]["delta_mR_pp"],
            "gate_is_bottleneck": partial["learned_gate_oracle_chooser"]["delta_mR_pp"] < 0
            and partial["oracle_gate_learned_chooser"]["delta_mR_pp"] > 0,
        },
        "O0_actionable_oracle": {
            "delta_mR_pp": oracle["results"]["delta_mR_pp"],
            "delta_R_pp": oracle["results"]["delta_R_pp"],
            "pass_headroom": oracle["results"]["delta_mR_pp"] >= 8.0,
        },
        "S0_S7_simple_controls": {
            "strongest": simple["strongest"],
            "simple_gain_pp": simple["simple_gain_pp"],
            "captured_fraction": simple["captured_fraction"],
            "complex_method_allowed_by_retrospective_gate": simple["captured_fraction"] < 0.70,
        },
        "M0_historical_mear": {
            "gain_vs_simple_mR_pp": mear["gate3"]["gain_vs_simple_mR_pp"],
            "residual_oracle_capture": mear["gate3"]["residual_oracle_capture"],
            "pass": False,
            "decision": "do_not_resurrect_historical_method",
        },
        "next": "obtain_registered_DSFormer_checkpoint_and_images_then_run_single_source_B0_export",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

