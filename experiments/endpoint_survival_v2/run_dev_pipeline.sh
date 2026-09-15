#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$ROOT/experiments/endpoint_survival_v2/export_dev_float32.sh"
bash "$ROOT/experiments/endpoint_survival_v2/run_dev_reproduction.sh"
