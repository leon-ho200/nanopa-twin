#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-main_nhanes}"
DEVICE="${2:-cuda}"
OUT="${3:-runs}"

python -m nanopa_twin.dial fit --preset "${PRESET}" --device "${DEVICE}" --out "${OUT}"
