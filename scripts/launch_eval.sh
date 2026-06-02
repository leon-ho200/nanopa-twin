#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?path to a checkpoint .pt}"
PRESET="${2:-main_nhanes}"
DEVICE="${3:-cuda}"

python -m nanopa_twin.dial evaluate --checkpoint "${CHECKPOINT}" --preset "${PRESET}" --device "${DEVICE}"
python -m nanopa_twin.dial forecast --checkpoint "${CHECKPOINT}" --preset "${PRESET}" --device "${DEVICE}"
