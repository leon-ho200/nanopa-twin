#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-runs}"
DEVICE="${2:-cpu}"

python -m nanopa_twin.dial encode --preset main_nhanes --out "${OUT}" --device "${DEVICE}"
