#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-artifacts/profiles/ncu}"
mkdir -p "${OUT_DIR}"

PYTHONPATH=src ncu \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --nvtx \
  --nvtx-include "forward" \
  --export "${OUT_DIR}/world_model_forward" \
  python3 -m world_model_lab.train \
  --config configs/throughput.toml \
  --device cuda \
  --max-steps 10 \
  --emit-nvtx
