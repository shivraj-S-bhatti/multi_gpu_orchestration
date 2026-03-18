#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"
OUT_DIR="${OUT_DIR:-artifacts/profiles/nsys}"
mkdir -p "${OUT_DIR}"

PYTHONPATH=src nsys profile \
  --output="${OUT_DIR}/world_model_ddp" \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  -m world_model_lab.train \
  --config configs/throughput.toml \
  --max-steps 20 \
  --emit-nvtx
