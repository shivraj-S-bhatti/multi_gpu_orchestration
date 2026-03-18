#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"

PYTHONPATH=src torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  -m world_model_lab.train \
  --config configs/throughput.toml \
  --max-steps 100

