#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m world_model_lab.train \
  --config configs/throughput.toml \
  --device cuda \
  --max-steps 100

