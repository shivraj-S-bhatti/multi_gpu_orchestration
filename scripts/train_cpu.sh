#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m world_model_lab.train \
  --config configs/baseline.toml \
  --device cpu \
  --max-steps 5

