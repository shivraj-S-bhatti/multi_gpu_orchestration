# Visualizations

These visuals are meant to make the repo easier to reason about before you even launch a profiler.

## Architecture Overview

![Architecture](assets/world_model_architecture.svg)

## Profiling Workflow

![Profiling Workflow](assets/profiling_workflow.svg)

## Sample Training Metrics

Generate this from JSONL step logs with:

```bash
python3 scripts/visualize_metrics.py \
  artifacts/metrics/tutorial_metrics.jsonl \
  docs/assets/tutorial_metrics.svg \
  --title "Tutorial Run Metrics"
```

![Sample Metrics](assets/tutorial_metrics.svg)

