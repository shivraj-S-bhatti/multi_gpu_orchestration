# Visualizations

These visuals are meant to make the repo easier to reason about before you even launch a profiler.

## Architecture Overview

What it means:

- This is the static map of the training pipeline.
- Read it left to right: synthetic dataset -> modality encoders -> temporal backbone -> prediction heads.
- It tells you where image compute, state/action compute, sequence modeling, and prediction losses live.

How to use it:

- Look at this first before reading the code so you know the high-level data flow.
- Use it while reading [`src/world_model_lab/data.py`](../src/world_model_lab/data.py), [`src/world_model_lab/model.py`](../src/world_model_lab/model.py), and [`src/world_model_lab/train.py`](../src/world_model_lab/train.py).
- Treat each box as a profiling hypothesis:
  `CNN Encoder` suggests conv kernels, `MLP Encoder` suggests GEMMs, and `Causal Transformer` suggests attention/MLP sequence kernels.

What to study:

- Which parts are image-heavy versus sequence-heavy.
- Which blocks are likely to scale well with batch size.
- Which blocks are likely to show up in Nsight Systems versus Nsight Compute.

![Architecture](assets/world_model_architecture.svg)

## Profiling Workflow

What it means:

- This is the dynamic view of one training step.
- The top panel shows the order of work across CPU input staging and two GPU ranks.
- The bottom panels show the questions each profiler is best at answering.

How to use it:

- Use this after you can run the trainer successfully once.
- Compare the labels in the image with the NVTX ranges emitted by the trainer:
  `host_to_device`, `forward`, `loss`, `backward`, and `optimizer_step`.
- When a step feels slow, use this image to decide which tool to open first:
  `nsys` for time layout and overlap, `ncu` for kernel-level diagnosis.

What to study:

- Whether H2D copies overlap with useful compute.
- Whether both ranks stay aligned or one lags behind.
- Whether communication is becoming visible inside the backward region.

![Profiling Workflow](assets/profiling_workflow.svg)

## Sample Training Metrics

What it means:

- This chart is the lightweight scorecard for a run.
- `Total Loss` shows whether optimization is moving in the right direction.
- `Step Time (ms)` shows end-to-end latency per optimizer step.
- `Frames / Second` shows effective throughput.

How to use it:

- Generate one chart for a baseline run.
- Change one knob at a time, then regenerate the chart and compare:
  batch size, AMP dtype, number of workers, `pin_memory`, `channels_last`, or DDP world size.
- Use it as a quick summary before opening a full profiler trace.

What to study:

- If step time drops while throughput rises, the change likely helped.
- If throughput does not improve after adding GPUs, the workflow image tells you to inspect overlap and communication next.
- If loss behavior changes after a performance tweak, you may have traded speed for unstable training.

Generate this from JSONL step logs with:

```bash
python3 scripts/visualize_metrics.py \
  artifacts/metrics/tutorial_metrics.jsonl \
  docs/assets/tutorial_metrics.svg \
  --title "Tutorial Run Metrics"
```

![Sample Metrics](assets/tutorial_metrics.svg)

## Recommended Study Order

1. Read the architecture image to understand the static pipeline.
2. Run the CPU or single-GPU trainer once.
3. Use the profiling workflow image to understand what happens during one step.
4. Generate the metrics chart and compare runs after changing one performance knob.
5. Open `nsys` first for timeline questions, then `ncu` for suspicious kernels.
