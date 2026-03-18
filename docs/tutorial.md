# Tutorial: Learning Multi-GPU World-Model Profiling

This tutorial is designed to mirror the workflow behind the original resume bullet, but on a compact project you can actually understand in one sitting.

## 1. Start From the Real Goal

You are not just trying to "train a model."

You are trying to answer systems questions like:

- Is my GPU actually busy?
- Am I bottlenecked by dataloading or host-to-device copies?
- Does multi-GPU scaling improve throughput enough to justify communication overhead?
- Which kernels are low-occupancy or memory-bound?

This repo is intentionally small so those questions stay visible.

## 2. Read the Pipeline Once

The full path is:

1. `SyntheticWorldModelDataset` generates multimodal trajectories.
2. The dataloader batches them into long-horizon sequences.
3. The trainer moves batches to device and records a `host_to_device` range.
4. The world model fuses image and state/action streams.
5. A causal transformer models temporal dynamics.
6. Decoder heads predict next frame, next proprio state, reward, and done.
7. The trainer measures step time and throughput and writes JSONL metrics.

Core code:

- Dataset: [`src/world_model_lab/data.py`](../src/world_model_lab/data.py)
- Model: [`src/world_model_lab/model.py`](../src/world_model_lab/model.py)
- Trainer: [`src/world_model_lab/train.py`](../src/world_model_lab/train.py)

## 3. Run the Easiest Possible Version First

CPU smoke run:

```bash
bash scripts/train_cpu.sh
```

This is not about speed. It is about establishing that the loop, loss, logging, and checkpoint paths all work.

## 4. Move to One GPU

Single GPU:

```bash
bash scripts/train_single_gpu.sh
```

What to watch:

- `step_time_sec`
- `h2d_time_sec`
- `frames_per_second`

Those are your first throughput signals. If step time stays high while `h2d_time_sec` is tiny, compute is more likely the bottleneck than transfers.

## 5. Move to DDP

Two GPUs:

```bash
NUM_GPUS=2 bash scripts/train_multi_gpu.sh
```

Now you are asking a different question:

> Did global throughput scale meaningfully, or did communication and input overhead eat the gain?

Look for:

- higher `frames_per_second`
- similar or slightly higher `step_time_sec`
- diminishing returns if the per-rank batch gets too small

## 6. Capture an Nsight Systems Trace

```bash
NUM_GPUS=2 bash scripts/profile_nsys.sh
```

In `nsys`, inspect:

1. Whether `host_to_device` is a thin bar or a wide stall.
2. Whether forward/backward regions leave bubbles between steps.
3. Whether NCCL collectives become visible bottlenecks.
4. Whether dataloader work is late enough to starve the next step.

The most important beginner habit is to stop guessing from wall-clock time alone. Timeline tools tell you *where* the time is going.

## 7. Capture an Nsight Compute Report

```bash
bash scripts/profile_ncu.sh
```

In `ncu`, inspect:

1. Occupancy on attention and GEMM-heavy kernels.
2. Whether kernels are memory-bound or compute-bound.
3. How much work is happening in many tiny kernels versus a few heavy kernels.
4. Whether your batch size is large enough to keep kernels well utilized.

Nsight Systems tells you which region is slow. Nsight Compute tells you why a kernel inside that region is slow.

## 8. Compare Baseline vs Throughput Configs

The repo gives you two starting points:

- [`configs/baseline.toml`](../configs/baseline.toml)
- [`configs/throughput.toml`](../configs/throughput.toml)

The throughput config turns on more aggressive settings like:

- `channels_last`
- `pin_memory`
- more dataloader workers
- BF16 autocast
- cuDNN benchmark

Treat these as experiments, not magic flags. Flip one thing at a time and re-measure.

## 9. Generate a Plot From Training Logs

The trainer writes JSONL metrics. You can turn them into an SVG without external plotting libraries:

```bash
python3 scripts/visualize_metrics.py \
  artifacts/metrics/tutorial_metrics.jsonl \
  docs/assets/tutorial_metrics.svg \
  --title "Tutorial Run Metrics"
```

This keeps the repo self-contained and gives you a lightweight way to compare runs over time.

## 10. A Good First Study Plan

1. Run CPU once.
2. Run one GPU.
3. Run two GPUs.
4. Compare throughput.
5. Record an `nsys` trace.
6. Record an `ncu` report.
7. Change one knob.
8. Re-run and compare.

If you follow that loop a few times, you will start building the intuition behind the resume bullet instead of just memorizing the words.

