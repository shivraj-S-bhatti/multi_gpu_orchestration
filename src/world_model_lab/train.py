from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import ExperimentConfig, load_config
from .data import SyntheticWorldModelDataset
from .distributed import cleanup_distributed, init_distributed, reduce_mean, seed_everything
from .model import WorldModel
from .profiling import TorchProfilerController, nvtx_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy multimodal world model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a TOML config.")
    parser.add_argument("--device", type=str, default=None, help="Override config device.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training steps.")
    parser.add_argument("--log-every", type=int, default=None, help="Override logging frequency.")
    parser.add_argument("--metrics-path", type=str, default=None, help="Override metrics output path.")
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Capture a PyTorch profiler trace in addition to NVTX ranges.",
    )
    parser.add_argument(
        "--emit-nvtx",
        action="store_true",
        help="Emit NVTX ranges and delimit training steps for Nsight tools.",
    )
    return parser.parse_args()


def maybe_override_config(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.device is not None:
        config.system.device = args.device
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.log_every is not None:
        config.train.log_every = args.log_every
    if args.metrics_path is not None:
        config.train.metrics_path = args.metrics_path
    return config


def configure_backends(config: ExperimentConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = config.system.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.system.allow_tf32
    torch.backends.cudnn.benchmark = config.system.cudnn_benchmark


def build_dataloader(config: ExperimentConfig, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = SyntheticWorldModelDataset(
        dataset_size=config.data.dataset_size,
        seq_len=config.data.seq_len,
        image_size=config.data.image_size,
        proprio_dim=config.data.proprio_dim,
        action_dim=config.data.action_dim,
        seed=config.system.seed,
    )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    dataloader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.data.batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "drop_last": True,
    }
    if config.data.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.data.persistent_workers
        dataloader_kwargs["prefetch_factor"] = config.data.prefetch_factor

    return DataLoader(**dataloader_kwargs), sampler


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True)
    return moved


def compute_losses(
    outputs: dict[str, torch.Tensor],
    target_frames: torch.Tensor,
    target_proprio: torch.Tensor,
    target_rewards: torch.Tensor,
    target_dones: torch.Tensor,
) -> dict[str, torch.Tensor]:
    frame_loss = F.mse_loss(outputs["next_frames"], target_frames)
    proprio_loss = F.mse_loss(outputs["next_proprio"], target_proprio)
    reward_loss = F.mse_loss(outputs["reward"], target_rewards)
    done_loss = F.binary_cross_entropy_with_logits(outputs["done_logits"], target_dones)
    total_loss = frame_loss + 0.5 * proprio_loss + 0.1 * reward_loss + 0.1 * done_loss
    return {
        "total": total_loss,
        "frame": frame_loss,
        "proprio": proprio_loss,
        "reward": reward_loss,
        "done": done_loss,
    }


def autocast_context(config: ExperimentConfig, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if config.train.amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if config.train.amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(config: ExperimentConfig, device: torch.device) -> torch.cuda.amp.GradScaler:
    enabled = device.type == "cuda" and config.train.amp_dtype == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    step: int,
) -> None:
    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {
            "step": step,
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )


def append_metrics(config: ExperimentConfig, payload: dict[str, float | int]) -> None:
    metrics_path = Path(config.train.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def train(config: ExperimentConfig, emit_nvtx: bool, torch_profile: bool) -> None:
    context = init_distributed(config.system.device)
    try:
        seed_everything(config.system.seed, context.rank)
        configure_backends(config, context.device)

        dataloader, sampler = build_dataloader(config, context.rank, context.world_size)
        data_iter = iter(dataloader)

        model = WorldModel(
            config.data,
            config.model,
            use_channels_last=config.system.channels_last and context.device.type == "cuda",
        ).to(context.device)
        if config.system.channels_last and context.device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        if config.system.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)
        if context.is_distributed:
            model = DDP(model, device_ids=[context.local_rank] if context.device.type == "cuda" else None)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        scaler = make_grad_scaler(config, context.device)

        if context.is_main:
            Path(config.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(config.train.metrics_path).parent.mkdir(parents=True, exist_ok=True)

        with TorchProfilerController(
            enabled=torch_profile and context.is_main,
            trace_dir=config.train.torch_profile_dir,
            use_cuda=context.device.type == "cuda",
            worker_name=f"rank{context.rank}",
        ) as profiler:
            for step in range(1, config.train.max_steps + 1):
                if sampler is not None and step == 1:
                    sampler.set_epoch(0)

                try:
                    batch = next(data_iter)
                except StopIteration:
                    if sampler is not None:
                        sampler.set_epoch(step)
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                optimizer.zero_grad(set_to_none=True)
                step_start = time.perf_counter()

                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                with nvtx_range("host_to_device", emit_nvtx):
                    transfer_start = time.perf_counter()
                    batch = move_batch(batch, context.device)
                    if context.device.type == "cuda":
                        torch.cuda.synchronize(context.device)
                    h2d_time = time.perf_counter() - transfer_start

                input_frames = batch["frames"][:, :-1]
                target_frames = batch["frames"][:, 1:]
                input_proprio = batch["proprio"][:, :-1]
                target_proprio = batch["proprio"][:, 1:]
                input_actions = batch["actions"][:, :-1]
                target_rewards = batch["rewards"][:, 1:]
                target_dones = batch["dones"][:, 1:]

                with nvtx_range("step", emit_nvtx):
                    with nvtx_range("forward", emit_nvtx):
                        with autocast_context(config, context.device):
                            outputs = model(input_frames, input_proprio, input_actions)
                    with nvtx_range("loss", emit_nvtx):
                        losses = compute_losses(outputs, target_frames, target_proprio, target_rewards, target_dones)

                    with nvtx_range("backward", emit_nvtx):
                        scaler.scale(losses["total"]).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)

                    with nvtx_range("optimizer_step", emit_nvtx):
                        scaler.step(optimizer)
                        scaler.update()

                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                step_time = time.perf_counter() - step_start
                profiler.step()

                if step % config.train.log_every == 0 or step == 1:
                    reduced_total = reduce_mean(losses["total"], context.world_size)
                    reduced_frame = reduce_mean(losses["frame"], context.world_size)
                    global_frames = (
                        config.data.batch_size
                        * input_frames.shape[1]
                        * context.world_size
                    )
                    frames_per_second = global_frames / max(step_time, 1e-6)
                    if context.is_main:
                        record = {
                            "step": step,
                            "loss_total": float(reduced_total.item()),
                            "loss_frame": float(reduced_frame.item()),
                            "step_time_sec": step_time,
                            "h2d_time_sec": h2d_time,
                            "frames_per_second": frames_per_second,
                        }
                        append_metrics(config, record)
                        print(json.dumps(record), flush=True)

                if context.is_main and config.train.save_every > 0 and step % config.train.save_every == 0:
                    save_checkpoint(model, optimizer, config, step)
    finally:
        cleanup_distributed()


def main() -> None:
    args = parse_args()
    config = maybe_override_config(load_config(args.config), args)
    train(config, emit_nvtx=args.emit_nvtx, torch_profile=args.torch_profile)


if __name__ == "__main__":
    main()
