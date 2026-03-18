from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(slots=True)
class SystemConfig:
    seed: int = 7
    device: str = "auto"
    compile_model: bool = False
    channels_last: bool = False
    allow_tf32: bool = True
    cudnn_benchmark: bool = False


@dataclass(slots=True)
class DataConfig:
    dataset_size: int = 4096
    seq_len: int = 33
    image_size: int = 32
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    action_dim: int = 4
    proprio_dim: int = 8


@dataclass(slots=True)
class ModelConfig:
    latent_dim: int = 128
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    mlp_ratio: float = 4.0


@dataclass(slots=True)
class TrainConfig:
    max_steps: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    amp_dtype: str = "none"
    log_every: int = 10
    save_every: int = 50
    checkpoint_dir: str = "artifacts/checkpoints"
    metrics_path: str = "artifacts/metrics/train_metrics.jsonl"
    torch_profile_dir: str = "artifacts/profiles/torch"


@dataclass(slots=True)
class ExperimentConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    if not is_dataclass(instance):
        return updates

    merged: dict[str, Any] = {}
    for field in fields(instance):
        current_value = getattr(instance, field.name)
        incoming_value = updates.get(field.name, current_value)
        if is_dataclass(current_value) and isinstance(incoming_value, dict):
            merged[field.name] = _merge_dataclass(current_value, incoming_value)
        else:
            merged[field.name] = incoming_value
    return type(instance)(**merged)


def load_config(path: str | Path | None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config

    config_path = Path(path)
    with config_path.open("rb") as handle:
        updates = tomllib.load(handle)
    return _merge_dataclass(config, updates)
