from __future__ import annotations

import torch
from torch import nn

from .config import DataConfig, ModelConfig


class FrameEncoder(nn.Module):
    def __init__(self, image_size: int, latent_dim: int) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")

        final_resolution = image_size // 8
        hidden_dim = 128 * final_resolution * final_resolution

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.conv(frames)
        return self.proj(features.flatten(start_dim=1))


class FrameDecoder(nn.Module):
    def __init__(self, image_size: int, latent_dim: int) -> None:
        super().__init__()
        final_resolution = image_size // 8
        hidden_dim = 128 * final_resolution * final_resolution
        self.final_resolution = final_resolution

        self.proj = nn.Linear(latent_dim, hidden_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[0]
        features = self.proj(hidden).view(batch, 128, self.final_resolution, self.final_resolution)
        return self.deconv(features)


class StateActionEncoder(nn.Module):
    def __init__(self, proprio_dim: int, action_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim + action_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, proprio: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([proprio, actions], dim=-1))


class WorldModel(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        use_channels_last: bool = False,
    ) -> None:
        super().__init__()
        self.use_channels_last = use_channels_last
        self.frame_encoder = FrameEncoder(data_config.image_size, model_config.latent_dim)
        self.frame_decoder = FrameDecoder(data_config.image_size, model_config.latent_dim)
        self.state_action_encoder = StateActionEncoder(
            proprio_dim=data_config.proprio_dim,
            action_dim=data_config.action_dim,
            latent_dim=model_config.latent_dim,
        )

        self.input_proj = nn.Linear(model_config.latent_dim * 2, model_config.model_dim)
        self.output_proj = nn.Linear(model_config.model_dim, model_config.latent_dim)
        self.position_embedding = nn.Embedding(data_config.seq_len, model_config.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.model_dim,
            nhead=model_config.num_heads,
            dim_feedforward=int(model_config.model_dim * model_config.mlp_ratio),
            dropout=model_config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.num_layers,
            enable_nested_tensor=False,
        )
        self.state_head = nn.Sequential(
            nn.Linear(model_config.latent_dim, model_config.latent_dim),
            nn.GELU(),
            nn.Linear(model_config.latent_dim, data_config.proprio_dim),
        )
        self.reward_head = nn.Linear(model_config.latent_dim, 1)
        self.done_head = nn.Linear(model_config.latent_dim, 1)

    def forward(
        self,
        frames: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = frames.shape[:2]
        flat_frames = frames.reshape(batch_size * seq_len, *frames.shape[2:])
        if self.use_channels_last:
            flat_frames = flat_frames.contiguous(memory_format=torch.channels_last)
        image_latents = self.frame_encoder(flat_frames).view(batch_size, seq_len, -1)
        sa_latents = self.state_action_encoder(proprio, actions)

        tokens = self.input_proj(torch.cat([image_latents, sa_latents], dim=-1))
        positions = torch.arange(seq_len, device=frames.device)
        tokens = tokens + self.position_embedding(positions)[None, :, :]

        mask = self._causal_mask(seq_len, frames.device)
        hidden = self.transformer(tokens, mask=mask)
        prediction_latents = self.output_proj(hidden)

        next_frames = self.frame_decoder(prediction_latents.reshape(batch_size * seq_len, -1))
        next_frames = next_frames.view(batch_size, seq_len, *frames.shape[2:])

        return {
            "next_frames": next_frames,
            "next_proprio": self.state_head(prediction_latents),
            "reward": self.reward_head(prediction_latents),
            "done_logits": self.done_head(prediction_latents),
        }

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)
