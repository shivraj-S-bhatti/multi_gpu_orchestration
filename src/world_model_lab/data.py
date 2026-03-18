from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


class SyntheticWorldModelDataset(Dataset[dict[str, torch.Tensor]]):
    """Generates long-horizon multimodal sequences on the fly."""

    def __init__(
        self,
        dataset_size: int,
        seq_len: int,
        image_size: int,
        proprio_dim: int,
        action_dim: int,
        seed: int = 7,
    ) -> None:
        self.dataset_size = dataset_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.seed = seed

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)

        frames = torch.zeros(self.seq_len, 3, self.image_size, self.image_size, dtype=torch.float32)
        proprio = torch.zeros(self.seq_len, self.proprio_dim, dtype=torch.float32)
        actions = torch.zeros(self.seq_len, self.action_dim, dtype=torch.float32)
        rewards = torch.zeros(self.seq_len, 1, dtype=torch.float32)
        dones = torch.zeros(self.seq_len, 1, dtype=torch.float32)

        position = 0.2 + 0.6 * torch.rand(2, generator=generator)
        velocity = 0.04 * torch.randn(2, generator=generator)
        goal = 0.2 + 0.6 * torch.rand(2, generator=generator)
        phase = 2.0 * math.pi * torch.rand(1, generator=generator).item()

        for step in range(self.seq_len):
            delta = goal - position
            distance = delta.norm().clamp_min(1e-6)
            action_xy = 0.45 * (delta / distance) + 0.08 * torch.randn(2, generator=generator)
            action_xy = action_xy.clamp(-1.0, 1.0)

            velocity = 0.88 * velocity + 0.10 * action_xy
            velocity = velocity.clamp(-0.08, 0.08)
            next_position = position + velocity

            bounced = torch.zeros(2, dtype=torch.float32)
            for axis in range(2):
                if next_position[axis] < 0.05:
                    next_position[axis] = 0.05
                    velocity[axis] = velocity[axis].abs()
                    bounced[axis] = 1.0
                elif next_position[axis] > 0.95:
                    next_position[axis] = 0.95
                    velocity[axis] = -velocity[axis].abs()
                    bounced[axis] = 1.0

            reward = -distance
            done = float(distance < 0.08)

            action_features = torch.tensor(
                [
                    action_xy[0].item(),
                    action_xy[1].item(),
                    distance.item(),
                    bounced.sum().item(),
                ],
                dtype=torch.float32,
            )
            proprio_features = torch.tensor(
                [
                    position[0].item(),
                    position[1].item(),
                    velocity[0].item(),
                    velocity[1].item(),
                    goal[0].item(),
                    goal[1].item(),
                    math.sin(phase),
                    math.cos(phase),
                ],
                dtype=torch.float32,
            )

            actions[step, : min(self.action_dim, action_features.numel())] = action_features[: self.action_dim]
            proprio[step, : min(self.proprio_dim, proprio_features.numel())] = proprio_features[: self.proprio_dim]
            rewards[step, 0] = reward
            dones[step, 0] = done
            frames[step] = self._render_frame(position, goal, velocity, phase)

            position = next_position
            phase += 0.17

            if done:
                goal = 0.1 + 0.8 * torch.rand(2, generator=generator)

        return {
            "frames": frames,
            "proprio": proprio,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def _render_frame(
        self,
        position: torch.Tensor,
        goal: torch.Tensor,
        velocity: torch.Tensor,
        phase: float,
    ) -> torch.Tensor:
        frame = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        agent_color = torch.tensor(
            [
                0.70 + 0.15 * math.sin(phase),
                0.25 + 0.10 * math.cos(phase),
                0.35 + 0.20 * velocity.norm().clamp(max=1.0).item(),
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        goal_color = torch.tensor([0.20, 0.85, 0.35], dtype=torch.float32)

        self._draw_square(frame, position, agent_color, width=3)
        self._draw_square(frame, goal, goal_color, width=2)
        return frame

    def _draw_square(
        self,
        frame: torch.Tensor,
        xy: torch.Tensor,
        color: torch.Tensor,
        width: int,
    ) -> None:
        center_x = int(xy[0].item() * (self.image_size - 1))
        center_y = int(xy[1].item() * (self.image_size - 1))

        x0 = max(0, center_x - width)
        x1 = min(self.image_size, center_x + width + 1)
        y0 = max(0, center_y - width)
        y1 = min(self.image_size, center_y + width + 1)
        frame[:, y0:y1, x0:x1] = color[:, None, None]

