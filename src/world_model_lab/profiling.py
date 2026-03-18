from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def nvtx_range(name: str, enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class TorchProfilerController:
    def __init__(
        self,
        enabled: bool,
        trace_dir: str,
        use_cuda: bool,
        worker_name: str,
    ) -> None:
        self.enabled = enabled
        self.profiler: torch.profiler.profile | None = None
        if not enabled:
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, worker_name=worker_name),
            profile_memory=True,
            record_shapes=True,
        )

    def __enter__(self) -> "TorchProfilerController":
        if self.profiler is not None:
            self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc, tb)

    def step(self) -> None:
        if self.profiler is not None:
            self.profiler.step()

