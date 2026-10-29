"""Portable CPU, RAM, and NVIDIA GPU resource monitoring for SALAZAR."""

import shutil
import subprocess
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time snapshot of available system resources."""

    cpu_count: int
    cpu_percent: float
    ram_total_gib: float
    ram_available_gib: float
    ram_used_percent: float
    gpu_available: bool
    gpu_name: str | None = None
    vram_total_mib: int | None = None
    vram_used_mib: int | None = None
    vram_free_mib: int | None = None
    gpu_utilization_percent: int | None = None


class ResourceManager:
    """Measure resources and make configurable RAM admission decisions."""

    def __init__(self, max_ram_used_percent: float = 90.0):
        if not 0 < max_ram_used_percent <= 100:
            raise ValueError(
                "max_ram_used_percent must be greater than 0 and at most 100."
            )

        self.max_ram_used_percent = float(max_ram_used_percent)

    @staticmethod
    def _query_gpu() -> dict | None:
        """Return NVIDIA telemetry, or None when unavailable."""
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return None

        command = [
            executable,
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            line = completed.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in line.split(",")]

            if len(parts) != 5:
                return None

            return {
                "name": parts[0],
                "vram_total_mib": int(parts[1]),
                "vram_used_mib": int(parts[2]),
                "vram_free_mib": int(parts[3]),
                "gpu_utilization_percent": int(parts[4]),
            }
        except (
            OSError,
            subprocess.SubprocessError,
            IndexError,
            ValueError,
        ):
            return None

    def snapshot(self) -> ResourceSnapshot:
        """Collect a best-effort snapshot of CPU, RAM, and GPU resources."""
        memory = psutil.virtual_memory()
        gpu = self._query_gpu()

        return ResourceSnapshot(
            cpu_count=psutil.cpu_count(logical=True) or 1,
            cpu_percent=psutil.cpu_percent(interval=0.1),
            ram_total_gib=round(memory.total / (1024 ** 3), 2),
            ram_available_gib=round(memory.available / (1024 ** 3), 2),
            ram_used_percent=memory.percent,
            gpu_available=gpu is not None,
            gpu_name=gpu["name"] if gpu else None,
            vram_total_mib=gpu["vram_total_mib"] if gpu else None,
            vram_used_mib=gpu["vram_used_mib"] if gpu else None,
            vram_free_mib=gpu["vram_free_mib"] if gpu else None,
            gpu_utilization_percent=(
                gpu["gpu_utilization_percent"] if gpu else None
            ),
        )

    def can_start_task(self, estimated_ram_gib: float = 0.0) -> bool:
        """Check RAM admission against a configured threshold.

        This is a best-effort check, not a reservation or a hard memory limit.
        """
        if estimated_ram_gib < 0:
            raise ValueError("estimated_ram_gib must not be negative.")

        memory = psutil.virtual_memory()
        projected_used_percent = (
            (memory.total - memory.available + estimated_ram_gib * (1024 ** 3))
            / memory.total
        ) * 100

        return projected_used_percent < self.max_ram_used_percent
