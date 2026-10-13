"""하드웨어(GPU) 감지와 메모리/디스크 요구량 검사."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .audio_io import CHANNELS, SAMPLE_RATE

log = logging.getLogger(__name__)

GB = 1024 ** 3
MB = 1024 ** 2


@dataclass
class DeviceInfo:
    key: str          # "cuda" | "mps" | "cpu"
    label: str        # 사용자 표시용
    memory_gb: float = 0.0


def detect_devices() -> list[DeviceInfo]:
    """사용 가능한 처리 장치 목록 (좋은 순서)."""
    import torch

    devices: list[DeviceInfo] = []
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem = props.total_memory / GB
            devices.append(DeviceInfo("cuda", f"NVIDIA {props.name} ({mem:.0f}GB)", mem))
    except Exception as exc:
        log.warning("CUDA 감지 실패: %s", exc)
    try:
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            devices.append(DeviceInfo("mps", "Apple Silicon GPU (Metal)"))
    except Exception as exc:
        log.warning("MPS 감지 실패: %s", exc)
    devices.append(DeviceInfo("cpu", f"CPU ({os.cpu_count() or 1}코어)"))
    return devices


def choose_device(preference: str) -> DeviceInfo:
    """preference: 'auto' | 'gpu' | 'cpu'."""
    devices = detect_devices()
    if preference == "cpu":
        return devices[-1]
    gpus = [d for d in devices if d.key != "cpu"]
    if gpus:
        return gpus[0]
    return devices[-1]


def stem_bytes(n_samples: int) -> int:
    return n_samples * CHANNELS * 4  # float32


@dataclass
class Requirement:
    ram_needed: int
    ram_available: int
    disk_needed: int
    disk_available: int
    disk_path: Path

    @property
    def ram_ok(self) -> bool:
        return self.ram_available <= 0 or self.ram_available >= self.ram_needed

    @property
    def disk_ok(self) -> bool:
        return self.disk_available >= self.disk_needed


def available_ram() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return -1


def free_disk(path: Path) -> int:
    path = Path(path)
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def separation_requirement(n_samples: int, n_sources: int, n_outputs: int, cache_dir: Path,
                           bag_size: int = 1) -> Requirement:
    """분리 작업 전 메모리/디스크 요구량 추정.

    RAM : 입력 + 모델 출력(소스 수) + bag 평균용 버퍼 + 모델/연산 여유(약 1.5GB)
    디스크 : 출력 트랙 수 × float32 크기 + 여유 200MB (캐시 폴더)
    """
    b = stem_bytes(n_samples)
    extra = 1 if bag_size > 1 else 0
    ram = b + (1 + extra) * n_sources * b + int(1.5 * GB)
    disk = n_outputs * b + 200 * MB
    return Requirement(ram, available_ram(), disk, free_disk(cache_dir), Path(cache_dir))


def human_bytes(n: int) -> str:
    if n >= GB:
        return f"{n / GB:.1f}GB"
    return f"{n / MB:.0f}MB"


def samples_for(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE)
