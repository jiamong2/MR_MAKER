"""저장(내보내기) 계산 — 음질 손상을 최소화하는 방식 (GUI 와 독립, 테스트 가능).

원칙
1. 원본 샘플레이트로 저장: AI 처리는 44.1kHz 에서 하지만, 원본이 48/96kHz 면
   분리된 트랙만 원래 레이트로 되돌리고 '원본'은 원래 레이트 사본을 그대로 사용.
2. MR = 원본 − 뺄 트랙: 남기는 악기는 AI 를 거치지 않은 원본 그대로.
   (볼륨을 바꾼 트랙이 있으면 그 차이만 더함: 원본 − Σ제외 + Σ(g−1)·남김)
3. 마지막(반주/기타 악기/나머지) 트랙도 원본 레이트에서 '원본 − 나머지 트랙'으로 다시 계산.
4. 키 조절은 저장 직전에 한 번만 적용 (드럼은 원래 음높이 유지 옵션).
5. 클리핑 방지: 정수 형식(WAV 16/24, FLAC, MP3)에서 0dBFS 를 넘으면 함께 저장하는 파일 전체에
   같은 비율로 음량을 낮춤 (트랙 간 균형 유지). 32-bit float WAV 는 그대로 보존.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio_io, pitch
from .audio_io import SAMPLE_RATE

log = logging.getLogger(__name__)

MP3_RATES = (32000, 44100, 48000)


class ExportCancelled(Exception):
    pass


@dataclass
class ExportSource:
    """분리 결과 한 벌 (모든 트랙 + 원본)."""
    stems: dict                    # key -> 44.1kHz npy 경로 (분리 결과 순서 유지)
    residual_key: str              # '원본 − 나머지'로 계산되는 트랙
    mix44_path: Path               # 44.1kHz 원본
    native_path: Path              # 원본 샘플레이트 원본 (44.1kHz 원본이면 mix44_path 와 같음)
    native_rate: int = SAMPLE_RATE


@dataclass
class ExportJob:
    path: Path
    fmt: str
    title: str
    kind: str = "stem"             # "stem" | "mr"
    key: str = ""                  # kind == "stem"
    excluded: list = field(default_factory=list)   # kind == "mr"
    gains: dict = field(default_factory=dict)      # kind == "mr": 남기는 트랙의 볼륨 (1.0 = 그대로)


@dataclass
class ExportOptions:
    keep_native_rate: bool = True
    semitones: int = 0
    keep_drums_pitch: bool = True


@dataclass
class ExportReport:
    files: list
    sample_rate: int
    gain_db: float = 0.0           # 클리핑 방지로 낮춘 음량 (0 이면 조정 없음)


def target_rate(src: ExportSource, fmt: str, opts: ExportOptions) -> int:
    rate = src.native_rate if opts.keep_native_rate else SAMPLE_RATE
    if fmt == "mp3" and rate not in MP3_RATES:
        rate = 48000 if rate > 48000 else SAMPLE_RATE
    return rate


class _Scaled:
    """memmap 을 블록 단위로 읽으면서 음량 배율을 적용 (전체를 메모리에 올리지 않음)."""

    def __init__(self, data: np.ndarray, gain: float):
        self.data, self.gain = data, np.float32(gain)
        self.shape = data.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.asarray(self.data[idx], dtype=np.float32) * self.gain


class Exporter:
    def __init__(self, src: ExportSource, opts: ExportOptions, work_dir: Path,
                 progress: Optional[Callable[[float, str], None]] = None,
                 cancel: Optional[threading.Event] = None):
        self.src, self.opts = src, opts
        self.work = Path(tempfile.mkdtemp(prefix="export-", dir=str(work_dir)))
        self.progress = progress or (lambda f, m: None)
        self.cancel = cancel or threading.Event()
        self._cache: dict = {}

    # ----------------------------------------------------------- 원본/트랙 ---
    def _check(self):
        if self.cancel.is_set():
            raise ExportCancelled()

    def original(self, rate: int) -> np.ndarray:
        key = ("orig", rate)
        if key not in self._cache:
            if rate == self.src.native_rate:
                arr = np.load(self.src.native_path, mmap_mode="r")
            elif rate == SAMPLE_RATE:
                arr = np.load(self.src.mix44_path, mmap_mode="r")
            else:
                arr = self._store(audio_io.resample(np.load(self.src.native_path, mmap_mode="r"),
                                                    self.src.native_rate, rate), f"orig{rate}")
            self._cache[key] = arr
        return self._cache[key]

    def stem(self, k: str, rate: int) -> np.ndarray:
        """원하는 샘플레이트의 트랙. 마지막(잔여) 트랙은 그 레이트의 원본에서 다시 계산."""
        key = ("stem", k, rate)
        if key in self._cache:
            return self._cache[key]
        self._check()
        if k == self.src.residual_key:
            arr = np.array(self.original(rate), dtype=np.float32)
            for other in self.src.stems:
                if other != k:
                    arr -= self._len(self.stem(other, rate), len(arr))
        else:
            data = np.load(self.src.stems[k], mmap_mode="r")
            arr = audio_io.resample(data, SAMPLE_RATE, rate)
        arr = self._store(self._len(arr, len(self.original(rate))), f"{k}{rate}")
        self._cache[key] = arr
        return arr

    @staticmethod
    def _len(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) == n:
            return arr
        if len(arr) > n:
            return arr[:n]
        return np.pad(np.asarray(arr), ((0, n - len(arr)), (0, 0)))

    def _store(self, arr: np.ndarray, name: str) -> np.ndarray:
        p = self.work / f"{name}.npy"
        np.save(p, np.asarray(arr, dtype=np.float32))
        return np.load(p, mmap_mode="r")

    # --------------------------------------------------------------- 계산 ---
    def compute(self, job: ExportJob, rate: int) -> np.ndarray:
        semis = self.opts.semitones
        keep_drums = self.opts.keep_drums_pitch
        if job.kind == "stem":
            data = np.array(self.stem(job.key, rate), dtype=np.float32)
            if semis and not (job.key == "drums" and keep_drums):
                data = pitch.shift(data, semis, rate)
            return data
        # MR: 원본 − 제외 트랙 + 볼륨 차이
        data = np.array(self.original(rate), dtype=np.float32)
        for k in job.excluded:
            data -= self.stem(k, rate)
        for k, g in job.gains.items():
            if g != 1.0 and k not in job.excluded:
                data += (np.float32(g) - 1) * np.asarray(self.stem(k, rate), dtype=np.float32)
        if semis:
            drums = None
            if keep_drums and "drums" in self.src.stems and "drums" not in job.excluded:
                drums = np.float32(job.gains.get("drums", 1.0)) * np.asarray(self.stem("drums", rate),
                                                                           dtype=np.float32)
                data -= drums
            data = pitch.shift(data, semis, rate)
            if drums is not None:
                data += drums
        return data

    def run(self, jobs: list[ExportJob]) -> ExportReport:
        rendered: list = []
        try:
            n = len(jobs)
            peak = 0.0
            # 1단계: 계산 (리샘플/키 조절) → 임시 파일, 최대 음량 측정
            for i, job in enumerate(jobs):
                self._check()
                self.progress(i / n * 0.7, f"계산 중 ({i + 1}/{n}) {job.title}")
                rate = target_rate(self.src, job.fmt, self.opts)
                data = self.compute(job, rate)
                peak = max(peak, float(np.max(np.abs(data))) if len(data) else 0.0)
                rendered.append((job, rate, self._store(data, f"out{i}")))
                del data
            # 2단계: 클리핑 방지 음량 조정 후 인코딩
            int_fmt = any(j.fmt not in audio_io.FLOAT_FORMATS for j in jobs)
            gain = 1.0
            limit = 10 ** (-0.1 / 20)   # -0.1 dBFS
            if int_fmt and peak > limit:
                gain = limit / peak
            written = []
            for i, (job, rate, data) in enumerate(rendered):
                self._check()
                g = gain if job.fmt not in audio_io.FLOAT_FORMATS else 1.0
                job.path.parent.mkdir(parents=True, exist_ok=True)

                def prog(f, i=i, job=job):
                    self.progress(0.7 + (i + f) / n * 0.3, f"저장 중 ({i + 1}/{n}) {job.title}")
                audio_io.write_audio(job.path, _Scaled(data, g) if g != 1.0 else data, job.fmt, rate,
                                     progress=prog)
                written.append(job.path)
            self.progress(1.0, "저장 완료")
            gain_db = 20 * np.log10(gain) if gain < 1.0 else 0.0
            return ExportReport(written, rendered[0][1] if rendered else SAMPLE_RATE, float(gain_db))
        finally:
            import gc

            rendered.clear()          # memmap 을 닫아야 Windows 에서 임시 파일 삭제 가능
            self._cache.clear()
            gc.collect()
            shutil.rmtree(self.work, ignore_errors=True)
