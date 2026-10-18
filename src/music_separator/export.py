"""저장(내보내기) 계산 — 음질 손상을 최소화하는 방식 (GUI 와 독립, 테스트 가능).

원칙
1. 샘플레이트
   - AI 모델이 직접 만든 트랙(보컬 등): 모델 출력 샘플레이트(44.1kHz) 그대로 저장 (업샘플링 안 함)
   - MR(원본 − 보컬)과 마지막(잔여) 트랙: 원본 샘플레이트에서 계산 → 남는 악기는 리샘플링 없이 원본 그대로.
     원본이 48kHz 면 보컬 추정치만 한 번(soxr VHQ) 변환해 뺍니다.
2. MR = 원본 − 뺄 트랙 (볼륨을 바꾼 트랙이 있으면 그 차이만 더함: 원본 − Σ제외 + Σ(g−1)·남김)
3. 키·템포 조절은 선택한 결과물에만, 저장 직전에 한 번만 (둘을 함께) 적용 (키 0 · 템포 100% 면 건너뜀).
   보컬 트랙은 포먼트 유지 옵션 적용 가능. 드럼은 키를 바꾸지 않고 템포만 맞춤.
4. 피크 처리: 리미터·컴프레서·노멀라이즈는 쓰지 않음. 정수 형식이 0dBFS 를 넘을 때만
   모든 파일에 같은 고정 배율로 음량을 낮춤 (파형 모양·트랙 간 균형 그대로).
   - WAV/FLAC: 샘플 피크 ≤ -0.1dBFS
   - MP3: 4배 오버샘플링 트루 피크 ≤ -1.0dBTP (인코딩 후 디코딩해 다시 확인)
   - WAV 32-bit float: 0dBFS 를 넘어도 잘리지 않으므로 그대로 보존
5. 손실 압축(MP3)은 마지막 단계에서 단 한 번만. 중간 결과는 모두 float32 무손실.
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
PCM_CEILING_DB = -0.1
MP3_TRUE_PEAK_DB = -1.0


class ExportCancelled(Exception):
    pass


@dataclass
class ExportSource:
    """분리 결과 한 벌 (모든 트랙 + 원본)."""
    stems: dict                    # key -> 44.1kHz npy 경로 (분리 결과 순서 유지)
    residual_key: str              # '원본 − 나머지'로 계산되는 트랙
    mix44_path: Path               # 44.1kHz 원본 (AI 입력)
    native_path: Path              # 원본 샘플레이트 원본 (44.1kHz 원본이면 mix44_path 와 같음)
    native_rate: int = SAMPLE_RATE
    model_rate: int = SAMPLE_RATE  # AI 출력 트랙의 샘플레이트


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
    tempo: float = 1.0             # 빠르기 배율 (0.9 = 10% 느리게, 길이 1/0.9 배)
    keep_drums_pitch: bool = True
    formant_vocals: bool = True    # 보컬 트랙 키 변경 시 포먼트(음색) 유지


@dataclass
class FileReport:
    path: Path
    fmt: str
    sample_rate: int
    seconds: float
    channels: int
    peak_db: float                 # 저장된 소리의 샘플 피크 (dBFS)
    true_peak_db: Optional[float]  # MP3 만 측정
    resampled_from: Optional[int]  # 이 파일을 만들 때 리샘플링한 원래 레이트 (없으면 None)
    semitones: int = 0
    formant: bool = False
    verified: str = ""             # 저장 후 다시 읽어 확인한 형식 요약


@dataclass
class ExportReport:
    files: list
    sample_rate: int
    gain_db: float = 0.0           # 클리핑 방지로 낮춘 음량 (0 이면 조정 없음)
    details: list = field(default_factory=list)   # FileReport 목록
    key_engine: str = ""


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


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

    def target_rate(self, job: ExportJob) -> int:
        """이 결과물을 저장할 샘플레이트."""
        model_output = job.kind == "stem" and job.key != self.src.residual_key
        if model_output:
            rate = self.src.model_rate                 # AI 출력 그대로 (업샘플링 안 함)
        else:
            rate = self.src.native_rate if self.opts.keep_native_rate else self.src.model_rate
        if job.fmt == "mp3" and rate not in MP3_RATES:
            rate = 48000 if rate > 48000 else (44100 if rate > 32000 else 32000)
        return rate

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
            arr = audio_io.resample(data, self.src.model_rate, rate)
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

    def _resampled_from(self, job: ExportJob, rate: int) -> Optional[int]:
        if job.kind == "stem" and job.key != self.src.residual_key:
            return self.src.model_rate if rate != self.src.model_rate else None
        if rate != self.src.native_rate:
            return self.src.native_rate
        return None

    # --------------------------------------------------------------- 계산 ---
    def formant_for(self, job: ExportJob) -> bool:
        return bool(self.opts.formant_vocals and job.kind == "stem" and job.key.startswith("vocals"))

    def compute(self, job: ExportJob, rate: int) -> np.ndarray:
        semis = self.opts.semitones
        tempo = float(self.opts.tempo)
        keep_drums = self.opts.keep_drums_pitch
        if job.kind == "stem":
            data = np.array(self.stem(job.key, rate), dtype=np.float32)
            s = 0 if (job.key == "drums" and keep_drums) else semis
            if s or tempo != 1.0:
                data = pitch.shift(data, s, rate, formant=self.formant_for(job), cancel=self.cancel, tempo=tempo)
            return data
        # MR: 원본 − 제외 트랙 + 볼륨 차이
        data = np.array(self.original(rate), dtype=np.float32)
        for k in job.excluded:
            data -= self.stem(k, rate)
        for k, g in job.gains.items():
            if g != 1.0 and k not in job.excluded:
                data += (np.float32(g) - 1) * np.asarray(self.stem(k, rate), dtype=np.float32)
        if semis or tempo != 1.0:
            drums = None
            if semis and keep_drums and "drums" in self.src.stems and "drums" not in job.excluded:
                drums = np.float32(job.gains.get("drums", 1.0)) * np.asarray(self.stem("drums", rate),
                                                                           dtype=np.float32)
                data -= drums
            data = pitch.shift(data, semis, rate, formant=False, cancel=self.cancel, tempo=tempo)
            if drums is not None:
                data += pitch.shift(drums, 0, rate, cancel=self.cancel, tempo=tempo)
        return data

    def run(self, jobs: list[ExportJob]) -> ExportReport:
        rendered: list = []
        try:
            n = len(jobs)
            need_gain = 1.0
            # 1단계: 계산 (리샘플/키 조절) → 임시 파일, 피크 측정
            for i, job in enumerate(jobs):
                self._check()
                self.progress(i / n * 0.6, f"계산 중 ({i + 1}/{n}) {job.title}")
                rate = self.target_rate(job)
                data = self.compute(job, rate)
                stored = self._store(data, f"out{i}")
                del data
                self._check()
                if job.fmt in audio_io.FLOAT_FORMATS:
                    peak, ceiling = 0.0, 1.0
                elif job.fmt == "mp3":
                    self.progress(i / n * 0.6 + 0.3 / n, f"피크 검사 중 ({i + 1}/{n})")
                    peak, ceiling = audio_io.true_peak(stored, rate), 10 ** (MP3_TRUE_PEAK_DB / 20)
                else:
                    peak, ceiling = float(np.max(np.abs(stored))) if len(stored) else 0.0, 10 ** (PCM_CEILING_DB / 20)
                if peak > ceiling:
                    need_gain = min(need_gain, ceiling / peak)
                rendered.append((job, rate, stored))
            gain = need_gain
            # 2단계: 같은 배율로 음량 조정 후 단 한 번 인코딩 → 다시 읽어 확인
            written, details = [], []
            for i, (job, rate, data) in enumerate(rendered):
                self._check()
                g = gain if job.fmt not in audio_io.FLOAT_FORMATS else 1.0
                job.path.parent.mkdir(parents=True, exist_ok=True)

                def prog(f, i=i, job=job):
                    self.progress(0.6 + (i + f) / n * 0.4, f"저장 중 ({i + 1}/{n}) {job.title}")
                for attempt in range(3):
                    audio_io.write_audio(job.path, _Scaled(data, g) if g != 1.0 else data, job.fmt, rate,
                                         progress=prog)
                    if job.fmt != "mp3":
                        break
                    dec = audio_io.decode_audio(job.path, None)
                    mp3_peak = float(np.max(np.abs(dec))) if len(dec) else 0.0
                    del dec
                    if mp3_peak <= 1.0:
                        break
                    # 드물게 인코딩 후 피크가 0dBFS 를 넘으면 그만큼 더 낮춰 다시 한 번 인코딩
                    g *= 10 ** (MP3_TRUE_PEAK_DB / 20) / mp3_peak
                    gain = min(gain, g)
                info = audio_io.probe_audio(job.path)
                if info.sample_rate and info.sample_rate != rate:
                    raise audio_io.AudioError(f"저장된 파일의 샘플레이트가 다릅니다: {job.path.name}")
                peak_lin = float(np.max(np.abs(data))) * g if len(data) else 0.0
                details.append(FileReport(
                    job.path, job.fmt, rate, len(data) / rate, data.shape[1], _db(peak_lin),
                    _db(audio_io.true_peak(_Scaled(data, g), rate)) if job.fmt == "mp3" else None,
                    self._resampled_from(job, rate), self.opts.semitones,
                    self.formant_for(job) and bool(self.opts.semitones), info.summary()))
                written.append(job.path)
            self.progress(1.0, "저장 완료")
            gain_db = _db(gain) if gain < 1.0 else 0.0
            return ExportReport(written, rendered[0][1] if rendered else SAMPLE_RATE, float(gain_db), details,
                                pitch.engine_name() if self.opts.semitones or self.opts.tempo != 1.0 else "")
        finally:
            import gc

            rendered.clear()          # memmap 을 닫아야 Windows 에서 임시 파일 삭제 가능
            self._cache.clear()
            gc.collect()
            shutil.rmtree(self.work, ignore_errors=True)


def target_rate(src: ExportSource, fmt: str, opts: ExportOptions) -> int:
    """(이전 버전 호환) MR/잔여 트랙 기준 저장 샘플레이트."""
    rate = src.native_rate if opts.keep_native_rate else src.model_rate
    if fmt == "mp3" and rate not in MP3_RATES:
        rate = 48000 if rate > 48000 else SAMPLE_RATE
    return rate
