"""오디오 입출력.

- 디코딩: FFmpeg(번들된 imageio-ffmpeg 바이너리 → 시스템 PATH 순으로 탐색)로 스테레오 float32 변환
  (AI 처리용 44.1kHz 와, 원본 샘플레이트 그대로의 사본)
- 리샘플링: julius (고품질 sinc, Demucs 의존성)
- WAV 저장: 16/24-bit PCM, 32-bit float (직접 헤더 작성)
- FLAC 저장: FFmpeg 파이프 (24-bit)
- MP3 저장: lameenc (LAME 인코더)
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
CHANNELS = 2

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".aif", ".aiff",
    ".wma", ".alac", ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi",
}

# 키: (표시 이름, 확장자, 무손실 여부, 정수 형식이라 0dBFS 초과 시 잘림)
EXPORT_FORMATS = {
    "wav24": ("WAV 24-bit (권장)", ".wav"),
    "wav32f": ("WAV 32-bit float", ".wav"),
    "flac": ("FLAC 24-bit (무손실 압축)", ".flac"),
    "wav16": ("WAV 16-bit", ".wav"),
    "mp3": ("MP3 320kbps", ".mp3"),
}
FLOAT_FORMATS = {"wav32f"}           # 0dBFS 를 넘어도 잘리지 않는 형식
DEFAULT_FORMAT = "wav24"


class AudioError(Exception):
    """사용자에게 보여줄 수 있는 오디오 관련 오류."""


_ffmpeg_cache: Optional[str] = None


def find_ffmpeg() -> str:
    global _ffmpeg_cache
    if _ffmpeg_cache and os.path.exists(_ffmpeg_cache):
        return _ffmpeg_cache
    candidates = []
    env = os.environ.get("MUSIC_SEPARATOR_FFMPEG")
    if env:
        candidates.append(env)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # pragma: no cover - 번들 누락 시
        log.warning("imageio-ffmpeg 를 찾지 못함: %s", exc)
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)
    for c in candidates:
        if c and os.path.exists(c):
            _ffmpeg_cache = c
            return c
    raise AudioError(
        "FFmpeg 를 찾을 수 없습니다. 프로그램을 다시 설치해 주세요.\n"
        "(소스 실행 시: pip install imageio-ffmpeg)"
    )


def _popen_kwargs() -> dict:
    kw: dict = {"stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return kw


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


_RATE_RE = re.compile(r"Audio:.*?(\d{4,6}) Hz")


def _probe_text(path: Path) -> str:
    try:
        proc = subprocess.run(
            [find_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, **_popen_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("probe failed: %s", exc)
        return ""
    return proc.stderr.decode("utf-8", "replace")


def probe_duration(path: Path) -> Optional[float]:
    """파일 길이(초). 알 수 없으면 None."""
    m = _DURATION_RE.search(_probe_text(path))
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def probe_sample_rate(path: Path) -> Optional[int]:
    """첫 오디오 스트림의 샘플레이트. 알 수 없으면 None."""
    m = _RATE_RE.search(_probe_text(path))
    return int(m.group(1)) if m else None


def resample(data: np.ndarray, src: int, dst: int) -> np.ndarray:
    """(n, ch) float32 고품질 리샘플링 (julius sinc). 같은 레이트면 그대로 반환."""
    if src == dst:
        return np.asarray(data, dtype=np.float32)
    import torch
    from julius import resample_frac

    x = torch.from_numpy(np.ascontiguousarray(np.asarray(data, dtype=np.float32).T))
    with torch.no_grad():
        y = resample_frac(x, src, dst)
    n_out = int(round(len(data) * dst / src))
    y = y[:, :n_out]
    if y.shape[1] < n_out:
        y = torch.nn.functional.pad(y, (0, n_out - y.shape[1]))
    return np.ascontiguousarray(y.T.numpy(), dtype=np.float32)


def decode_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """오디오 파일을 (samples, 2) float32 배열로 디코딩."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"파일을 찾을 수 없습니다: {path}")
    cmd = [
        find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(path), "-vn", "-map", "0:a:0",
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(CHANNELS), "-ar", str(sample_rate),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_kwargs())
    except OSError as exc:
        raise AudioError(f"FFmpeg 실행 실패: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = err[-1] if err else "알 수 없는 오류"
        raise AudioError(
            "오디오를 읽을 수 없습니다. 지원되지 않거나 손상된 파일일 수 있습니다.\n" f"({detail})"
        )
    data = np.frombuffer(proc.stdout, dtype="<f4")
    data = data[: (len(data) // CHANNELS) * CHANNELS].reshape(-1, CHANNELS)
    if len(data) < sample_rate // 10:
        raise AudioError("오디오가 너무 짧습니다 (0.1초 미만).")
    return np.ascontiguousarray(data, dtype=np.float32)


# ---------------------------------------------------------------- export ---

def _to_int(block: np.ndarray, bits: int) -> np.ndarray:
    block = np.clip(block, -1.0, 1.0)
    if bits == 16:
        return (block * 32767.0).round().astype("<i2")
    return (block * 8388607.0).round().astype("<i4")


def write_wav(path: Path, data: np.ndarray, bits: int = 16, sample_rate: int = SAMPLE_RATE,
              progress: Optional[Callable[[float], None]] = None, block: int = SAMPLE_RATE * 10) -> None:
    """(samples, channels) float 배열을 PCM WAV 로 저장. memmap 도 블록 단위로 처리."""
    if bits not in (16, 24):
        raise ValueError("bits must be 16 or 24")
    n = len(data)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(data.shape[1])
        wf.setsampwidth(bits // 8)
        wf.setframerate(sample_rate)
        for start in range(0, n, block):
            chunk = np.asarray(data[start:start + block], dtype=np.float32)
            ints = _to_int(chunk, bits)
            if bits == 24:
                raw = ints.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
            else:
                raw = ints.tobytes()
            wf.writeframes(raw)
            if progress:
                progress(min(1.0, (start + block) / max(1, n)))


_has_lame_cache: Optional[bool] = None


def ffmpeg_has_lame() -> bool:
    global _has_lame_cache
    if _has_lame_cache is None:
        try:
            out = subprocess.run([find_ffmpeg(), "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, timeout=30, **_popen_kwargs()).stdout
            _has_lame_cache = b"libmp3lame" in out
        except Exception:
            _has_lame_cache = False
    return _has_lame_cache


def _ffmpeg_encode(path: Path, data: np.ndarray, sample_rate: int, codec_args: list,
                   progress: Optional[Callable[[float], None]], block: int, label: str) -> None:
    n, ch = len(data), data.shape[1]
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "f32le", "-ar", str(sample_rate), "-ac", str(ch), "-i", "pipe:0", *codec_args, str(path)]
    kw = _popen_kwargs()
    kw.pop("stdin", None)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)
    try:
        for start in range(0, n, block):
            chunk = np.clip(np.asarray(data[start:start + block], dtype=np.float32), -1.0, 1.0)
            proc.stdin.write(np.ascontiguousarray(chunk, dtype="<f4").tobytes())
            if progress:
                progress(min(1.0, (start + block) / max(1, n)))
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise AudioError(f"{label} 저장 실패: {err.strip()[-300:]}")
    except BrokenPipeError as exc:
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait()
        raise AudioError(f"{label} 저장 실패: {err.strip()[-300:]}") from exc


def write_mp3(path: Path, data: np.ndarray, bitrate: int = 320, sample_rate: int = SAMPLE_RATE,
              progress: Optional[Callable[[float], None]] = None, block: int = SAMPLE_RATE * 10) -> None:
    """MP3. FFmpeg(libmp3lame)이 있으면 LAME 헤더(인코더 지연 정보)를 기록해
    재생 시 앞부분 무음/싱크 어긋남이 없도록 하고, 없으면 lameenc 로 인코딩."""
    if ffmpeg_has_lame():
        _ffmpeg_encode(path, data, sample_rate, ["-c:a", "libmp3lame", "-b:a", f"{bitrate}k", "-q:a", "0",
                                                 "-write_xing", "1"], progress, block, "MP3")
        return
    try:
        import lameenc
    except ImportError as exc:  # pragma: no cover
        raise AudioError("MP3 인코더(lameenc)를 불러올 수 없습니다.") from exc
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(data.shape[1])
    enc.set_quality(2)  # 2 = 고품질
    n = len(data)
    with open(path, "wb") as f:
        for start in range(0, n, block):
            chunk = np.asarray(data[start:start + block], dtype=np.float32)
            f.write(enc.encode(_to_int(chunk, 16).tobytes()))
            if progress:
                progress(min(1.0, (start + block) / max(1, n)))
        f.write(enc.flush())


def write_wav_float(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE,
                    progress: Optional[Callable[[float], None]] = None, block: int = SAMPLE_RATE * 10) -> None:
    """32-bit float WAV (WAVE_FORMAT_IEEE_FLOAT). 0dBFS 를 넘는 값도 그대로 보존."""
    import struct

    n, ch = len(data), data.shape[1]
    data_bytes = n * ch * 4
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 26 + 12 + 8 + data_bytes) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHHH", 18, 3, ch, sample_rate, sample_rate * ch * 4, ch * 4, 32, 0))
        f.write(b"fact" + struct.pack("<II", 4, n))
        f.write(b"data" + struct.pack("<I", data_bytes))
        for start in range(0, n, block):
            f.write(np.ascontiguousarray(data[start:start + block], dtype="<f4").tobytes())
            if progress:
                progress(min(1.0, (start + block) / max(1, n)))


def write_flac(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE,
               progress: Optional[Callable[[float], None]] = None, block: int = SAMPLE_RATE * 10) -> None:
    """FLAC 24-bit (FFmpeg 로 인코딩)."""
    _ffmpeg_encode(path, data, sample_rate, ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24",
                                             "-compression_level", "8"], progress, block, "FLAC")


def write_audio(path: Path, data: np.ndarray, fmt: str, sample_rate: int = SAMPLE_RATE,
                progress: Optional[Callable[[float], None]] = None) -> None:
    if fmt == "wav16":
        write_wav(path, data, 16, sample_rate, progress=progress)
    elif fmt == "wav24":
        write_wav(path, data, 24, sample_rate, progress=progress)
    elif fmt == "wav32f":
        write_wav_float(path, data, sample_rate, progress=progress)
    elif fmt == "flac":
        write_flac(path, data, sample_rate, progress=progress)
    elif fmt == "mp3":
        write_mp3(path, data, sample_rate=sample_rate, progress=progress)
    else:
        raise ValueError(f"unknown format {fmt}")


def estimate_export_bytes(n_samples: int, fmt: str, sample_rate: int = SAMPLE_RATE) -> int:
    if fmt == "wav16":
        return n_samples * CHANNELS * 2 + 1024
    if fmt in ("wav24", "flac"):
        return n_samples * CHANNELS * 3 + 1024
    if fmt == "wav32f":
        return n_samples * CHANNELS * 4 + 1024
    return int(n_samples / sample_rate * 320_000 / 8) + 4096


_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_filename(name: str, max_len: int = 120) -> str:
    name = _BAD_CHARS.sub("_", name).strip().rstrip(".")
    return (name or "track")[:max_len]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 2
    while True:
        cand = path.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand
        i += 1


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
