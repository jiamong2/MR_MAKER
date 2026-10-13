"""Demucs 음원 분리 엔진 (GUI 와 독립적으로 동작, 테스트 가능).

- GPU(CUDA / Apple MPS) 우선 사용, 실패·메모리 부족·비정상 출력 시 자동으로 CPU 재시도
- Demucs 4.1 apply_model 의 callback 으로 세그먼트 단위 진행률 보고 및 즉시 취소
"""
from __future__ import annotations

import errno
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from . import models as model_store
from .dsp import compute_peaks

log = logging.getLogger(__name__)

# 표시 이름(한국어, 영어), 색상
STEM_STYLE = {
    "vocals":       ("보컬", "Vocals", "#F5A623"),
    "instrumental": ("반주", "Instrumental", "#7B8CFF"),
    "drums":        ("드럼", "Drums", "#3FB8F0"),
    "bass":         ("베이스", "Bass", "#B57BFF"),
    "guitar":       ("기타", "Guitar", "#5BD08A"),
    "piano":        ("피아노", "Piano", "#FF7EB6"),
    "other":        ("기타 악기", "Other", "#8FA3B0"),
    "original":     ("원본", "Original", "#C9CED6"),
}


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    description: str
    model: str
    hq_model: str
    hq_shifts: int
    outputs: tuple
    note: str = ""


MODES: dict[str, Mode] = {
    "2": Mode("2", "2-Stem", "보컬 / 반주", "htdemucs", "htdemucs_ft", 1,
              ("vocals", "instrumental"),
              "노래방 MR이나 보컬만 필요할 때 가장 깔끔합니다."),
    "4": Mode("4", "4-Stem", "보컬 / 드럼 / 베이스 / 기타 악기", "htdemucs", "htdemucs_ft", 1,
              ("vocals", "drums", "bass", "other"),
              "Demucs가 가장 잘하는 기본 분리입니다. 기타·피아노 등은 '기타 악기'에 함께 들어갑니다."),
    "6": Mode("6", "6-Stem", "보컬 / 드럼 / 베이스 / 기타 / 피아노 / 나머지", "htdemucs_6s", "htdemucs_6s", 2,
              ("vocals", "drums", "bass", "guitar", "piano", "other"),
              "실험적 모델입니다. 피아노는 다른 악기 소리가 섞이거나 작게 분리되는 경우가 많고, "
              "기타는 비교적 양호합니다. 보컬·드럼·베이스 품질은 4-Stem과 비슷합니다."),
}


def stem_label(key: str, mode_key: str = "4") -> str:
    ko, en, _ = STEM_STYLE[key]
    if key == "other" and mode_key == "6":
        ko = "나머지 악기"
    return f"{ko}"


def stem_english(key: str) -> str:
    return STEM_STYLE[key][1]


def stem_color(key: str) -> str:
    return STEM_STYLE[key][2]


def model_for(mode_key: str, high_quality: bool) -> str:
    m = MODES[mode_key]
    return m.hq_model if high_quality else m.model


def shifts_for(mode_key: str, high_quality: bool) -> int:
    return MODES[mode_key].hq_shifts if high_quality else 1


class SeparationCancelled(Exception):
    pass


class SeparationError(Exception):
    """사용자에게 그대로 보여줄 수 있는 오류 메시지."""


@dataclass
class StemResult:
    key: str
    path: Path
    peaks: np.ndarray


@dataclass
class SeparationResult:
    mode: str
    model: str
    device: str
    stems: list[StemResult]
    n_samples: int
    seconds: float
    elapsed: float
    notices: list[str] = field(default_factory=list)


ProgressCb = Callable[[float, str], None]


def configure_threads() -> None:
    import torch

    n = max(1, (os.cpu_count() or 2) - 1)  # UI 용 코어 1개 남김
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def friendly_error(exc: BaseException, action: str = "분리") -> str:
    text = str(exc)
    low = text.lower()
    if isinstance(exc, MemoryError) or "out of memory" in low or "can't allocate" in low \
            or "not enough memory" in low or "defaultcpuallocator" in low:
        return (f"메모리(RAM)가 부족하여 {action} 작업을 완료하지 못했습니다.\n"
                "다른 프로그램을 종료하거나, 곡을 짧게 잘라서 다시 시도해 주세요.")
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "디스크 공간이 부족합니다. 저장 공간을 확보한 뒤 다시 시도해 주세요."
    if isinstance(exc, (model_store.ModelError, SeparationError)):
        return text
    return f"{action} 중 오류가 발생했습니다.\n({type(exc).__name__}: {text})"


def _is_gpu_problem(exc: BaseException) -> bool:
    return not isinstance(exc, (SeparationCancelled, MemoryError, KeyboardInterrupt))


def separate(
    mix: np.ndarray,
    mode_key: str,
    out_dir: Path,
    *,
    high_quality: bool = False,
    device: str = "cpu",
    model_name: Optional[str] = None,
    progress: Optional[ProgressCb] = None,
    cancel: Optional[threading.Event] = None,
    notice: Optional[Callable[[str], None]] = None,
) -> SeparationResult:
    """mix: (samples, 2) float32 (memmap 가능). 결과 트랙은 out_dir 에 .npy 로 저장."""
    import torch
    from demucs.apply import apply_model

    t0 = time.monotonic()
    mode = MODES[mode_key]
    model_name = model_name or model_for(mode_key, high_quality)
    shifts = shifts_for(mode_key, high_quality)
    cancel = cancel or threading.Event()
    notices: list[str] = []

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    def check_cancel() -> None:
        if cancel.is_set():
            raise SeparationCancelled()

    report(0.01, "AI 모델 불러오는 중…")
    model = model_store.load_model(model_name)
    check_cancel()

    n = len(mix)
    wav = torch.from_numpy(np.ascontiguousarray(np.asarray(mix, dtype=np.float32).T))
    ref = wav.mean(0)
    mean = ref.mean()
    std = ref.std() + 1e-8
    wav = (wav - mean) / std

    # 진행률 계산용 예상 세그먼트 수
    subs = list(getattr(model, "models", [model]))
    est_total = 0
    for sub in subs:
        seg = float(getattr(sub, "segment", 8.0))
        seg_len = int(sub.samplerate * seg)
        stride = max(1, int(0.75 * seg_len))
        length = n + (int(0.5 * sub.samplerate) if shifts else 0)
        est_total += max(1, math.ceil(length / stride)) * max(1, shifts)
    done = [0]
    last_emit = [0.0]

    def callback(d: dict) -> None:
        if cancel.is_set():
            raise SeparationCancelled()
        if d.get("state") == "end":
            done[0] += 1
            now = time.monotonic()
            if now - last_emit[0] > 0.15:
                last_emit[0] = now
                frac = min(0.99, done[0] / est_total)
                elapsed = now - t_run[0]
                eta = ""
                if done[0] >= 3 and frac > 0.02:
                    remain = elapsed / frac * (1 - frac)
                    eta = f" · 남은 시간 약 {_fmt_eta(remain)}"
                report(0.05 + 0.88 * frac, f"분리 중 ({run_device[0].upper()}) {frac * 100:.0f}%{eta}")

    run_device = [device]
    t_run = [time.monotonic()]

    def run(dev: str):
        done[0] = 0
        run_device[0] = dev
        t_run[0] = time.monotonic()
        report(0.05, f"분리 시작 ({dev.upper()})…")
        out = apply_model(model, wav[None], device=dev, shifts=shifts, split=True,
                          overlap=0.25, progress=False, num_workers=0, callback=callback)
        if not torch.isfinite(out).all():
            raise SeparationError(f"{dev.upper()} 처리 결과에 잘못된 값(NaN)이 포함되었습니다.")
        return out

    try:
        out = run(device)
    except SeparationCancelled:
        _release(model, device)
        raise
    except Exception as exc:
        if device == "cpu" or not _is_gpu_problem(exc):
            _release(model, device)
            raise
        log.exception("GPU(%s) 처리 실패, CPU 로 재시도", device)
        msg = (f"GPU({device.upper()}) 처리 중 문제가 발생하여 CPU로 다시 처리합니다. "
               f"시간이 더 걸릴 수 있습니다.\n({type(exc).__name__}: {str(exc)[:200]})")
        notices.append(msg)
        if notice:
            notice(msg)
        _release(model, device)
        check_cancel()
        # GPU 에서 실패한 모델 객체는 상태를 신뢰할 수 없으므로 디스크에서 새로 불러옴
        model_store.forget(model_name)
        model = model_store.load_model(model_name)
        out = run("cpu")
    _release(model, run_device[0])
    check_cancel()

    out = out[0] * std + mean  # (sources, 2, n)
    sources = list(model.sources)
    report(0.94, "트랙 저장 중…")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems: list[StemResult] = []
    for key in mode.outputs:
        check_cancel()
        if key == "instrumental":
            tensor = sum(out[i] for i, s in enumerate(sources) if s != "vocals")
        else:
            tensor = out[sources.index(key)]
        arr = np.ascontiguousarray(tensor.T.cpu().numpy(), dtype=np.float32)
        path = out_dir / f"{key}.npy"
        np.save(path, arr)
        stems.append(StemResult(key, path, compute_peaks(arr)))
        del arr
    del out
    report(1.0, "완료")
    return SeparationResult(mode_key, model_name, run_device[0], stems, n, n / 44100.0,
                            time.monotonic() - t0, notices)


def _release(model, device: str) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _fmt_eta(sec: float) -> str:
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec}초"
    m, s = divmod(sec, 60)
    return f"{m}분 {s:02d}초"
