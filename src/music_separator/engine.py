"""음원 분리 엔진 (GUI 와 독립적으로 동작, 테스트 가능).

- 2-Stem(보컬/반주): 보컬만 AI 로 추정하고 반주는 '원본 − 보컬'로 계산 (두 트랙의 합 = 원본)
- 엔진: BS-RoFormer SW (품질 우선, 기본) / Demucs htdemucs_ft 보컬 모델 (속도 우선)
  선정 근거 (MVSEP Multisong 데이터셋, 같은 기준): 보컬 SDR — BS-RoFormer SW 11.30dB,
  htdemucs_ft 8.33dB, htdemucs 8.18dB / 반주 SDR — 17.50dB, 14.63dB, 14.49dB
- GPU(CUDA / Apple MPS) 우선 사용, 실패·메모리 부족·비정상 출력 시 자동으로 CPU 재시도
- 진행률 보고 및 즉시 취소, 모든 처리는 float32
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
    "vocals":       ("보컬", "Vocals", "#f29d12"),
    "instrumental": ("반주 (MR)", "Instrumental", "#3e8e7e"),
    "drums":        ("드럼", "Drums", "#2f8ff0"),
    "bass":         ("베이스", "Bass", "#9b51f0"),
    "guitar":       ("기타", "Guitar", "#22b573"),
    "piano":        ("피아노", "Piano", "#ee4f92"),
    "other":        ("기타 악기", "Other", "#7d889b"),
    "original":     ("원본", "Original", "#8f98a8"),
}


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    description: str
    outputs: tuple       # 마지막 트랙은 항상 "원본 − 나머지 트랙" (트랙 합 = 원본)


MODES: dict[str, Mode] = {
    "2": Mode("2", "2-Stem", "보컬 / 반주", ("vocals", "instrumental")),
    "4": Mode("4", "4-Stem", "보컬 / 드럼 / 베이스 / 기타 악기", ("vocals", "drums", "bass", "other")),
    "6": Mode("6", "6-Stem", "보컬 / 드럼 / 베이스 / 기타 / 피아노 / 나머지",
              ("vocals", "drums", "bass", "guitar", "piano", "other")),
}


@dataclass(frozen=True)
class Engine:
    key: str
    title: str
    short: str
    note: str


ENGINES: dict[str, Engine] = {
    "auto": Engine("auto", "자동 (권장)", "자동",
                   "품질 우선·최고 품질은 BS-RoFormer SW, 속도 우선은 Demucs 보컬 모델을 사용합니다."),
    "roformer": Engine("roformer", "BS-RoFormer SW (최고 품질)", "BS-RoFormer",
                       "보컬 SDR 11.30dB · 반주 17.50dB (MVSEP Multisong). 첫 사용 시 667MB 다운로드, GPU 권장. "
                       "사용 조건이 명시되지 않은 커뮤니티 모델이라 개인 용도로 사용하세요."),
    "demucs": Engine("demucs", "Demucs htdemucs_ft (빠름)", "Demucs",
                     "보컬 SDR 8.33dB · 반주 14.63dB (MVSEP Multisong). 가볍고 빠르며 MIT 라이선스입니다."),
}


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    description: str
    auto_engine: str          # 엔진 '자동'일 때 사용할 엔진
    roformer_overlap: int     # 구간 겹침 횟수 (2 = 50% 겹침)
    demucs_shifts: int        # 시간 이동 평균 횟수 (Demucs 논문: 늘릴수록 약간 향상)
    demucs_overlap: float


PRESETS: dict[str, Preset] = {
    "speed": Preset("speed", "속도 우선", "Demucs 보컬 모델로 빠르게 분리합니다 (CPU 에서도 곡 길이 정도).",
                    "demucs", 2, 1, 0.25),
    "quality": Preset("quality", "품질 우선", "BS-RoFormer SW 로 분리합니다 (기본·권장).",
                      "roformer", 2, 2, 0.5),
    "best": Preset("best", "최고 품질", "구간을 4번 겹쳐 처리해 경계를 더 매끄럽게 합니다 (시간 약 2배, 차이는 작음).",
                   "roformer", 4, 4, 0.5),
}
DEFAULT_PRESET = "quality"

# 한 가지 소스(보컬)만 학습된 하위 모델 — 나머지 출력은 쓰지 않음
SINGLE_TARGET_MODELS = {"htdemucs_ft_vocals"}

# 6-Stem(숨김 기능) 2단계 시간 이동 횟수
SIX_SHIFTS = {False: 1, True: 2}


def _preset(preset) -> Preset:
    if isinstance(preset, bool):           # 이전 버전 호환: high_quality
        return PRESETS["best" if preset else "quality"]
    return PRESETS.get(str(preset), PRESETS[DEFAULT_PRESET])


def resolve_engine(engine: str, preset="quality", override: Optional[str] = None) -> str:
    if override:
        return "roformer" if override.startswith(("roformer", "bs_roformer")) else "demucs"
    if engine in ("roformer", "demucs"):
        return engine
    return _preset(preset).auto_engine


def plan_models(engine: str, mode_key: str, preset="quality", override: Optional[str] = None) -> dict:
    """역할별 필요한 모델 이름. main = 1단계 분리, six = 6-Stem 2단계(기타·피아노)."""
    p = _preset(preset)
    eng = resolve_engine(engine, p.key, override)
    if eng == "roformer":
        return {"roformer": override or "bs_roformer_sw"}
    if override == "demucs_unittest":
        plan = {"main": "demucs_unittest"}
        if mode_key == "6":
            plan["six"] = "demucs_unittest6"
        return plan
    if mode_key == "2":
        return {"main": override or "htdemucs_ft_vocals"}
    plan = {"main": override or ("htdemucs" if p.key == "speed" else "htdemucs_ft")}
    if mode_key == "6":
        plan["six"] = "htdemucs_6s"
    return plan


def required_models(engine: str, mode_key: str, preset="quality", override: Optional[str] = None) -> list:
    return list(dict.fromkeys(plan_models(engine, mode_key, preset, override).values()))


MODEL_RATE = 44100        # 두 엔진 모두 44.1kHz 에서 학습·처리


def stem_label(key: str, mode_key: str = "4") -> str:
    ko, en, _ = STEM_STYLE[key]
    if key == "other" and mode_key == "6":
        ko = "나머지 악기"
    return f"{ko}"


def stem_english(key: str) -> str:
    return STEM_STYLE[key][1]


def stem_color(key: str) -> str:
    return STEM_STYLE[key][2]


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
    model: str          # 사용한 모델 이름들 (표시용)
    device: str
    stems: list[StemResult]
    n_samples: int
    seconds: float
    elapsed: float
    notices: list[str] = field(default_factory=list)
    engine: str = "demucs"
    raw_recon_error: float = 0.0    # 모델 출력 합과 원본의 차이 (자가진단용)
    preset: str = DEFAULT_PRESET
    model_rate: int = MODEL_RATE    # AI 모델 출력(보컬 등) 샘플레이트


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


class _Progress:
    """여러 단계로 나뉜 작업의 전체 진행률/남은 시간 계산."""

    def __init__(self, report: Callable[[float, str], None], stages: list[tuple[str, float]]):
        self.report = report
        self.stages = stages
        total = sum(w for _, w in stages) or 1.0
        self.offsets, acc = [], 0.0
        for _, w in stages:
            self.offsets.append(acc / total)
            acc += w
        self.weights = [w / total for _, w in stages]
        self.idx = 0
        self.t0 = time.monotonic()
        self.last = 0.0
        self.device = "cpu"

    def stage(self, i: int):
        self.idx = i
        self.update(0.0, force=True)

    def update(self, frac: float, force: bool = False):
        now = time.monotonic()
        if not force and now - self.last < 0.15:
            return
        self.last = now
        overall = self.offsets[self.idx] + self.weights[self.idx] * max(0.0, min(1.0, frac))
        overall = 0.04 + 0.9 * overall
        eta = ""
        el = now - self.t0
        if overall > 0.08 and el > 3:
            eta = f" · 남은 시간 약 {_fmt_eta(el / (overall - 0.04) * (0.94 - overall))}"
        label = self.stages[self.idx][0]
        self.report(overall, f"{label} ({self.device.upper()}) {frac * 100:.0f}%{eta}")


def _run_demucs(model, mix: np.ndarray, device: str, shifts: int, overlap: float,
                cancel: threading.Event, on_progress: Callable[[float], None]) -> dict:
    """Demucs 모델 실행 → {source: (n, 2) ndarray}."""
    import torch
    from demucs.apply import apply_model

    n = len(mix)
    wav = torch.from_numpy(np.ascontiguousarray(np.asarray(mix, dtype=np.float32).T))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    subs = list(getattr(model, "models", [model]))
    est = 0
    for sub in subs:
        seg_len = int(sub.samplerate * float(getattr(sub, "segment", 8.0)))
        stride = max(1, int((1 - overlap) * seg_len))
        length = n + (int(0.5 * sub.samplerate) if shifts else 0)
        est += max(1, math.ceil(length / stride)) * max(1, shifts)
    done = [0]

    def callback(d: dict) -> None:
        if cancel.is_set():
            raise SeparationCancelled()
        if d.get("state") == "end":
            done[0] += 1
            on_progress(min(0.99, done[0] / est))

    out = apply_model(model, wav[None], device=device, shifts=shifts, split=True, overlap=overlap,
                      progress=False, num_workers=0, callback=callback)
    if not torch.isfinite(out).all():
        raise SeparationError(f"{device.upper()} 처리 결과에 잘못된 값(NaN)이 포함되었습니다.")
    out = (out[0] * std + mean).cpu().numpy()
    return {name: np.ascontiguousarray(out[i].T, dtype=np.float32) for i, name in enumerate(model.sources)}


def _run_roformer(model, mix: np.ndarray, device: str, overlap: int, cancel: threading.Event,
                  on_progress: Callable[[float], None]) -> dict:
    from . import roformer

    try:
        out = roformer.demix(model, mix, device, num_overlap=overlap, progress=on_progress, cancel=cancel)
    except roformer.Cancelled as exc:
        raise SeparationCancelled() from exc
    if not np.isfinite(out).all():
        raise SeparationError(f"{device.upper()} 처리 결과에 잘못된 값(NaN)이 포함되었습니다.")
    names = getattr(model, "instruments", roformer.SW_INSTRUMENTS)
    return {name: out[i] for i, name in enumerate(names)}


def separate(
    mix: np.ndarray,
    mode_key: str,
    out_dir: Path,
    *,
    engine: str = "auto",
    preset: Optional[str] = None,
    high_quality: bool = False,
    device: str = "cpu",
    model_name: Optional[str] = None,
    progress: Optional[ProgressCb] = None,
    cancel: Optional[threading.Event] = None,
    notice: Optional[Callable[[str], None]] = None,
) -> SeparationResult:
    """mix: (samples, 2) float32 44.1kHz (memmap 가능). 결과 트랙은 out_dir 에 .npy 로 저장.

    마지막 트랙(반주/기타 악기/나머지)은 "원본 − 다른 트랙들"로 계산하므로
    모든 트랙을 더하면 원본과 정확히 같습니다 (분리 과정에서 소리가 사라지지 않음).
    """
    t0 = time.monotonic()
    mode = MODES[mode_key]
    p = _preset(preset if preset is not None else high_quality)
    high_quality = p.key == "best"
    engine = resolve_engine(engine, p.key, model_name)
    plan = plan_models(engine, mode_key, p.key, model_name)
    cancel = cancel or threading.Event()
    notices: list[str] = []

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    def check_cancel() -> None:
        if cancel.is_set():
            raise SeparationCancelled()

    if engine == "roformer":
        stages = [("보컬 분리 중 · BS-RoFormer", 1.0)]
    else:
        main_cost = float(len(getattr(model_store.MODELS.get(plan["main"]), "sigs", ()) or (1,)))
        main_cost *= p.demucs_shifts
        stages = [("보컬 분리 중 · Demucs" if mode_key == "2" else "분리 중 · 보컬/드럼/베이스", main_cost)]
        if "six" in plan:
            stages.append(("분리 중 · 기타/피아노", 1.0 * SIX_SHIFTS[high_quality]))
    prog = _Progress(report, stages)

    report(0.01, "AI 모델 불러오는 중…")
    loaded = {role: model_store.load_model(name) for role, name in plan.items()}
    check_cancel()
    mix = np.asarray(mix, dtype=np.float32)

    def pipeline(dev: str) -> dict:
        prog.device = dev
        if engine == "roformer":
            prog.stage(0)
            raw = _run_roformer(loaded["roformer"], mix, dev, p.roformer_overlap, cancel, prog.update)
            _release(loaded["roformer"], dev)
            return raw
        prog.stage(0)
        raw = _run_demucs(loaded["main"], mix, dev, p.demucs_shifts, p.demucs_overlap, cancel, prog.update)
        _release(loaded["main"], dev)
        if "six" in loaded:
            # 2단계: '기타 악기' 트랙만 6-Stem 모델에 넣어 기타·피아노를 추가로 분리
            check_cancel()
            prog.stage(1)
            sub = _run_demucs(loaded["six"], raw["other"], dev, SIX_SHIFTS[high_quality],
                              p.demucs_overlap, cancel, prog.update)
            _release(loaded["six"], dev)
            raw["guitar"] = sub.get("guitar", np.zeros_like(mix))
            raw["piano"] = sub.get("piano", np.zeros_like(mix))
        return raw

    run_device = device
    try:
        raw = pipeline(device)
    except SeparationCancelled:
        for m in loaded.values():
            _release(m, device)
        raise
    except Exception as exc:
        if device == "cpu" or not _is_gpu_problem(exc):
            for m in loaded.values():
                _release(m, device)
            raise
        log.exception("GPU(%s) 처리 실패, CPU 로 재시도", device)
        msg = (f"GPU({device.upper()}) 처리 중 문제가 발생하여 CPU로 다시 처리합니다. "
               f"시간이 더 걸릴 수 있습니다.\n({type(exc).__name__}: {str(exc)[:200]})")
        notices.append(msg)
        if notice:
            notice(msg)
        for m in loaded.values():
            _release(m, device)
        check_cancel()
        # GPU 에서 실패한 모델 객체는 상태를 신뢰할 수 없으므로 디스크에서 새로 불러옴
        for role, name in plan.items():
            model_store.forget(name)
            loaded[role] = model_store.load_model(name)
        run_device = "cpu"
        raw = pipeline("cpu")
    check_cancel()

    # 모델 출력 합 vs 원본 (진단용)
    base_keys = [k for k in ("vocals", "drums", "bass", "other") if k in raw]
    if engine == "roformer":
        base_keys += [k for k in ("guitar", "piano") if k in raw]
    if any(n in SINGLE_TARGET_MODELS for n in plan.values()):
        # 보컬 전용 하위 모델은 보컬 출력만 학습되어 있어 '모든 출력의 합 = 원본' 검사가 의미 없음
        recon = float("nan")
    else:
        raw_sum = sum(raw[k] for k in base_keys)
        recon = float(np.linalg.norm(raw_sum - mix) / (np.linalg.norm(mix) + 1e-9))
        del raw_sum

    report(0.95, "트랙 저장 중…")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    explicit = mode.outputs[:-1]
    residual_key = mode.outputs[-1]
    stems: list[StemResult] = []
    acc = np.zeros_like(mix)
    for key in explicit:
        check_cancel()
        arr = np.ascontiguousarray(raw[key], dtype=np.float32)
        acc += arr
        path = out_dir / f"{key}.npy"
        np.save(path, arr)
        stems.append(StemResult(key, path, compute_peaks(arr)))
    residual = mix - acc
    path = out_dir / f"{residual_key}.npy"
    np.save(path, residual)
    stems.append(StemResult(residual_key, path, compute_peaks(residual)))
    del raw, acc, residual
    report(1.0, "완료")
    return SeparationResult(mode_key, " + ".join(dict.fromkeys(plan.values())), run_device, stems, len(mix),
                            len(mix) / 44100.0, time.monotonic() - t0, notices, engine, recon, p.key, MODEL_RATE)


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
