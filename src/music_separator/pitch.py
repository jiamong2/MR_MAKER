"""키(음높이)·템포(빠르기) 조절.

- 키: 템포·길이는 그대로 두고 반음 단위로 음높이만 변경
- 템포: 음높이는 그대로 두고 빠르기만 변경 (50~200%, 길이는 1/배율 로 바뀜)
둘 다 바꿀 때는 한 번에 처리합니다 (두 번 처리할 때보다 음질 손실이 적음).

엔진 선택 근거 (직접 측정, docs/QUALITY.md):
- Rubber Band Library R3("finer") 엔진: 배음 주파수 오차 0.0센트, 스테레오 상관 변화 0.0002,
  ±2~3반음 왕복 스펙트럼 거리(LSD) 0.83~0.85 → 기본 엔진
- Signalsmith Stretch (python-stretch): 저음(55~220Hz)에서 6~30센트 음정 오차가 측정되어 대체용으로만 사용

Rubber Band 는 GPL 라이선스라 FFmpeg 처럼 별도 실행 파일(rbshift)로 호출합니다 (소스: native/rbshift).
키 0 · 템포 100% 이면 아무 처리도 하지 않고 원본을 그대로 돌려줍니다 (비트 단위 동일).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

KEY_MIN, KEY_MAX = -12, 12
TEMPO_MIN, TEMPO_MAX = 50, 200          # % (반 박자 ~ 두 배 박자. 100% 에서 멀수록 소리가 덜 자연스러움)


def out_frames(n: int, tempo: float) -> int:
    """템포 배율을 적용한 출력 길이 (프레임)."""
    return n if tempo == 1.0 else int(round(n / tempo))


class PitchError(Exception):
    pass


class PitchCancelled(Exception):
    pass


_rb_cache: Optional[str] = None


def find_rbshift() -> Optional[str]:
    """번들된 rbshift 실행 파일 경로 (없으면 None → Signalsmith 로 대체)."""
    global _rb_cache
    if _rb_cache and os.path.exists(_rb_cache):
        return _rb_cache
    exe = "rbshift.exe" if sys.platform == "win32" else "rbshift"
    cands = []
    env = os.environ.get("MUSIC_SEPARATOR_RBSHIFT")
    if env:
        cands.append(Path(env))
    from .paths import resource_dir

    cands += [resource_dir() / "bin" / exe, Path(__file__).resolve().parents[2] / "build" / "rbshift" / exe,
              Path(__file__).resolve().parents[2] / "build" / "rbshift" / "Release" / exe]
    w = shutil.which(exe)
    if w:
        cands.append(Path(w))
    for c in cands:
        if c and c.is_file():
            _rb_cache = str(c)
            return _rb_cache
    return None


def engine_name() -> str:
    if find_rbshift():
        return "Rubber Band R3"
    if _signalsmith_available():
        return "Signalsmith Stretch"
    return ""


def _signalsmith_available() -> bool:
    try:
        from python_stretch import Signalsmith  # noqa: F401
        return True
    except Exception:
        return False


def available() -> bool:
    return bool(engine_name())


def _popen_kwargs() -> dict:
    kw: dict = {}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return kw


def _shift_rubberband(exe: str, data: np.ndarray, semitones: float, sample_rate: int, formant: bool,
                      cancel: Optional[threading.Event], tempo: float = 1.0) -> np.ndarray:
    n, ch = len(data), data.shape[1]
    m = out_frames(n, tempo)
    cmd = [exe, "--rate", str(int(sample_rate)), "--channels", str(ch), "--semitones", f"{semitones:g}"]
    if tempo != 1.0:
        cmd += ["--tempo", repr(float(tempo)), "--out-frames", str(m)]
    if formant:
        cmd.append("--formant")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            **_popen_kwargs())
    out: dict = {}

    def reader():
        out["data"] = proc.stdout.read()

    def err_reader():
        out["err"] = proc.stderr.read()

    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=err_reader, daemon=True)]
    for t in threads:
        t.start()
    try:
        block = sample_rate * 10
        for start in range(0, n, block):
            if cancel is not None and cancel.is_set():
                raise PitchCancelled()
            proc.stdin.write(np.ascontiguousarray(np.asarray(data[start:start + block], dtype="<f4")).tobytes())
        proc.stdin.close()
        while proc.poll() is None:
            if cancel is not None and cancel.wait(0.2):
                raise PitchCancelled()
        for t in threads:
            t.join(5)
    except PitchCancelled:
        proc.kill()
        proc.wait()
        raise
    except (BrokenPipeError, OSError) as exc:
        proc.kill()
        proc.wait()
        for t in threads:
            t.join(2)
        raise PitchError(f"키 변경 엔진 실행 실패: {exc} {out.get('err', b'').decode('utf-8', 'replace')[-300:]}")
    if proc.returncode != 0:
        raise PitchError(f"키 변경 엔진 오류 (코드 {proc.returncode}): "
                         f"{out.get('err', b'').decode('utf-8', 'replace')[-300:]}")
    y = np.frombuffer(out.get("data", b""), dtype="<f4")
    if len(y) != m * ch:
        raise PitchError(f"키·템포 변경 결과 길이가 다릅니다 ({len(y) // max(1, ch)} != {m})")
    return y.reshape(m, ch).astype(np.float32, copy=True)


def _shift_signalsmith(data: np.ndarray, semitones: float, sample_rate: int, tempo: float = 1.0) -> np.ndarray:
    from python_stretch import Signalsmith

    n, ch = len(data), data.shape[1]
    m = out_frames(n, tempo)
    st = Signalsmith.Stretch()
    st.preset(ch, float(sample_rate))
    st.setTransposeSemitones(float(semitones))
    if tempo != 1.0:
        st.setTimeFactor(float(tempo))
    out = st.process(np.ascontiguousarray(np.asarray(data, dtype=np.float32).T)).T
    if len(out) < m:
        out = np.pad(out, ((0, m - len(out)), (0, 0)))
    return np.ascontiguousarray(out[:m], dtype=np.float32)


def shift(data: np.ndarray, semitones: float, sample_rate: int = 44100, formant: bool = False,
          cancel: Optional[threading.Event] = None, tempo: float = 1.0) -> np.ndarray:
    """(n, ch) float32 → (out_frames(n, tempo), ch) float32.

    - semitones == 0 이고 tempo == 1: 처리하지 않고 원본 그대로 반환 (완전 생략)
    - tempo: 빠르기 배율 (0.9 = 10% 느리게, 음높이는 그대로)
    - formant=True: 보컬 음색(포먼트) 유지 (보컬 결과물용, Rubber Band 에서만 지원)
    곡 전체를 한 번에 처리합니다 (블록 경계 잡음 없음, 오프라인 모드로 앞뒤 지연 없이 정렬)."""
    tempo = float(tempo)
    if semitones == 0 and tempo == 1.0:
        return np.asarray(data, dtype=np.float32)
    if not (KEY_MIN <= semitones <= KEY_MAX):
        raise ValueError(f"키 조절 범위는 {KEY_MIN}~{KEY_MAX} 반음입니다")
    if not (TEMPO_MIN / 100 - 1e-9 <= tempo <= TEMPO_MAX / 100 + 1e-9):
        raise ValueError(f"템포 조절 범위는 {TEMPO_MIN}~{TEMPO_MAX}% 입니다")
    exe = find_rbshift()
    if exe:
        return _shift_rubberband(exe, data, float(semitones), int(sample_rate), formant, cancel, tempo)
    if _signalsmith_available():
        log.warning("rbshift 를 찾지 못해 Signalsmith Stretch 로 키·템포를 변경합니다 (포먼트 유지 미지원)")
        return _shift_signalsmith(data, semitones, sample_rate, tempo)
    raise PitchError("키 변경 엔진을 찾을 수 없습니다. 프로그램을 다시 설치해 주세요.")


def key_label(semitones: int) -> str:
    return "원래 키" if semitones == 0 else f"{semitones:+d}"


def tempo_label(percent: int) -> str:
    return "원래 빠르기" if percent == 100 else f"{percent}%"


def tempo_suffix(percent: int) -> str:
    """저장 파일 이름에 붙이는 표시."""
    return f" [Tempo {percent}%]" if percent != 100 else ""
