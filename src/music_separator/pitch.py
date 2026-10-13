"""키(음높이) 조절 — 템포는 그대로 두고 반음 단위로 음높이만 변경.

Signalsmith Stretch (MIT) 의 Python 바인딩(python-stretch)을 사용합니다.
다성(화음) 음악에 맞춘 고품질 알고리즘이며, 0 반음이면 원본을 그대로 돌려줍니다(무손실).
"""
from __future__ import annotations

import numpy as np

KEY_MIN, KEY_MAX = -12, 12


def available() -> bool:
    try:
        from python_stretch import Signalsmith  # noqa: F401
        return True
    except Exception:
        return False


def shift(data: np.ndarray, semitones: float, sample_rate: int = 44100) -> np.ndarray:
    """(n, ch) float32 → 같은 길이의 (n, ch) float32.

    곡 전체를 한 번에 처리합니다. 블록으로 나눠 따로 처리하면 경계에서 위상이 어긋나
    소리가 뭉개지므로 피합니다 (5분 곡 기준 메모리 약 200MB).
    """
    if semitones == 0:
        return np.asarray(data, dtype=np.float32)
    from python_stretch import Signalsmith

    n, ch = len(data), data.shape[1]
    st = Signalsmith.Stretch()
    st.preset(ch, float(sample_rate))
    st.setTransposeSemitones(float(semitones))
    out = st.process(np.ascontiguousarray(np.asarray(data, dtype=np.float32).T))
    out = out.T
    if len(out) < n:
        out = np.pad(out, ((0, n - len(out)), (0, 0)))
    return np.ascontiguousarray(out[:n], dtype=np.float32)


def key_label(semitones: int) -> str:
    return "원래 키" if semitones == 0 else f"{semitones:+d}"
