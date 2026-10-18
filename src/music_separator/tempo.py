"""원곡 BPM 추정 (참고 표시용, numpy 만 사용).

템포 조절 자체는 원곡 대비 배율(%)로 처리하므로 이 값이 틀려도 결과물에는 영향이 없다.
화면에 "원곡 약 92 BPM → 83 BPM" 처럼 보여 주고, 확실하지 않으면 표시하지 않는다.

방법
1. 11.025kHz 모노 → STFT(1024, hop 256 ≈ 23ms) → 로그 스펙트럼의 증가량 합 = 소리가 시작되는 세기(onset)
2. 곡을 8초 구간으로 나눠 구간마다 onset 의 자기상관 → 평균 (중간에 템포가 조금 흔들려도 전체 경향을 봄)
3. 40~240 BPM 중 자기상관이 가장 큰 값, 사람들이 흔히 느끼는 템포(115 BPM 중심)를 조금 더 선호
   (60 ↔ 120 처럼 반/두 배로 틀리는 경우를 줄임)
4. 신뢰도: 구간마다 따로 구한 템포가 전체 값(또는 그 반/두 배)과 4% 안에서 맞는 비율 × 봉우리의 뚜렷함.
   소리가 시작되는 순간이 거의 없는 곡(지속음만 있는 곡)은 모름
※ 60 ↔ 120 처럼 반/두 배로 읽힐 수 있는 곡이 많아 화면에서 ×2 / ÷2 로 바로 고칠 수 있게 함.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SR = 11025
N_FFT = 1024
HOP = 256
FPS = SR / HOP                 # 약 43.07 프레임/초
BPM_MIN, BPM_MAX = 40.0, 240.0
WINDOW_SEC = 8.0
MIN_CONFIDENCE = 0.35
PRIOR_BPM, PRIOR_OCT = 115.0, 0.8   # 사람들이 흔히 느끼는 템포 쪽을 조금 선호 (로그 정규 분포)
MIN_PERIODIC = 0.07            # 박자 위치의 자기상관(0~1)이 이보다 작으면 주기성 없음 (잡음은 0.03~0.065)
FULL_PERIODIC = 0.15           # 이 이상이면 주기성 충분
MIN_ONSET = 1.5                # onset 세기 평균이 이보다 작으면 (타악기·뚜렷한 시작음이 거의 없음) 모름으로 봄


@dataclass
class TempoEstimate:
    bpm: float                 # 추정 BPM (0 = 모름)
    confidence: float          # 0~1
    stable: float              # 구간별 템포가 전체와 맞는 비율

    @property
    def reliable(self) -> bool:
        return self.bpm > 0 and self.confidence >= MIN_CONFIDENCE


def onset_envelope(audio: np.ndarray, rate: int) -> np.ndarray:
    """(n,) 또는 (n, ch) → onset 세기 (FPS 간격)."""
    from .audio_io import resample

    x = np.asarray(audio, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    y = resample(mono[:, None], rate, SR)[:, 0] if rate != SR else mono
    if len(y) < N_FFT * 4:
        return np.zeros(0)
    win = np.hanning(N_FFT).astype(np.float32)
    T = 1 + (len(y) - N_FFT) // HOP
    out = np.zeros(T, np.float64)
    prev = None
    block = 2048
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        idx = (np.arange(t0, t1) * HOP)[:, None] + np.arange(N_FFT)[None, :]
        mag = np.abs(np.fft.rfft(y[idx] * win, axis=1))[:, 1:300]      # 약 3.2kHz 까지
        logm = np.log1p(1000.0 * mag)
        if prev is not None:
            logm_all = np.vstack([prev[None, :], logm])
        else:
            logm_all = np.vstack([logm[:1], logm])
        diff = np.maximum(np.diff(logm_all, axis=0), 0.0).sum(axis=1)
        out[t0:t1] = diff
        prev = logm[-1]
    # 느린 음량 변화를 빼고 (1초 이동 평균), 음수는 0
    k = int(FPS)
    c = np.concatenate([[0.0], np.cumsum(out)])
    i = np.arange(T)
    lo, hi = np.clip(i - k // 2, 0, T), np.clip(i + k // 2 + 1, 0, T)
    out = np.maximum(out - (c[hi] - c[lo]) / (hi - lo), 0.0)
    return out


def _autocorr(o: np.ndarray, max_lag: int) -> np.ndarray:
    o = o - o.mean()
    n = len(o)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(o, m)
    ac = np.fft.irfft(f * np.conj(f), m)[:max_lag + 1]
    return ac / max(ac[0], 1e-12)


def _bpm_of_lag(lag: np.ndarray) -> np.ndarray:
    return 60.0 * FPS / np.maximum(lag, 1e-9)


def _score(ac: np.ndarray) -> tuple:
    """평균 자기상관 → (BPM 후보들, 점수)."""
    lags = np.arange(len(ac), dtype=np.float64)
    bpm = _bpm_of_lag(lags)
    ok = (bpm >= BPM_MIN) & (bpm <= BPM_MAX)
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-9) / PRIOR_BPM) / PRIOR_OCT) ** 2)
    s = np.where(ok, np.maximum(ac, 0) * prior, -np.inf)
    return bpm, s


def _refine(ac_lag: np.ndarray, k: int) -> float:
    """봉우리 위치를 포물선으로 보정한 lag."""
    if 0 < k < len(ac_lag) - 1:
        a, b, c = ac_lag[k - 1], ac_lag[k], ac_lag[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            return k + 0.5 * (a - c) / den
    return float(k)


def estimate(audio: np.ndarray, rate: int) -> TempoEstimate:
    o = onset_envelope(audio, rate)
    if len(o) == 0 or float(o.mean()) < MIN_ONSET:
        return TempoEstimate(0.0, 0.0, 0.0)
    max_lag = int(round(60.0 * FPS / BPM_MIN)) + 2
    win = int(WINDOW_SEC * FPS)
    if len(o) < win:
        return TempoEstimate(0.0, 0.0, 0.0)
    acs = []
    local = []
    for s in range(0, len(o) - win + 1, win // 2):
        seg = o[s:s + win]
        if seg.std() < 1e-6:
            continue
        ac = _autocorr(seg, max_lag)
        acs.append(ac)
        bpm_c, sc = _score(ac)
        k = int(np.argmax(sc))
        local.append(_bpm_of_lag(np.array([_refine(ac, k)]))[0])
    if not acs:
        return TempoEstimate(0.0, 0.0, 0.0)
    ac = np.mean(acs, axis=0)
    bpm_c, sc = _score(ac)
    k = int(np.argmax(sc))
    bpm = float(_bpm_of_lag(np.array([_refine(ac, k)]))[0])
    # 봉우리의 뚜렷함: 최고점과 (최고점 ±10% 밖의) 다음 봉우리 차이
    finite = np.isfinite(sc)
    near = np.abs(np.log2(np.maximum(bpm_c, 1e-9) / bpm)) < np.log2(1.1)
    rel = {r: np.abs(np.log2(np.maximum(bpm_c, 1e-9) / (bpm * r))) < np.log2(1.06) for r in (0.5, 2.0, 1.5, 2 / 3)}
    excl = near | rel[0.5] | rel[2.0] | rel[1.5] | rel[2 / 3]
    rest = sc[finite & ~excl]
    second = float(rest.max()) if len(rest) else 0.0
    peak = float(sc[k])
    sharp = float(np.clip((peak - second) / max(peak, 1e-9), 0.0, 1.0))
    loc = np.array(local)
    ratio = loc / bpm
    match = np.min(np.abs(np.log2(ratio[:, None]) - np.log2(np.array([0.5, 1.0, 2.0]))[None, :]), axis=1) < np.log2(1.04)
    stable = float(match.mean()) if len(loc) else 0.0
    periodic = float(np.clip((ac[k] - MIN_PERIODIC) / (FULL_PERIODIC - MIN_PERIODIC), 0.0, 1.0))
    conf = float(np.clip(0.4 * stable + 0.6 * sharp * 2.0, 0.0, 1.0) * stable ** 0.5 * periodic)
    return TempoEstimate(round(bpm, 1), conf, stable)


def target_bpm(original: float, percent: int) -> float:
    return original * percent / 100.0


def percent_for(original: float, target: float, lo: int = 70, hi: int = 130) -> Optional[int]:
    """원곡 BPM → 목표 BPM 이 되는 템포(%) (범위 밖이면 None)."""
    if original <= 0 or target <= 0:
        return None
    p = int(round(100.0 * target / original))
    return p if lo <= p <= hi else None
