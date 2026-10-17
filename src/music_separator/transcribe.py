"""음정 곡선(f0) → 음표 → MIDI 파일 (화음 따기용, numpy 만 사용).

배음 마스크: 음정의 배음 자리(주파수에 비례하는 폭)만 통과시키는 STFT 마스크.

음표 정리 입력: 10ms 간격의 음정(또는 f0, Hz)과 같은 간격의 음량(dBFS).
1. 소리가 있는 구간: f0 > 0 이고 음량이 기준 이상 (분리 후 조금 남은 다른 목소리는 기준 아래라 무시)
2. 30ms 이하의 짧은 끊김은 이어 붙임 (같은 음 안의 순간적인 끊김, 앞뒤 높이가 1반음 안쪽일 때)
3. 곡 전체의 튜닝(A4 = 440Hz 에서 벗어난 정도)을 추정해 보정
4. 비브라토를 줄인 곡선(90ms 이동 평균)에서 반음의 0.7배 이상 벗어나면 음이 바뀐 것으로 보고,
   바뀐 위치는 두 음의 가운데를 지나는 지점으로 잡음. 각 음의 높이는 그 구간 음정의 중앙값
5. 70ms 보다 짧은 음은 가까운 높이의 옆 음에 합침
6. 0.18초보다 짧으면서 음정이 흔들리는 조각(미끄러지는 중간에 생긴 지나가는 음, 두 반음 사이를 오가는 조각)은
   옆 음에 합침 — 다음 음 쪽으로 움직이는 중이면 다음 음의 시작으로, 아니면 높이가 가까운 쪽으로.
   음정이 안정된 짧은 음(빠른 꾸밈음)은 그대로 둠

MIDI: 형식 0, 120 BPM, 480 PPQ → 원곡과 시각(초)이 그대로 맞음 (DAW 프로젝트 템포를 120 으로 두고 1마디에 놓기).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

FPS = 100
MIN_NOTE = 0.07          # 초
MAX_GAP = 0.03           # 초
SMOOTH = 0.09            # 초
HYSTERESIS = 0.7         # 반음
UNSTABLE_MAX = 0.18      # 초: 이보다 짧은 조각만 흔들림 검사
UNSTABLE_SLOPE = 7.0     # 반음/초: 조각 안의 기울기가 이보다 크면 미끄러지는 중
UNSTABLE_OFF = 0.3       # 반음: 조각 중앙값이 가장 가까운 반음에서 이만큼 벗어나면 두 음 사이
UNSTABLE_SPREAD = 0.8    # 반음: 조각 안 음정 폭(10~90%)이 이보다 크면 흔들림
PITCH_RANGE = (28, 100)  # MIDI (E1 ~ E7) 밖은 버림
TEMPO_BPM = 120.0
PPQ = 480
CHOIR_AAHS = 52          # General MIDI 프로그램 번호 (0부터)


@dataclass(frozen=True)
class Note:
    start: float         # 초
    end: float           # 초
    pitch: int           # MIDI 번호 (60 = 가온 다)
    velocity: int = 90


def hz_to_midi(f0) -> np.ndarray:
    f = np.asarray(f0, dtype=np.float64)
    out = np.full(f.shape, np.nan)
    ok = f > 0
    out[ok] = 69.0 + 12.0 * np.log2(f[ok] / 440.0)
    return out


def frame_db(audio: np.ndarray, rate: int, n_frames: int, fps: int = FPS, win: float = 0.064) -> np.ndarray:
    """프레임 i (중심 i / fps 초) 주변 win 초의 RMS (dBFS). (n,) 또는 (n, ch) — 채널 평균 전력."""
    x = np.asarray(audio, dtype=np.float32)
    p = (x.astype(np.float64) ** 2).mean(axis=1) if x.ndim == 2 else x.astype(np.float64) ** 2
    c = np.concatenate([[0.0], np.cumsum(p)])
    half = max(1, int(win * rate / 2))
    centers = np.round(np.arange(n_frames) * rate / fps).astype(np.int64)
    lo = np.clip(centers - half, 0, len(p))
    hi = np.clip(centers + half, 0, len(p))
    e = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return 10.0 * np.log10(np.maximum(e, 1e-12))


def estimate_tuning(*tracks: np.ndarray) -> float:
    """곡 전체 튜닝 (반음 단위, -0.5 ~ 0.5). 비브라토를 줄인 곡선(150ms 평균)에서 음이 안정된 프레임의
    소수 부분을 원형 평균."""
    vals = []
    for m in tracks:
        m = np.asarray(m, dtype=np.float64)
        for a, b in _runs(np.isfinite(m)):
            if b - a < 15:
                continue
            s = _smooth(m[a:b], 15)
            d = np.abs(np.diff(s))
            vals.append(s[1:][d < 0.03])
    v = np.concatenate(vals) if vals else np.array([])
    if len(v) < 50:
        return 0.0
    z = np.mean(np.exp(2j * np.pi * (v - np.round(v))))
    if abs(z) < 0.2:          # 소수 부분이 고르게 퍼져 있음 → 판단 불가
        return 0.0
    return float(np.angle(z) / (2 * np.pi))


def _runs(mask: np.ndarray) -> list:
    """True 가 이어진 구간 [(시작, 끝)] (끝은 포함하지 않음)."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """가운데 기준 이동 평균 (양 끝은 창을 줄여서)."""
    k = width // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo, hi = np.clip(i - k, 0, len(x)), np.clip(i + k + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def _segment(m: np.ndarray, width: int, hysteresis: float) -> list:
    """한 유성 구간(m: 반음 단위 음정) → 음 경계 [(시작, 끝)]."""
    s = _smooth(m, width)
    cur = int(np.round(s[0]))
    cuts = [0]
    for i in range(1, len(s)):
        if abs(s[i] - cur) <= hysteresis:
            continue
        # 새 음 높이는 바로 뒤 원래 곡선(50ms)의 중앙값 → 평균 곡선이 지나가는 중간 음을 거치지 않음
        new = int(np.round(np.median(m[i:i + 5])))
        if new == cur:
            continue
        mid = (cur + new) / 2.0
        sign = 1.0 if new > cur else -1.0
        j = i
        while j > cuts[-1] + 1 and (s[j - 1] - mid) * sign > 0:   # 가운데를 지나는 지점까지 되돌아감
            j -= 1
        cuts.append(j)
        cur = new
    cuts.append(len(s))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _merge_short(m: np.ndarray, segs: list, min_len: int) -> list:
    segs = [list(x) for x in segs]
    while len(segs) > 1:
        lens = [b - a for a, b in segs]
        k = int(np.argmin(lens))
        if lens[k] >= min_len:
            break
        p = np.median(m[segs[k][0]:segs[k][1]])
        cands = []
        if k > 0:
            cands.append((abs(np.median(m[segs[k - 1][0]:segs[k - 1][1]]) - p), k - 1))
        if k < len(segs) - 1:
            cands.append((abs(np.median(m[segs[k + 1][0]:segs[k + 1][1]]) - p), k + 1))
        _, j = min(cands)
        lo, hi = min(segs[k][0], segs[j][0]), max(segs[k][1], segs[j][1])
        segs[min(j, k)] = [lo, hi]
        del segs[max(j, k)]
    return [tuple(x) for x in segs]


def _unstable(x: np.ndarray, fps: int) -> bool:
    """짧은 조각의 음정이 한 음에 머물지 않고 움직이거나 두 반음 사이에 걸쳐 있는지.
    경계 프레임(앞뒤 각 20%)은 옆 음이 섞여 있을 수 있어 빼고 봄."""
    cut = len(x) // 5
    if len(x) - 2 * cut >= 3:
        x = x[cut:len(x) - cut]
    med = float(np.median(x))
    if abs(med - np.round(med)) > UNSTABLE_OFF:
        return True
    if len(x) >= 3:
        slope = abs(np.polyfit(np.arange(len(x)) / fps, x, 1)[0])
        if slope > UNSTABLE_SLOPE:
            return True
    return float(np.percentile(x, 90) - np.percentile(x, 10)) > UNSTABLE_SPREAD


def _merge_unstable(m: np.ndarray, segs: list, fps: int, max_len: float = UNSTABLE_MAX) -> list:
    segs = [list(x) for x in segs]
    limit = int(round(max_len * fps))
    while len(segs) > 1:
        cands = [k for k, (a, b) in enumerate(segs) if b - a < limit and _unstable(m[a:b], fps)]
        if not cands:
            break
        k = min(cands, key=lambda i: segs[i][1] - segs[i][0])
        a, b = segs[k]
        x = m[a:b]
        pitch = lambda i: float(np.round(np.median(m[segs[i][0]:segs[i][1]])))  # noqa: E731
        prev_p = pitch(k - 1) if k > 0 else None
        next_p = pitch(k + 1) if k < len(segs) - 1 else None
        if prev_p is not None and next_p is not None:
            moving = float(x[-1] - x[0]) * (next_p - prev_p) > 0 and abs(next_p - prev_p) >= 1
            if moving:
                j = k + 1                                          # 다음 음으로 가는 중 → 다음 음의 시작
            else:
                med = float(np.median(x))
                j = k - 1 if abs(med - prev_p) <= abs(med - next_p) else k + 1
        else:
            j = k - 1 if prev_p is not None else k + 1
        segs[min(j, k)] = [min(segs[j][0], a), max(segs[j][1], b)]
        del segs[max(j, k)]
    return [tuple(x) for x in segs]


def notes_from_f0(f0: np.ndarray, db: Optional[np.ndarray] = None, gate_db=-60.0, tuning: float = 0.0,
                  **kw) -> list:
    """f0(Hz, 10ms 간격, 0 = 무성) → 음표 목록."""
    return notes_from_pitch(hz_to_midi(f0), db, gate_db, tuning, **kw)


def notes_from_pitch(midi: np.ndarray, db: Optional[np.ndarray] = None, gate_db=-60.0, tuning: float = 0.0,
                     fps: int = FPS, min_note: float = MIN_NOTE, max_gap: float = MAX_GAP, smooth: float = SMOOTH,
                     hysteresis: float = HYSTERESIS) -> list:
    """음정(MIDI 실수, NaN = 무성, 10ms 간격) → 음표 목록.
    gate_db: 숫자 하나 또는 프레임별 기준 (db 가 이보다 작으면 무성)."""
    pm = np.asarray(midi, dtype=np.float64)
    n = len(pm)
    if n == 0:
        return []
    db = np.zeros(n) if db is None else np.asarray(db, dtype=np.float64)[:n]
    if len(db) < n:
        db = np.pad(db, (0, n - len(db)), constant_values=-120.0)
    gate = np.broadcast_to(np.asarray(gate_db, dtype=np.float64), (n,))
    voiced = np.isfinite(pm) & (db >= gate)
    m = np.where(voiced, pm, np.nan) - tuning

    # 짧은 끊김 이어 붙이기 (앞뒤 음 높이가 1반음 안쪽일 때만)
    gap = int(round(max_gap * fps))
    runs: list = []
    for a, b in _runs(voiced):
        if runs and a - runs[-1][1] <= gap and abs(m[a] - m[runs[-1][1] - 1]) < 1.0:
            pa = runs[-1][1] - 1
            m[pa + 1:a] = np.linspace(m[pa], m[a], a - pa + 1)[1:-1]
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))

    min_len = max(1, int(round(min_note * fps)))
    width = max(1, int(round(smooth * fps)) | 1)
    raw = []
    for a, b in runs:
        if b - a < min_len:
            continue
        seg = m[a:b]
        segs = _merge_unstable(seg, _merge_short(seg, _segment(seg, width, hysteresis), min_len), fps)
        for x, y in segs:
            raw.append([a + x, a + y, int(np.round(np.median(seg[x:y])))])
    # 붙어 있는 같은 높이 음은 하나로
    merged: list = []
    for a, b, p in raw:
        if merged and merged[-1][1] == a and merged[-1][2] == p:
            merged[-1][1] = b
        else:
            merged.append([a, b, p])
    if not merged:
        return []
    ref = float(np.percentile(db[voiced], 95)) if voiced.any() else 0.0
    out = []
    for a, b, p in merged:
        if not PITCH_RANGE[0] <= p <= PITCH_RANGE[1]:
            continue
        vel = int(np.clip(np.round(100 + 1.5 * (float(np.median(db[a:b])) - ref)), 25, 120))
        out.append(Note(a / fps, b / fps, p, vel))
    return out


def harmonic_mask(f0: np.ndarray, freqs: np.ndarray, rel: float = 0.03, max_frac: Optional[float] = None,
                  block: int = 2048) -> np.ndarray:
    """(주파수, 프레임) 마스크 0~1. 각 프레임 f0 의 배음 h·f0 둘레 ±max(2칸, rel·h·f0) 는 1, 바깥 한 칸에 걸쳐 0 으로.
    max_frac: 폭 상한 (f0 의 비율, 배음 사이 틈을 남겨 다른 목소리 자리를 지킴). f0 ≤ 0 인 프레임은 0."""
    f0 = np.asarray(f0, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    df = float(freqs[1] - freqs[0])
    out = np.zeros((len(freqs), len(f0)), np.float32)
    idx = np.nonzero(f0 > 0)[0]
    F = freqs[:, None]
    for s in range(0, len(idx), block):
        cols = idx[s:s + block]
        f = f0[cols][None, :]
        h = np.maximum(1.0, np.round(F / f))
        d = np.abs(F - h * f)
        half = np.maximum(2.0 * df, rel * h * f)
        if max_frac is not None:
            half = np.minimum(half, np.maximum(df, max_frac * f))
        out[:, cols] = np.clip((half + df - d) / df, 0.0, 1.0)
    return out


def transpose(notes: Iterable[Note], semitones: int) -> list:
    return [Note(n.start, n.end, n.pitch + semitones, n.velocity) for n in notes
            if 0 <= n.pitch + semitones <= 127]


# ------------------------------------------------------------------ MIDI ---

def _vlq(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def midi_bytes(notes: Sequence[Note], name: str = "Vocal", tempo_bpm: float = TEMPO_BPM, ppq: int = PPQ,
               program: int = CHOIR_AAHS, channel: int = 0) -> bytes:
    """표준 MIDI 파일(형식 0) 한 트랙. 시각(초)은 템포에 맞춰 틱으로 그대로 변환."""
    tps = ppq * tempo_bpm / 60.0
    events = []
    for nt in notes:
        a = int(round(nt.start * tps))
        b = max(a + 1, int(round(nt.end * tps)))
        events.append((a, 1, bytes([0x90 | channel, nt.pitch & 0x7F, max(1, min(127, nt.velocity))])))
        events.append((b, 0, bytes([0x80 | channel, nt.pitch & 0x7F, 0])))
    events.sort(key=lambda e: (e[0], e[1]))
    trk = bytearray()
    label = name.encode("ascii", "replace")
    trk += b"\x00\xff\x03" + _vlq(len(label)) + label
    trk += b"\x00\xff\x51\x03" + int(round(60_000_000 / tempo_bpm)).to_bytes(3, "big")
    trk += b"\x00\xff\x58\x04\x04\x02\x18\x08"                      # 4/4
    trk += bytes([0x00, 0xC0 | channel, program & 0x7F])
    last = 0
    for t, _, data in events:
        trk += _vlq(t - last) + data
        last = t
    trk += b"\x00\xff\x2f\x00"
    head = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + ppq.to_bytes(2, "big")
    return head + b"MTrk" + len(trk).to_bytes(4, "big") + bytes(trk)


def write_midi(path: Path, notes: Sequence[Note], name: str = "Vocal", **kw) -> Path:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(midi_bytes(notes, name, **kw))
    tmp.replace(path)
    return path


def read_midi_notes(data: bytes) -> tuple:
    """(검사용) 형식 0/1 MIDI → ([Note], 템포 BPM, PPQ). 템포 변경은 첫 값만 사용."""
    if data[:4] != b"MThd":
        raise ValueError("MIDI 파일이 아닙니다")
    ppq = int.from_bytes(data[12:14], "big")
    pos, tempo, notes = 14, 500000, []
    while pos < len(data):
        tag, ln = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "big")
        body, pos = data[pos + 8:pos + 8 + ln], pos + 8 + ln
        if tag != b"MTrk":
            continue
        i, tick, status, on = 0, 0, 0, {}
        while i < len(body):
            v = 0
            while True:
                c = body[i]
                i += 1
                v = (v << 7) | (c & 0x7F)
                if not c & 0x80:
                    break
            tick += v
            if body[i] & 0x80:
                status = body[i]
                i += 1
            if status == 0xFF:
                typ = body[i]
                ln2, i = body[i + 1], i + 2
                if typ == 0x51:
                    tempo = int.from_bytes(body[i:i + 3], "big")
                i += ln2
                continue
            hi = status & 0xF0
            if hi in (0xC0, 0xD0):
                i += 1
                continue
            d1, d2 = body[i], body[i + 1]
            i += 2
            if hi == 0x90 and d2 > 0:
                on[d1] = (tick, d2)
            elif hi in (0x80, 0x90) and d1 in on:
                t0, vel = on.pop(d1)
                notes.append((t0, tick, d1, vel))
    sec = tempo / 1e6 / ppq
    return ([Note(a * sec, b * sec, p, v) for a, b, p, v in sorted(notes)], 60e6 / tempo, ppq)
