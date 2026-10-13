"""numpy 기반 신호 처리 도우미 (파형 피크, 믹스다운)."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

PEAK_BUCKETS = 3000


def compute_peaks(data: np.ndarray, buckets: int = PEAK_BUCKETS, block: int = 1 << 20) -> np.ndarray:
    """(n, ch) 배열 → (buckets, 2) [min, max] 모노 피크. memmap 도 블록 단위로 처리."""
    n = len(data)
    if n == 0:
        return np.zeros((buckets, 2), dtype=np.float32)
    buckets = max(1, min(buckets, n))
    edges = np.linspace(0, n, buckets + 1).astype(np.int64)
    out = np.zeros((buckets, 2), dtype=np.float32)
    # 버킷 단위로 블록 처리
    b = 0
    while b < buckets:
        start = edges[b]
        # block 크기만큼 버킷 묶기
        e = int(np.searchsorted(edges, start + block, side="right")) - 1
        e = max(e, b + 1)
        e = min(e, buckets)
        seg = np.asarray(data[start:edges[e]], dtype=np.float32)
        mono = seg.mean(axis=1) if seg.ndim == 2 else seg
        local = edges[b:e + 1] - start
        mins = np.minimum.reduceat(mono, local[:-1])
        maxs = np.maximum.reduceat(mono, local[:-1])
        # 빈 버킷(길이 0) 보정
        empty = local[1:] == local[:-1]
        mins[empty] = 0
        maxs[empty] = 0
        out[b:e, 0] = mins
        out[b:e, 1] = maxs
        b = e
    return out


def mixdown(tracks: Sequence[np.ndarray], gains: Sequence[float], block: int = 44100 * 20,
            out: Optional[np.ndarray] = None) -> np.ndarray:
    """여러 (n, ch) 트랙을 게인 적용해 합산. 클리핑이 생기면 피크 정규화(-0.3dBFS)."""
    if not tracks:
        raise ValueError("no tracks")
    n = min(len(t) for t in tracks)
    ch = tracks[0].shape[1]
    if out is None:
        out = np.zeros((n, ch), dtype=np.float32)
    for start in range(0, n, block):
        end = min(n, start + block)
        acc = np.zeros((end - start, ch), dtype=np.float32)
        for t, g in zip(tracks, gains):
            if g != 0:
                acc += np.asarray(t[start:end], dtype=np.float32) * np.float32(g)
        out[start:end] = acc
    peak = float(np.max(np.abs(out))) if n else 0.0
    limit = 10 ** (-0.3 / 20)
    if peak > limit:
        out *= np.float32(limit / peak)
    return out


def audible_flags(mutes: Iterable[bool], solos: Iterable[bool]) -> list[bool]:
    mutes, solos = list(mutes), list(solos)
    any_solo = any(solos)
    return [(not m) and (s or not any_solo) for m, s in zip(mutes, solos)]
