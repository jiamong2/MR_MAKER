"""템포 조절 (빠르기 배율) 과 원곡 BPM 추정 (참고 표시용)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from music_separator import pitch, tempo  # noqa: E402

SR = 44100


def drums(bpm, sec=24.0, hats=True, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    x = np.zeros(n)
    kick = np.sin(2 * np.pi * 60 * np.arange(int(0.15 * SR)) / SR) * np.exp(-np.arange(int(0.15 * SR)) / SR * 25)
    snare = rng.normal(0, 1, int(0.12 * SR)) * np.exp(-np.arange(int(0.12 * SR)) / SR * 30) * 0.6
    hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 120) * 0.2
    period = 60 / bpm
    for b in range(int(sec / period) - 1):
        i = int(b * period * SR)
        w = kick if b % 2 == 0 else snare
        x[i:i + len(w)] += w
        if hats:
            x[i:i + len(hat)] += hat
            j = int((b + 0.5) * period * SR)
            x[j:j + len(hat)] += hat
    x += rng.normal(0, 0.02, n)
    return (np.stack([x, x], 1) * 0.3).astype(np.float32)


def octave_ok(est, bpm, tol=0.04):
    return min(abs(np.log2(est / (bpm * r))) for r in (0.5, 1.0, 2.0)) < np.log2(1 + tol)


@pytest.mark.parametrize("bpm", [72, 96, 120, 145])
def test_bpm_estimate_on_drum_loops(bpm):
    e = tempo.estimate(drums(bpm), SR)
    assert e.reliable and octave_ok(e.bpm, bpm), e
    assert 70 <= e.bpm <= 175                       # 사람들이 흔히 느끼는 범위 쪽으로


def test_bpm_unknown_without_beats():
    t = np.arange(20 * SR) / SR
    pad = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, 1).astype(np.float32)
    assert not tempo.estimate(pad, SR).reliable
    noise = np.random.default_rng(1).normal(0, 0.1, (20 * SR, 2)).astype(np.float32)
    assert not tempo.estimate(noise, SR).reliable
    assert not tempo.estimate(np.zeros((SR, 2), np.float32), SR).reliable    # 너무 짧음


def test_bpm_follows_tempo_change():
    """원곡을 90% 로 늦추면 추정 BPM 도 90% (배율로만 바꾸므로 흔들림도 그대로 비례)."""
    x = drums(110)
    y = pitch._shift_signalsmith(x, 0, SR, 0.9)
    a, b = tempo.estimate(x, SR), tempo.estimate(y, SR)
    assert abs(b.bpm / a.bpm - 0.9) < 0.02


def test_percent_conversion():
    assert tempo.percent_for(100.0, 90.0) == 90
    assert tempo.percent_for(92.0, 83.0) == 90
    assert tempo.percent_for(100.0, 200.0) is None and tempo.percent_for(0, 90) is None
    assert tempo.target_bpm(92.0, 90) == pytest.approx(82.8)


def test_tempo_shift_length_and_pitch():
    """길이는 1/배율, 음높이는 그대로 (기본 엔진이 없으면 대체 엔진으로)."""
    t = np.arange(3 * SR) / SR
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, 1).astype(np.float32)

    def f0(y):
        seg = y[SR // 2:SR // 2 + 32768, 0] * np.hanning(32768)
        return np.argmax(np.abs(np.fft.rfft(seg, 1 << 18))) * SR / (1 << 18)
    if not pitch.available():
        pytest.skip("키·템포 엔진 없음")
    for tp, semi in ((0.8, 0), (1.25, 0), (0.9, 2)):
        y = pitch.shift(x, semi, SR, tempo=tp)
        assert len(y) == pitch.out_frames(len(x), tp) == int(round(len(x) / tp))
        assert abs(f0(y) - 220 * 2 ** (semi / 12)) < 0.6
    assert pitch.shift(x, 0, SR, tempo=1.0) is not None and len(pitch.shift(x, 0, SR)) == len(x)
    with pytest.raises(ValueError):
        pitch.shift(x, 0, SR, tempo=0.5)
    assert pitch.tempo_suffix(90) == " [Tempo 90%]" and pitch.tempo_suffix(100) == ""


def test_export_applies_tempo_to_all_outputs(tmp_path):
    """저장: MR·트랙 모두 같은 배율로 늘어나고, 드럼은 키는 그대로 템포만."""
    from music_separator.export import Exporter, ExportJob, ExportOptions, ExportSource

    if not pitch.available():
        pytest.skip("키·템포 엔진 없음")
    n = 2 * SR
    t = np.arange(n) / SR
    voc = np.stack([0.2 * np.sin(2 * np.pi * 330 * t)] * 2, 1).astype(np.float32)
    inst = np.stack([0.2 * np.sin(2 * np.pi * 110 * t)] * 2, 1).astype(np.float32)
    paths = {}
    for k, a in (("vocals", voc), ("instrumental", inst), ("mix", voc + inst)):
        paths[k] = tmp_path / f"{k}.npy"
        np.save(paths[k], a)
    src = ExportSource(stems={"vocals": paths["vocals"], "instrumental": paths["instrumental"]},
                       residual_key="instrumental", mix44_path=paths["mix"], native_path=paths["mix"])
    jobs = [ExportJob(tmp_path / "mr.wav", "wav32f", "MR", "mr", excluded=["vocals"]),
            ExportJob(tmp_path / "v.wav", "wav32f", "보컬", "stem", key="vocals")]
    rep = Exporter(src, ExportOptions(tempo=0.8), tmp_path).run(jobs)
    for d in rep.details:
        assert abs(d.seconds - 2 / 0.8) < 0.01
