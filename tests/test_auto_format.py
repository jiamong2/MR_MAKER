"""저장 형식 '원본에 맞춤': 원본 음질 판단과 그에 맞춘 형식."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_separator import audio_io as a  # noqa: E402

SR = 44100


def _music(seconds=12, seed=0):
    t = np.arange(SR * seconds) / SR
    rng = np.random.default_rng(seed)
    x = 0.2 * sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330, 440, 880, 1760)) / 6
    x = x + 0.05 * rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    return np.stack([x, 0.9 * x], 1).astype(np.float32)


def _ff(*args):
    subprocess.run([a.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def _assess(p):
    info = a.probe_audio(p)
    return a.assess_quality(info, a.decode_audio(p, None), info.sample_rate)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("q")
    x = _music()
    a.write_wav(d / "cd.wav", x, 16, SR)
    a.write_wav(d / "hi.wav", x, 24, SR)
    _ff("-i", d / "cd.wav", "-c:a", "flac", d / "cd.flac")
    _ff("-i", d / "cd.wav", "-b:a", "128k", d / "m128.mp3")
    _ff("-i", d / "m128.mp3", "-c:a", "pcm_s16le", d / "fake128.wav")
    _ff("-i", d / "cd.wav", "-b:a", "320k", d / "m320.mp3")
    _ff("-i", d / "m320.mp3", "-c:a", "flac", d / "fake320.flac")
    _ff("-i", d / "cd.wav", "-c:a", "aac", "-b:a", "128k", d / "yt.m4a")
    # 고음이 완만하게 줄어드는 실제 녹음 같은 CD (손실 압축으로 잘못 보면 안 됨)
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / SR)[:, None]
    soft = np.fft.irfft(X / (1 + (f / 9000.0) ** 2), len(x), axis=0).astype(np.float32)
    a.write_wav(d / "soft.wav", soft, 16, SR)
    return d


@pytest.mark.parametrize("name,kind,fmt", [
    ("cd.wav", "cd", "flac16"), ("cd.flac", "cd", "flac16"), ("soft.wav", "cd", "flac16"),
    ("hi.wav", "hires", "flac"),
    ("m128.mp3", "lossy", "mp3"), ("yt.m4a", "lossy", "mp3"),
    ("fake128.wav", "lossy", "mp3"), ("fake320.flac", "lossy", "mp3"),
])
def test_quality_and_auto_format(files, name, kind, fmt):
    q = _assess(files / name)
    assert (q.kind, q.fmt) == (kind, fmt), q.describe()
    assert a.resolve_format("auto", q) == fmt
    assert a.resolve_format("wav24", q) == "wav24"          # 직접 고른 형식은 그대로


def test_fake_lossless_has_reason(files):
    q = _assess(files / "fake128.wav")
    assert "손실 압축을 풀어" in q.detail and "MP3 320kbps" in q.describe()


def test_unknown_quality_is_safe():
    assert a.resolve_format("auto", None) == "flac"
    assert a.assess_quality(a.AudioInfo()).fmt == "flac"


def test_flac16_matches_wav16(tmp_path):
    x = _music(4)
    a.write_audio(tmp_path / "o.flac", x, "flac16", SR)
    a.write_audio(tmp_path / "o.wav", x, "wav16", SR)
    info = a.probe_audio(tmp_path / "o.flac")
    assert info.codec == "flac" and a.source_bits(info) == 16
    assert np.array_equal(a.decode_audio(tmp_path / "o.flac", None), a.decode_audio(tmp_path / "o.wav", None))


def test_auto_is_default_and_sizes(tmp_path):
    assert a.DEFAULT_FORMAT == "auto" and "auto" in a.EXPORT_FORMATS
    x = _music(10)
    sizes = {}
    for fmt in ("wav24", "flac16", "mp3"):
        p = tmp_path / f"o{a.EXPORT_FORMATS[fmt][1]}"
        a.write_audio(p, x, fmt, SR)
        sizes[fmt] = p.stat().st_size
    assert sizes["mp3"] * 3 < sizes["wav24"] and sizes["flac16"] < sizes["wav24"]
