"""핵심 로직 테스트 (오디오 장치·GPU·인터넷 없이 실행 가능).

실행: pytest -q tests
"""
from __future__ import annotations

import functools
import http.server
import os
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_SEPARATOR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MUSIC_SEPARATOR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MUSIC_SEPARATOR_MODELS_DIR", raising=False)
    from music_separator import models

    models.clear_cache()
    yield


def signal(sec=6.0):
    from music_separator.selftest import make_test_signal

    return make_test_signal(sec)


# ------------------------------------------------------------------ dsp ---

def test_peaks_match_naive():
    from music_separator.dsp import compute_peaks

    x = np.random.default_rng(1).standard_normal((100_003, 2)).astype(np.float32)
    p = compute_peaks(x, 777, block=10_000)
    edges = np.linspace(0, len(x), 778).astype(int)
    mono = x.mean(1)
    naive = np.array([[mono[a:b].min(), mono[a:b].max()] for a, b in zip(edges[:-1], edges[1:])])
    assert np.allclose(p, naive)


def test_mixdown_normalizes_clipping():
    from music_separator.dsp import mixdown

    a = np.full((1000, 2), 0.8, np.float32)
    out = mixdown([a, a], [1.0, 1.0])
    assert np.abs(out).max() <= 10 ** (-0.3 / 20) + 1e-6
    out2 = mixdown([a, a], [0.5, 0.1])
    assert np.allclose(out2, 0.8 * 0.6)


def test_audible_flags():
    from music_separator.dsp import audible_flags

    assert audible_flags([False, True, False], [False, False, False]) == [True, False, True]
    assert audible_flags([False, False, False], [True, False, False]) == [True, False, False]
    assert audible_flags([True, False], [True, True]) == [False, True]


# ---------------------------------------------------------------- mixer ---

def test_mixer_mute_solo_gain_and_ramp():
    from music_separator.player import Mixer, Track

    n = 44100
    a = np.full((n, 2), 0.2, np.float32)
    b = np.full((n, 2), 0.1, np.float32)
    m = Mixer()
    m.master = 1.0
    m.set_tracks([Track("a", a), Track("b", b)])
    out, ended = m.render(512)
    assert np.allclose(out, 0.3) and not ended
    m.tracks[1].mute = True
    out, _ = m.render(512)                     # 램프 구간
    assert out[0, 0] > out[-1, 0] and np.isclose(out[-1, 0], 0.2, atol=1e-3)
    out, _ = m.render(512)
    assert np.allclose(out, 0.2)
    m.tracks[1].mute = False
    m.tracks[1].solo = True
    m.render(512)
    out, _ = m.render(512)
    assert np.allclose(out, 0.1)
    m.tracks[1].gain = 0.5
    m.render(512)
    out, _ = m.render(512)
    assert np.allclose(out, 0.05)
    m.seek(n - 100)
    out, ended = m.render(512)
    assert ended and np.allclose(out[100:], 0)


# ------------------------------------------------------------- audio io ---

def test_audio_roundtrip_formats(tmp_path):
    from music_separator import audio_io

    x = signal(3.0)
    for fmt in ("wav16", "wav24", "mp3"):
        p = tmp_path / f"t_{fmt}{audio_io.EXPORT_FORMATS[fmt][1]}"
        audio_io.write_audio(p, x, fmt)
        y = audio_io.decode_audio(p)
        assert abs(len(y) - len(x)) < 44100 * 0.2
        if fmt.startswith("wav"):
            tol = 1e-4 if fmt == "wav16" else 1e-6
            assert np.abs(y - np.clip(x, -1, 1)).max() < tol
    assert abs(audio_io.probe_duration(tmp_path / "t_wav16.wav") - 3.0) < 0.05


def test_decode_errors(tmp_path):
    from music_separator import audio_io

    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all" * 10)
    with pytest.raises(audio_io.AudioError):
        audio_io.decode_audio(bad)
    with pytest.raises(audio_io.AudioError):
        audio_io.decode_audio(tmp_path / "missing.wav")


def test_mono_input_becomes_stereo(tmp_path):
    import wave

    from music_separator import audio_io

    p = tmp_path / "mono.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes((np.sin(np.arange(22050) / 10) * 10000).astype("<i2").tobytes())
    y = audio_io.decode_audio(p)
    assert y.shape[1] == 2 and abs(len(y) - 44100) < 10


def test_safe_filename():
    from music_separator.audio_io import safe_filename

    assert safe_filename('a/b:c*?"<>|.') == "a_b_c______"
    assert safe_filename("노래 - 보컬 (Vocals).wav") == "노래 - 보컬 (Vocals).wav"


# --------------------------------------------------------------- engine ---

@pytest.mark.parametrize("mode", ["2", "4"])
def test_separate_unittest_model(tmp_path, mode):
    from music_separator import engine

    x = signal(8.0)
    res = engine.separate(x, mode, tmp_path / "out", model_name="demucs_unittest")
    assert [s.key for s in res.stems] == list(engine.MODES[mode].outputs)
    for s in res.stems:
        a = np.load(s.path)
        assert a.shape == x.shape and np.isfinite(a).all()


def test_two_stem_instrumental_is_sum_of_others(tmp_path):
    from music_separator import engine

    import random

    x = signal(5.0)
    random.seed(0)  # apply_model 의 무작위 shift 고정
    r4 = engine.separate(x, "4", tmp_path / "a", model_name="demucs_unittest")
    random.seed(0)
    r2 = engine.separate(x, "2", tmp_path / "b", model_name="demucs_unittest")
    s4 = {s.key: np.load(s.path) for s in r4.stems}
    s2 = {s.key: np.load(s.path) for s in r2.stems}
    assert np.allclose(s2["vocals"], s4["vocals"], atol=1e-5)
    assert np.allclose(s2["instrumental"], s4["drums"] + s4["bass"] + s4["other"], atol=1e-4)


def test_cancel(tmp_path):
    from music_separator import engine

    ev = threading.Event()

    def prog(f, msg):
        if f > 0.2:
            ev.set()

    with pytest.raises(engine.SeparationCancelled):
        engine.separate(signal(60.0), "4", tmp_path, model_name="demucs_unittest", progress=prog, cancel=ev)


def test_gpu_failure_falls_back_to_cpu(tmp_path, monkeypatch):
    """존재하지 않는 장치(가짜 GPU 실패)를 지정해도 CPU 로 재처리되어 성공해야 함."""
    from music_separator import engine

    notices = []
    res = engine.separate(signal(4.0), "4", tmp_path, model_name="demucs_unittest", device="meta",
                          notice=notices.append)
    assert res.device == "cpu" and notices


def test_friendly_errors():
    import errno

    from music_separator import engine

    assert "메모리" in engine.friendly_error(MemoryError())
    assert "메모리" in engine.friendly_error(RuntimeError("CUDA out of memory. Tried"))
    assert "디스크" in engine.friendly_error(OSError(errno.ENOSPC, "No space"))


# --------------------------------------------------------------- models ---

def test_model_loading_both_formats(tmp_path, monkeypatch):
    import fake_models

    from music_separator import engine, models

    for fmt in ("safetensors", "th"):
        root = tmp_path / fmt
        fake_models.make(root, "htdemucs", fmt)
        monkeypatch.setenv("MUSIC_SEPARATOR_MODELS_DIR", str(root))
        models.clear_cache()
        assert models.is_installed("htdemucs")
        m = models.load_model("htdemucs")
        assert list(m.sources) == ["drums", "bass", "other", "vocals"]
    res = engine.separate(signal(3.0), "4", tmp_path / "o", model_name="htdemucs")
    assert len(res.stems) == 4


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture
def http_root(tmp_path, monkeypatch):
    root = tmp_path / "srv"
    root.mkdir()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root)))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_download_from_hf_mirror(http_root, tmp_path, monkeypatch):
    import fake_models

    from music_separator import models

    root, url = http_root
    fake_models.make(tmp_path / "src", "htdemucs", "safetensors")
    d = root / "adefossez" / "HTDemucs" / "resolve" / "main"
    d.mkdir(parents=True)
    shutil.copy(tmp_path / "src" / "htdemucs" / "955717e8.safetensors", d)
    monkeypatch.setenv("HF_ENDPOINT", url)
    seen = []
    folder = models.download_model("htdemucs", lambda a, b, l: seen.append(a))
    assert (folder / "955717e8.safetensors").exists() and seen
    assert models.load_model("htdemucs") is not None


def test_download_fallback_and_checksum(http_root, tmp_path, monkeypatch):
    import fake_models

    from music_separator import models

    root, url = http_root
    fake_models.make(tmp_path / "src", "htdemucs", "th")
    shutil.copy(tmp_path / "src" / "htdemucs" / "955717e8.th", root / "955717e8-8726e21a.th")
    monkeypatch.setenv("HF_ENDPOINT", url + "/missing")         # HF 404 → 기존 서버로
    monkeypatch.setenv("MUSIC_SEPARATOR_LEGACY_ROOT", url + "/")
    with pytest.raises(models.ModelError) as ei:                 # 가짜 파일 → 체크섬 불일치
        models.download_model("htdemucs")
    assert "체크섬" in str(ei.value)
    assert not list((models.user_models_dir() / "htdemucs").glob("*.th"))


def test_download_cancel(http_root, tmp_path, monkeypatch):
    import fake_models

    from music_separator import models

    root, url = http_root
    fake_models.make(tmp_path / "src", "htdemucs", "safetensors")
    d = root / "adefossez" / "HTDemucs" / "resolve" / "main"
    d.mkdir(parents=True)
    shutil.copy(tmp_path / "src" / "htdemucs" / "955717e8.safetensors", d)
    monkeypatch.setenv("HF_ENDPOINT", url)
    ev = threading.Event()

    def prog(done, total, label):  # 진행률이 처음 보고되는 즉시 취소 (빠른 PC 에서도 결정적)
        ev.set()
    with pytest.raises(models.DownloadCancelled):
        models.download_model("htdemucs", prog, ev)
    assert not models.is_installed("htdemucs")


def test_requirements_estimate(tmp_path):
    from music_separator import system

    req = system.separation_requirement(44100 * 240, 4, 4, tmp_path)
    assert req.disk_needed > 4 * 44100 * 240 * 8
    assert req.ram_needed > 1.5 * system.GB
