"""핵심 로직 테스트 (오디오 장치·GPU·인터넷 없이 실행 가능).

실행: pytest -q tests
"""
from __future__ import annotations

import functools
import http.server
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
    for fmt in ("wav16", "wav24", "wav32f", "flac", "mp3"):
        p = tmp_path / f"t_{fmt}{audio_io.EXPORT_FORMATS[fmt][1]}"
        audio_io.write_audio(p, x, fmt)
        y = audio_io.decode_audio(p)
        assert len(y) == len(x), fmt          # MP3 도 인코더 지연 없이 같은 길이
        if fmt != "mp3":
            tol = {"wav16": 1e-4, "wav24": 1e-6, "flac": 1e-6, "wav32f": 1e-7}[fmt]
            ref = x if fmt == "wav32f" else np.clip(x, -1, 1)
            assert np.abs(y - ref).max() < tol, fmt
    assert abs(audio_io.probe_duration(tmp_path / "t_wav16.wav") - 3.0) < 0.05


def test_float_wav_keeps_overs_and_rates(tmp_path):
    from music_separator import audio_io

    x = signal(1.0) * 3.0                       # 0dBFS 를 크게 넘는 신호
    p = tmp_path / "f.wav"
    audio_io.write_wav_float(p, x, 48000)
    assert audio_io.probe_sample_rate(p) == 48000
    y = audio_io.decode_audio(p, sample_rate=48000)
    assert np.abs(y - x).max() < 1e-6 and np.abs(y).max() > 1.5


def test_resample_roundtrip():
    from music_separator import audio_io

    t = np.arange(44100 * 2) / 44100
    x = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2, 1).astype(np.float32) * 0.5
    y = audio_io.resample(audio_io.resample(x, 44100, 48000), 48000, 44100)
    assert len(y) == len(x)
    mid = slice(2000, -2000)
    assert np.abs(y[mid] - x[mid]).max() < 1e-3


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

@pytest.mark.parametrize("eng,mode", [("demucs", "2"), ("demucs", "4"), ("demucs", "6"),
                                      ("roformer", "2"), ("roformer", "4"), ("roformer", "6")])
def test_separate_unittest_model(tmp_path, eng, mode):
    from music_separator import engine

    x = signal(6.0)
    override = "roformer_unittest" if eng == "roformer" else "demucs_unittest"
    res = engine.separate(x, mode, tmp_path / "out", engine=eng, model_name=override)
    assert [s.key for s in res.stems] == list(engine.MODES[mode].outputs)
    total = 0
    for s in res.stems:
        a = np.load(s.path)
        assert a.shape == x.shape and np.isfinite(a).all()
        total = total + a
    assert np.abs(total - x).max() < 1e-5     # 모든 트랙의 합 = 원본 (소리가 사라지지 않음)


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
    assert np.allclose(s2["instrumental"], x - s2["vocals"], atol=1e-6)        # 반주 = 원본 − 보컬
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


# ---------------------------------------------------------------- roformer ---

def test_roformer_architecture_shapes():
    """구조가 BS-RoFormer SW 체크포인트와 같은 파라미터 이름/모양을 갖는지 (실제 가중치 없이)."""
    import torch

    from music_separator.roformer import SW_CONFIG, BSRoformer

    m = BSRoformer(**SW_CONFIG)
    sd = m.state_dict()
    assert len(sd) == 1939
    assert tuple(sd["layers.0.0.layers.0.0.to_qkv.weight"].shape) == (1536, 256)
    assert tuple(sd["mask_estimators.5.to_freqs.0.0.2.weight"].shape) == (16, 1024)
    assert abs(sum(v.numel() for v in sd.values()) / 1e6 - 174.66) < 0.1
    with torch.no_grad():
        small = BSRoformer(**dict(SW_CONFIG, dim=16, depth=1, dim_head=8, heads=2)).eval()
        y = small(torch.randn(1, 2, 44100))
    assert y.shape == (1, 6, 2, 44100)


def test_roformer_demix_overlap_add():
    """겹침 처리 경계에서 이어 붙이기가 맞는지: 항등 모델이면 출력 = 입력."""
    import torch

    from music_separator import roformer

    class Identity(torch.nn.Module):
        num_stems = 1

        def forward(self, x):
            return x[:, None]

    x = signal(9.0)
    out = roformer.demix(Identity(), x, "cpu", num_overlap=2, chunk=44100 * 2)
    assert out.shape == (1,) + x.shape
    assert np.abs(out[0] - x).max() < 1e-5


# ------------------------------------------------------------------ export ---

def _export_fixture(tmp_path, native_rate=48000, gain=1.0):
    from music_separator import audio_io
    from music_separator.export import ExportSource

    sr = native_rate
    t = np.arange(sr * 4) / sr
    comp = {"vocals": 0.3 * np.sin(2 * np.pi * 440 * t), "drums": 0.3 * np.sin(2 * np.pi * 3 * t) ** 9,
            "bass": 0.3 * np.sin(2 * np.pi * 60 * t)}
    comp = {k: (np.stack([v, v * 0.9], 1) * gain).astype(np.float32) for k, v in comp.items()}
    native = sum(comp.values())
    np.save(tmp_path / "native.npy", native)
    mix44 = audio_io.resample(native, sr, 44100)
    np.save(tmp_path / "mix44.npy", mix44)
    v = audio_io.resample(comp["vocals"], sr, 44100)
    d = audio_io.resample(comp["drums"], sr, 44100)
    np.save(tmp_path / "vocals.npy", v)
    np.save(tmp_path / "drums.npy", d)
    np.save(tmp_path / "other.npy", mix44 - v - d)
    stems = {k: tmp_path / f"{k}.npy" for k in ("vocals", "drums", "other")}
    return ExportSource(stems, "other", tmp_path / "mix44.npy", tmp_path / "native.npy", sr), comp, native


def test_export_full_mix_is_lossless_at_native_rate(tmp_path):
    from music_separator import audio_io
    from music_separator.export import Exporter, ExportJob, ExportOptions

    src, comp, native = _export_fixture(tmp_path)
    rep = Exporter(src, ExportOptions(), tmp_path).run(
        [ExportJob(tmp_path / "full.wav", "wav32f", "full", "mr", excluded=[], gains={})])
    assert rep.sample_rate == 48000
    y = audio_io.decode_audio(tmp_path / "full.wav", sample_rate=48000)
    assert np.abs(y - native).max() == 0.0


def test_export_mr_is_original_minus_removed(tmp_path):
    from music_separator import audio_io
    from music_separator.export import Exporter, ExportJob, ExportOptions

    src, comp, native = _export_fixture(tmp_path)
    Exporter(src, ExportOptions(), tmp_path).run(
        [ExportJob(tmp_path / "mr.wav", "wav32f", "mr", "mr", excluded=["vocals"], gains={})])
    y = audio_io.decode_audio(tmp_path / "mr.wav", sample_rate=48000)
    truth = comp["drums"] + comp["bass"]
    snr = 10 * np.log10((truth ** 2).sum() / ((y - truth) ** 2).sum())
    assert snr > 60, snr


def test_export_volume_gains_and_clipping_guard(tmp_path):
    from music_separator import audio_io
    from music_separator.export import Exporter, ExportJob, ExportOptions

    src, comp, native = _export_fixture(tmp_path, gain=3.0)     # 0dBFS 초과
    rep = Exporter(src, ExportOptions(keep_native_rate=False), tmp_path).run(
        [ExportJob(tmp_path / "a.wav", "wav24", "a", "mr", excluded=[], gains={"drums": 0.5})])
    assert rep.sample_rate == 44100 and rep.gain_db < 0
    y = audio_io.decode_audio(tmp_path / "a.wav")
    assert np.abs(y).max() <= 1.0
    rep = Exporter(src, ExportOptions(keep_native_rate=False), tmp_path).run(
        [ExportJob(tmp_path / "b.wav", "wav32f", "b", "mr", excluded=[], gains={"drums": 0.5})])
    assert rep.gain_db == 0.0


def test_export_key_shift_keeps_drums(tmp_path):
    from music_separator import audio_io
    from music_separator.export import Exporter, ExportJob, ExportOptions

    src, comp, native = _export_fixture(tmp_path)
    Exporter(src, ExportOptions(semitones=12, keep_drums_pitch=True), tmp_path).run([
        ExportJob(tmp_path / "d.wav", "wav32f", "d", "stem", key="drums"),
        ExportJob(tmp_path / "v.wav", "wav32f", "v", "stem", key="vocals")])
    d = audio_io.decode_audio(tmp_path / "d.wav", sample_rate=48000)
    truth = comp["drums"]
    assert 10 * np.log10((truth ** 2).sum() / ((d - truth) ** 2).sum()) > 40   # 드럼은 그대로
    v = audio_io.decode_audio(tmp_path / "v.wav", sample_rate=48000)[48000:96000, 0]
    f = np.fft.rfftfreq(len(v), 1 / 48000)[np.argmax(np.abs(np.fft.rfft(v * np.hanning(len(v)))))]
    assert abs(f - 880) < 3                                                  # 보컬은 한 옥타브 위


def test_pitch_shift_accuracy():
    from music_separator import pitch

    sr = 44100
    t = np.arange(sr * 2) / sr
    x = np.stack([0.4 * np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
    assert pitch.shift(x, 0) is not None and np.array_equal(pitch.shift(x, 0), x)
    for semis in (-12, -5, 7):
        y = pitch.shift(x, semis, sr)
        assert len(y) == len(x)
        seg = y[sr // 2: sr + sr // 2, 0]
        f = np.fft.rfftfreq(len(seg), 1 / sr)[np.argmax(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))))]
        assert abs(f - 440 * 2 ** (semis / 12)) < 3, (semis, f)


def test_model_import_raw_checkpoint(tmp_path, monkeypatch):
    """BS-Roformer-SW.ckpt 파일만 있는 폴더도 '모델 가져오기'로 설치되어야 함."""
    from music_separator import models

    spec = models.MODELS["bs_roformer_sw"]
    src = tmp_path / "dl"
    src.mkdir()
    (src / spec.file_name).write_bytes(b"0" * (spec.approx_mb * 1024 * 1024))
    assert models.copy_into_user_dir(src) == ["bs_roformer_sw"]
    assert models.is_installed("bs_roformer_sw")
