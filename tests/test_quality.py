"""음질·원클릭 파이프라인·YouTube 획득 테스트 (네트워크·GPU 없이 실행).

- 출력 형식: MP3 320kbps CBR, FLAC = WAV 24-bit 샘플 단위 동일, 16-bit 디더
- 리샘플링 품질(soxr VHQ), 원본 샘플레이트 디코딩, 트루 피크
- 키 변경: 0 반음 = 처리 생략(비트 동일), ±2/-3/±12 길이·음정·타이밍
- YouTube: 주소 판별, 스트림 선택 규칙, 오류 메시지, 가짜 yt-dlp 로 전체 파이프라인
- 원클릭 파이프라인: 선택한 결과물만 저장, MR = 원본 − 보컬, 클리핑 방지, 취소, 오류 처리
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SR = 44100


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_SEPARATOR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MUSIC_SEPARATOR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MUSIC_SEPARATOR_MODELS_DIR", raising=False)
    from music_separator import models

    models.clear_cache()
    yield


def song(sec=6.0, sr=SR, gain=0.8):
    from music_separator.selftest import make_test_signal

    return (make_test_signal(sec, sr) * gain).astype(np.float32)


def harmonic_tone(f0, sec=2.0, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    x = sum(0.3 / h * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 7))
    return np.stack([x, x], 1).astype(np.float32)


def f0_of(y, target, sr=SR):
    seg = y[sr // 2: sr // 2 + sr, 0].astype(np.float64)
    n = 16 * len(seg)
    spec = np.abs(np.fft.rfft(seg * np.blackman(len(seg)), n))
    fr = np.fft.rfftfreq(n, 1 / sr)
    band = np.where((fr > target * 0.95) & (fr < target * 1.05))[0]
    return fr[band[np.argmax(spec[band])]]


# ---------------------------------------------------------------- 출력 ---

def test_mp3_is_320kbps_cbr(tmp_path):
    from music_separator import audio_io

    p = tmp_path / "a.mp3"
    audio_io.write_mp3(p, song(5))
    info = audio_io.probe_audio(p)
    assert info.codec == "mp3" and info.bit_rate_kbps == 320
    size_kbps = p.stat().st_size * 8 / 1000 / 5.0
    assert 300 < size_kbps < 340, size_kbps


def test_flac_decodes_identical_to_wav24(tmp_path):
    from music_separator import audio_io

    x = song(4)
    audio_io.write_audio(tmp_path / "a.wav", x, "wav24")
    audio_io.write_audio(tmp_path / "a.flac", x, "flac")
    a = audio_io.decode_audio(tmp_path / "a.wav", None)
    b = audio_io.decode_audio(tmp_path / "a.flac", None)
    assert a.shape == b.shape and np.array_equal(a, b)
    assert audio_io.probe_audio(tmp_path / "a.flac").lossless


def test_wav16_uses_tpdf_dither(tmp_path):
    from music_separator import audio_io

    t = np.arange(SR) / SR
    x = np.stack([0.001 * np.sin(2 * np.pi * 1000 * t)] * 2, 1).astype(np.float32)   # -60dBFS
    audio_io.write_audio(tmp_path / "d.wav", x, "wav16")
    y = audio_io.decode_audio(tmp_path / "d.wav", None)
    err = y - x
    lsb = 1 / 32767
    assert np.sqrt(np.mean(err ** 2)) < 1.0 * lsb          # TPDF 잡음 ≈ 0.41 LSB rms (+ 반올림)
    # 디더가 없으면 오차가 신호와 상관(왜곡) — 디더가 있으면 상관이 거의 없음
    assert abs(np.corrcoef(err[:, 0], x[:, 0])[0, 1]) < 0.1


def test_decode_native_rate_without_resampling(tmp_path):
    from music_separator import audio_io

    x = song(3, 48000)
    audio_io.write_wav(tmp_path / "n.wav", x, 24, 48000)
    y = audio_io.decode_audio(tmp_path / "n.wav", None)
    assert len(y) == len(x)
    assert np.max(np.abs(y - np.clip(x, -1, 1))) < 2 / 8388607
    info = audio_io.probe_audio(tmp_path / "n.wav")
    assert info.sample_rate == 48000 and info.channels == 2 and info.lossless


def test_resampler_quality_soxr_vhq():
    from music_separator import audio_io

    src, dst = 48000, 44100
    t = np.arange(src * 2) / src
    alias = np.stack([0.5 * np.sin(2 * np.pi * 22600 * t)] * 2, 1).astype(np.float32)   # 44.1k 의 나이퀴스트 초과
    y = audio_io.resample(alias, src, dst)
    assert len(y) == int(round(len(alias) * dst / src))
    rms = np.sqrt(np.mean(y[dst // 4: -dst // 4] ** 2))
    assert 20 * np.log10(rms / (0.5 / np.sqrt(2)) + 1e-30) < -100      # FFmpeg 기본 swr 은 약 -15dB
    tone = np.stack([0.5 * np.sin(2 * np.pi * 1000 * t)] * 2, 1).astype(np.float32)
    y = audio_io.resample(tone, src, dst)
    ty = np.arange(len(y)) / dst
    ref = 0.5 * np.sin(2 * np.pi * 1000 * ty)
    err = y[dst // 4:-dst // 4, 0] - ref[dst // 4:-dst // 4]
    assert np.sqrt(np.mean(err ** 2)) < 1e-5                          # 지연·이득 오차 없음


def test_true_peak_finds_intersample_overs():
    from music_separator import audio_io

    n = SR
    t = np.arange(n)
    x = np.sin(2 * np.pi * (SR / 4) * t / SR + np.pi / 4)                # 샘플은 ±0.707, 실제 파형은 ±1
    x = np.stack([x, x], 1).astype(np.float32)
    assert np.max(np.abs(x)) < 0.71
    assert audio_io.true_peak(x, SR) > 0.97


# ---------------------------------------------------------------- 키 ---

def _have_rubberband():
    from music_separator import pitch

    return pitch.find_rbshift() is not None


def test_key_zero_is_bit_exact_bypass(monkeypatch):
    from music_separator import pitch

    def boom(*a, **k):
        raise AssertionError("pitch engine must not run for 0 semitones")
    monkeypatch.setattr(pitch, "_shift_rubberband", boom)
    monkeypatch.setattr(pitch, "_shift_signalsmith", boom)
    x = song(2)
    y = pitch.shift(x, 0, SR)
    assert np.array_equal(x, y)


@pytest.mark.skipif(not _have_rubberband(), reason="rbshift 미빌드 (cmake -S native/rbshift -B build/rbshift)")
@pytest.mark.parametrize("k", [2, -3, 12, -12])
def test_rubberband_key_shift_quality(k):
    from music_separator import pitch

    assert pitch.engine_name() == "Rubber Band R3"
    tone = harmonic_tone(220.0)
    y = pitch.shift(tone, k, SR)
    assert y.shape == tone.shape                                       # 곡 길이 유지
    target = 220.0 * 2 ** (k / 12)
    cents = 1200 * np.log2(f0_of(y, target) / target)
    assert abs(cents) < 3, cents                                       # 음정 정확도
    # BPM 유지: 클릭(타격음) 위치가 그대로
    x = np.zeros((SR * 3, 2), np.float32)
    pos = np.arange(int(0.25 * SR), SR * 3 - SR // 4, SR // 2)
    rng = np.random.default_rng(0)
    for p in pos:
        m = int(0.08 * SR)
        x[p:p + m] += (0.6 * rng.standard_normal(m) * np.exp(-np.arange(m) / (0.01 * SR)))[:, None]
    y = pitch.shift(x, k, SR)
    env = np.abs(y[:, 0])
    found = []
    for p in pos:
        seg = env[p - SR // 40: p + SR // 40]
        found.append(p - SR // 40 + int(np.argmax(seg > 0.3 * seg.max())))
    assert np.max(np.abs(np.array(found) - pos)) < 0.012 * SR           # 12ms 이내
    # 스테레오 특성 유지: 동일 채널 입력 → 동일 채널 출력
    assert np.max(np.abs(y[:, 0] - y[:, 1])) < 1e-4


@pytest.mark.skipif(not _have_rubberband(), reason="rbshift 미빌드")
def test_rubberband_formant_option_runs_and_cancel():
    from music_separator import pitch

    x = song(3)
    y = pitch.shift(x, 3, SR, formant=True)
    assert y.shape == x.shape and np.isfinite(y).all()
    ev = threading.Event()
    ev.set()
    with pytest.raises(pitch.PitchCancelled):
        pitch.shift(x, 3, SR, cancel=ev)


def test_signalsmith_fallback_when_rubberband_missing(monkeypatch):
    from music_separator import pitch

    monkeypatch.setattr(pitch, "find_rbshift", lambda: None)
    assert pitch.engine_name() == "Signalsmith Stretch"
    x = harmonic_tone(440.0)
    y = pitch.shift(x, 2, SR)
    assert y.shape == x.shape


# ------------------------------------------------------------- YouTube ---

def test_youtube_url_parsing():
    from music_separator import youtube

    vid = "dQw4w9WgXcQ"
    for u in (f"https://www.youtube.com/watch?v={vid}", f"youtube.com/watch?v={vid}&list=PLx&index=2",
              f"https://youtu.be/{vid}?t=30", f"https://m.youtube.com/watch?v={vid}",
              f"https://music.youtube.com/watch?v={vid}", f"https://www.youtube.com/shorts/{vid}",
              f"https://www.youtube.com/live/{vid}", f"  https://youtu.be/{vid}  "):
        assert youtube.video_id(u) == vid, u
    for u in ("https://vimeo.com/123456", "https://www.youtube.com/playlist?list=PL123", "hello",
              "https://www.youtube.com/watch?v=short", "", "https://evil.com/watch?v=dQw4w9WgXcQ"):
        assert youtube.video_id(u) is None, u
    assert youtube.canonical_url(f"https://youtu.be/{vid}") == f"https://www.youtube.com/watch?v={vid}"


def _fmt(fid, acodec, abr, asr, vcodec="none", ch=2, **kw):
    return dict(format_id=fid, acodec=acodec, vcodec=vcodec, abr=abr, asr=asr, audio_channels=ch, **kw)


def test_youtube_format_selection_rules():
    from music_separator import youtube

    base = [
        _fmt("139", "mp4a.40.5", 48.8, 22050), _fmt("140", "mp4a.40.2", 129.5, 44100),
        _fmt("140-drc", "mp4a.40.2", 129.5, 44100, format_note="medium, DRC"),
        _fmt("249", "opus", 52, 48000), _fmt("251", "opus", 118, 48000),
        _fmt("251-drc", "opus", 118, 48000, format_note="medium, DRC"),
        _fmt("18", "mp4a.40.2", 96, 44100, vcodec="avc1", height=360),
        dict(format_id="137", acodec="none", vcodec="avc1", height=1080),
    ]
    c = youtube.select_format(base)
    assert c.format["format_id"] == "251" and c.audio_only               # Opus 오디오 전용, DRC 제외
    # 비트레이트가 더 높아도 코덱 등급이 낮으면 선택하지 않음
    assert youtube.select_format([_fmt("141", "mp4a.40.2", 256, 44100),
                                  _fmt("251", "opus", 130, 48000)]).format["format_id"] == "251"
    # DRC 만 있는 Opus 보다 일반 AAC
    assert youtube.select_format([_fmt("251-drc", "opus", 130, 48000, format_note="DRC"),
                                  _fmt("140", "mp4a.40.2", 129, 44100)]).format["format_id"] == "140"
    # 원래 언어 트랙 우선
    langs = [_fmt("251-1", "opus", 130, 48000, language="ko", language_preference=-1),
             _fmt("251-0", "opus", 120, 48000, language="en", language_preference=10)]
    assert youtube.select_format(langs).format["format_id"] == "251-0"
    # 해설 음성·DRM 제외
    assert youtube.select_format([_fmt("251-desc", "opus", 130, 48000, language_preference=-10),
                                  _fmt("140", "mp4a.40.2", 129, 44100)]).format["format_id"] == "140"
    # 저비트레이트 보조 스트림(Opus 50kbps)만 있으면 일반 AAC 128kbps 를 사용
    assert youtube.select_format([_fmt("249", "opus", 50, 48000), _fmt("250", "opus", 70, 48000),
                                  _fmt("140", "mp4a.40.2", 129, 44100)]).format["format_id"] == "140"
    # 다채널보다 스테레오
    assert youtube.select_format([_fmt("256", "mp4a.40.2", 384, 48000, ch=6),
                                  _fmt("140", "mp4a.40.2", 129, 44100)]).format["format_id"] == "140"
    # 오디오 전용이 없으면 가장 낮은 해상도의 영상 포함 스트림
    c = youtube.select_format([_fmt("22", "mp4a.40.2", 192, 44100, vcodec="avc1", height=720),
                               _fmt("18", "mp4a.40.2", 96, 44100, vcodec="avc1", height=360)])
    assert not c.audio_only and c.format["format_id"] in ("18", "22")
    with pytest.raises(youtube.YouTubeError):
        youtube.select_format([dict(format_id="137", acodec="none", vcodec="avc1")])


def test_youtube_error_messages():
    from music_separator import youtube

    cases = {
        "ERROR: [youtube] x: Private video. Sign in if you've been granted access": "private",
        "ERROR: [youtube] x: Sign in to confirm your age.": "age",
        "ERROR: [youtube] x: Video unavailable": "unavailable",
        "ERROR: [youtube] x: This video is DRM protected": "drm",
        "ERROR: Unable to download webpage: <urlopen error [Errno -3] Temporary failure in name resolution>":
            "network",
        "ERROR: [youtube] x: Sign in to confirm you’re not a bot": "bot",
        "ERROR: unable to download video data: HTTP Error 403: Forbidden": "outdated",
        "ERROR: [youtube] x: Join this channel to get access to members-only content": "members",
    }
    for msg, kind in cases.items():
        e = youtube.friendly_error(Exception(msg))
        assert e.kind == kind, (msg, e.kind)
        assert any("가" <= ch <= "힣" for ch in str(e))           # 한국어 메시지


class _FakeYDL:
    """yt_dlp.YoutubeDL 대역: 정보 조회 + 선택된 형식 '다운로드'(실제 Opus 파일 생성)."""
    calls: list = []

    def __init__(self, params):
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False, process=True):
        _FakeYDL.calls.append(("extract", url))
        return {"id": "dQw4w9WgXcQ", "title": "테스트 영상", "duration": 5.0, "uploader": "me",
                "license": "Creative Commons Attribution license (reuse allowed)",
                "formats": [_fmt("140", "mp4a.40.2", 129.5, 44100, ext="m4a"),
                            _fmt("251", "opus", 128, 48000, ext="webm"),
                            _fmt("251-drc", "opus", 128, 48000, ext="webm", format_note="DRC")]}

    def process_ie_result(self, info, download=True):
        fid = self.params["format"]
        _FakeYDL.calls.append(("download", fid, self.params.get("postprocessors")))
        out = Path(self.params["paths"]["home"]) / f"{info['id']}.f{fid}.webm"
        from music_separator import audio_io

        x = song(5, 48000, 0.7)
        cmd = [audio_io.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", "48000",
               "-ac", "2", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "128k", str(out)]
        subprocess.run(cmd, input=x.astype("<f4").tobytes(), check=True)
        for hook in self.params.get("progress_hooks", []):
            hook({"status": "downloading", "downloaded_bytes": 10, "total_bytes": 20})
        return dict(info, requested_downloads=[{"filepath": str(out)}])


class _FakeYtModule:
    YoutubeDL = _FakeYDL

    class utils:
        class DownloadCancelled(Exception):
            pass

    class version:
        __version__ = "fake"


def _have_libopus():
    from music_separator import audio_io

    out = subprocess.run([audio_io.find_ffmpeg(), "-hide_banner", "-encoders"], stdout=subprocess.PIPE).stdout
    return b"libopus" in out


@pytest.mark.skipif(not _have_libopus(), reason="ffmpeg 에 libopus 인코더 없음")
def test_pipeline_youtube_keeps_stream_and_resamples_once(tmp_path, monkeypatch):
    from music_separator import pipeline, youtube

    monkeypatch.setattr(youtube, "load_ytdlp", lambda: _FakeYtModule)
    _FakeYDL.calls = []
    spec = pipeline.JobSpec("https://youtu.be/dQw4w9WgXcQ", tmp_path / "out", target="both", fmt="wav32f",
                            model_override="demucs_unittest")
    res = pipeline.Pipeline(spec, tmp_path / "work").run()
    rep = res.report
    y = rep["youtube"]
    assert y["format_id"] == "251" and y["codec"] == "opus" and y["audio_only"] and not y["reencoded"]
    assert ("download", "251", []) in _FakeYDL.calls                     # 후처리(MP3 변환 등) 없음
    assert rep["input"]["codec"] == "opus" and rep["input"]["sample_rate"] == 48000
    assert len(rep["decode"]["resample_steps"]) == 1                      # 48k → 44.1k (AI 입력) 한 번만
    outs = {Path(o["file"]).name: o for o in rep["outputs"]}
    voc = next(o for n, o in outs.items() if "보컬" in n)
    mr = next(o for n, o in outs.items() if "MR" in n)
    assert voc["sample_rate"] == 44100 and voc["resampled_from"] is None    # AI 출력 그대로
    assert mr["sample_rate"] == 48000 and mr["resampled_from"] is None      # 원본 레이트
    assert rep["lossy_encodes"] == 0
    assert res.prepared.source_path.suffix == ".webm"                      # 원래 컨테이너 그대로


# ------------------------------------------------------------ 파이프라인 ---

def _input(tmp_path, sr=48000, gain=0.8, name="곡.wav", sec=6.0):
    from music_separator import audio_io

    p = tmp_path / name
    audio_io.write_wav(p, song(sec, sr, gain), 24, sr)
    return p


def test_pipeline_outputs_only_selected_target(tmp_path):
    from music_separator import pipeline

    src = _input(tmp_path)
    for target, want in (("mr", ["MR"]), ("vocals", ["보컬"])):
        spec = pipeline.JobSpec(str(src), tmp_path / f"out_{target}", target=target,
                                model_override="demucs_unittest")
        res = pipeline.Pipeline(spec, tmp_path / "work").run()
        names = [p.name for p in res.files]
        assert len(names) == 1 and want[0] in names[0], names
        files = list((tmp_path / f"out_{target}").rglob("*.*"))
        assert len(files) == 1


def test_pipeline_original_target_skips_separation(tmp_path, monkeypatch):
    """'원본 그대로' : AI 분리 없이 원본(원래 샘플레이트)을 저장하고, 키는 원본에 한 번만 적용."""
    from music_separator import audio_io, engine, pipeline, pitch

    def no_sep(*a, **k):
        raise AssertionError("original target must not run separation")
    monkeypatch.setattr(pipeline.Pipeline, "separate", no_sep)
    monkeypatch.setattr(engine, "separate", no_sep, raising=False)
    src = _input(tmp_path, 48000)
    stages = []
    spec = pipeline.JobSpec(str(src), tmp_path / "out", target="original", fmt="wav32f")
    res = pipeline.Pipeline(spec, tmp_path / "work", progress=lambda s, f, o, m: stages.append((s, o))).run()
    assert res.separation is None and res.report["separation"] is None
    assert len(res.files) == 1 and "원본" in res.files[0].name
    y = audio_io.decode_audio(res.files[0], None)
    native = audio_io.decode_audio(src, None)
    assert audio_io.probe_audio(res.files[0]).sample_rate == 48000
    assert len(y) == len(native) and np.max(np.abs(y - native)) < 1e-6      # 비트 단위로 원본 그대로
    assert not any(s in ("model", "separate") for s, _ in stages)
    assert max(o for _, o in stages) <= 1.0

    calls = []
    real = pitch.shift
    monkeypatch.setattr(pitch, "shift", lambda d, s, r=44100, formant=False, cancel=None:
                        calls.append((s, r, formant)) or real(d, s, r, formant=formant, cancel=cancel))
    spec = pipeline.JobSpec(str(src), tmp_path / "out2", target="original", semitones=2)
    res = pipeline.Pipeline(spec, tmp_path / "work2").run()
    assert calls == [(2, 48000, False)] and "원본 [Key +2]" in res.files[0].name


def test_pipeline_mr_is_native_original_minus_vocals(tmp_path, monkeypatch):
    from music_separator import audio_io, pipeline, pitch

    def boom(*a, **k):
        raise AssertionError("key 0 must skip pitch shifting")
    monkeypatch.setattr(pitch, "_shift_rubberband", boom)
    src = _input(tmp_path, 48000)
    spec = pipeline.JobSpec(str(src), tmp_path / "out", target="both", fmt="wav32f",
                            model_override="demucs_unittest")
    res = pipeline.Pipeline(spec, tmp_path / "work").run()
    mr_path = next(p for p in res.files if "MR" in p.name)
    voc_path = next(p for p in res.files if "보컬" in p.name)
    mr = audio_io.decode_audio(mr_path, None)
    voc = audio_io.decode_audio(voc_path, None)
    native = audio_io.decode_audio(src, None)
    assert audio_io.probe_audio(mr_path).sample_rate == 48000
    assert audio_io.probe_audio(voc_path).sample_rate == 44100
    expect = native - audio_io.resample(voc, 44100, 48000)
    assert len(mr) == len(native)
    assert np.max(np.abs(mr - expect)) < 1e-5
    assert res.report["gain_db"] == 0.0


def test_pipeline_clipping_guard_pcm_and_mp3(tmp_path):
    from music_separator import audio_io, pipeline

    src = _input(tmp_path, 44100, gain=1.6)          # 원본 자체가 0dBFS 를 넘는 float 입력
    for fmt in ("wav24", "mp3", "wav32f"):
        spec = pipeline.JobSpec(str(src), tmp_path / f"out_{fmt}", target="mr", fmt=fmt,
                                model_override="demucs_unittest")
        res = pipeline.Pipeline(spec, tmp_path / "work").run()
        d = res.export.details[0]
        y = audio_io.decode_audio(res.files[0], None)
        if fmt == "wav32f":
            assert res.export.gain_db == 0.0 and np.max(np.abs(y)) > 1.0      # float 은 그대로 보존
        elif fmt == "wav24":
            assert res.export.gain_db < 0 and np.max(np.abs(y)) <= 10 ** (-0.1 / 20) + 1e-4
        else:
            assert res.export.gain_db < 0 and np.max(np.abs(y)) <= 1.0
            assert d.true_peak_db is not None and d.true_peak_db <= -0.99


def test_pipeline_key_applied_once_to_selected_output(tmp_path, monkeypatch):
    from music_separator import pipeline, pitch

    calls = []
    real = pitch.shift

    def spy(data, semis, rate=44100, formant=False, cancel=None):
        calls.append((semis, rate, formant, len(data)))
        return real(data, semis, rate, formant=formant, cancel=cancel)
    monkeypatch.setattr(pitch, "shift", spy)
    src = _input(tmp_path, 44100)
    spec = pipeline.JobSpec(str(src), tmp_path / "out", target="vocals", semitones=-3,
                            model_override="demucs_unittest", formant_vocals=True)
    res = pipeline.Pipeline(spec, tmp_path / "work").run()
    assert len(calls) == 1 and calls[0][0] == -3 and calls[0][2] is True
    assert "Key -3" in res.files[0].name


def test_pipeline_errors_and_cancel(tmp_path):
    from music_separator import pipeline

    for source, kind in ((str(tmp_path / "없는파일.mp3"), "not_found"),
                         ("https://vimeo.com/123", "invalid_url"), ("", "empty")):
        with pytest.raises(pipeline.PipelineError) as ei:
            pipeline.Pipeline(pipeline.JobSpec(source, tmp_path / "o"), tmp_path / "w").run()
        assert ei.value.kind == kind
        assert any("가" <= ch <= "힣" for ch in str(ei.value))
    bad = tmp_path / "notaudio.mp3"
    bad.write_text("this is not audio")
    with pytest.raises(pipeline.PipelineError) as ei:
        pipeline.Pipeline(pipeline.JobSpec(str(bad), tmp_path / "o"), tmp_path / "w").run()
    assert ei.value.kind in ("unsupported", "ffmpeg")

    src = _input(tmp_path, 44100)
    ev = threading.Event()

    def prog(stage, f, o, m):
        if stage == "separate":
            ev.set()
    spec = pipeline.JobSpec(str(src), tmp_path / "out_c", model_override="demucs_unittest")
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.Pipeline(spec, tmp_path / "w2", progress=prog, cancel=ev).run()
    assert not list((tmp_path / "out_c").rglob("*.*"))


def test_pipeline_reuses_decode_and_separation(tmp_path):
    from music_separator import pipeline

    src = _input(tmp_path, 44100)
    spec = pipeline.JobSpec(str(src), tmp_path / "out", target="mr", model_override="demucs_unittest")
    p1 = pipeline.Pipeline(spec, tmp_path / "work")
    r1 = p1.run()
    key = p1.separation_key(r1.prepared)
    spec2 = pipeline.JobSpec(str(src), tmp_path / "out", target="vocals", model_override="demucs_unittest")
    r2 = pipeline.Pipeline(spec2, tmp_path / "work").run(r1.prepared, r1.separation, key)
    assert r2.reused_separation and r2.prepared is r1.prepared


def test_presets_and_model_plan():
    from music_separator import engine

    assert engine.required_models("auto", "2", "quality") == ["bs_roformer_sw"]
    assert engine.required_models("auto", "2", "best") == ["bs_roformer_sw"]
    assert engine.required_models("auto", "2", "speed") == ["htdemucs_ft_vocals"]
    assert engine.required_models("demucs", "2", "quality") == ["htdemucs_ft_vocals"]
    assert engine.required_models("roformer", "2", "speed") == ["bs_roformer_sw"]
    assert engine.PRESETS["best"].roformer_overlap > engine.PRESETS["quality"].roformer_overlap
    assert engine.resolve_engine("auto", "speed") == "demucs"


def test_ft_vocals_model_found_in_htdemucs_ft_folder(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tests"))
    import fake_models
    from music_separator import models

    root = tmp_path / "m"
    fake_models.make(root, "htdemucs_ft", "th")
    monkeypatch.setenv("MUSIC_SEPARATOR_MODELS_DIR", str(root))
    assert models.find_model_dir("htdemucs_ft_vocals") == root / "htdemucs_ft"
    m = models.load_model("htdemucs_ft_vocals")
    assert len(m.models) == 1 and "vocals" in m.sources


def test_default_preset_by_build(monkeypatch, tmp_path):
    """CPU 버전·Intel Mac = 속도 우선, CUDA 버전·Apple Silicon = 품질 우선 (빌드 때 기록한 build_info.json 기준)."""
    import json

    from music_separator import engine, paths

    monkeypatch.setattr(paths, "resource_dir", lambda: tmp_path)
    for variant, want in [("cpu", "speed"), ("cuda", "quality"), ("arm64", "quality"), ("x86_64", "speed")]:
        (tmp_path / "build_info.json").write_text(json.dumps({"variant": variant}), encoding="utf-8")
        assert engine.build_variant() == variant
        assert engine.app_default_preset() == want


def test_safe_filename_windows_rules():
    from music_separator.audio_io import safe_filename
    # 잘린 지점이 공백이면 Windows가 폴더명 끝 공백을 지워 경로가 깨진다(실사용 버그)
    title = "2023 DAY6 Christmas Special Concert 'The Present : You are My Day' LIVE | Love me"
    for n in range(20, 90):
        s = safe_filename(title, n)
        assert s == s.rstrip(" ."), (n, s)
        assert len(s) <= n
    long = "a" * 50 + " " + "b" * 200 + ".wav"
    out = safe_filename(long)
    assert out.endswith(".wav") and len(out) <= 120 and not out[:-4].endswith(" ")
    assert safe_filename("CON") == "_CON"
    assert safe_filename("nul.wav") == "_nul.wav"
    assert safe_filename("  ...  ") == "track"
    assert safe_filename("a/b:c") == "a_b_c"


def test_updater_release_parsing():
    from music_separator import updater

    def rel(tag, names, **kw):
        return {"tag_name": tag, "html_url": "https://example/r", "draft": False, "prerelease": False,
                "assets": [{"name": n, "browser_download_url": "https://dl/" + n, "size": 10} for n in names], **kw}

    names = ["MRMaker-99.0.0-windows-x64-cpu-setup.exe", "MRMaker-99.0.0-windows-x64-cpu-portable.zip",
             "MRMaker-99.0.0-windows-x64-cuda-setup-1.bin", "MRMaker-99.0.0-windows-x64-cuda-setup.exe",
             "MRMaker-99.0.0-macos-arm64.dmg"]
    cpu = updater.parse_release(rel("v99.0.0", names), "cpu", auto=True)
    assert cpu.version == "99.0.0" and [a.name for a in cpu.assets] == [names[0]]
    cuda = updater.parse_release(rel("v99.0.0", names), "cuda", auto=True)
    assert [a.name for a in cuda.assets] == [names[3], names[2]]  # exe 먼저, 분할 bin 같은 폴더
    mac = updater.parse_release(rel("v99.0.0", names), "arm64", auto=False)
    assert mac and not mac.can_auto_install and mac.page_url == "https://example/r"
    assert updater.parse_release(rel("v0.0.1", names), "cpu", auto=True) is None      # 구버전
    assert updater.parse_release(rel("v99.0.0", names, draft=True), "cpu", auto=True) is None
    assert updater.is_newer("1.10.0", "1.9.9") and not updater.is_newer("1.2.2", "1.2.2")


def test_settings_are_per_build_variant(tmp_path, monkeypatch):
    """CUDA 버전에서 고른 처리 방식이 CPU 버전 기본값을 덮어쓰지 않아야 한다."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("MUSIC_SEPARATOR_NONINTERACTIVE", "1")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from music_separator import engine
    from music_separator.ui import main_window, theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    ini = str(tmp_path / "s.ini")
    legacy = QSettings(ini, QSettings.Format.IniFormat)
    legacy.setValue("preset2", "quality")        # 예전 버전이 남긴 공용 값
    legacy.setValue("engine2", "bs_roformer_sw")
    legacy.sync()

    def open_win(variant):
        monkeypatch.setattr(engine, "build_variant", lambda: variant)
        w = main_window.MainWindow(QSettings(ini, QSettings.Format.IniFormat))
        return w

    w = open_win("cpu")
    assert w.preset_key() == "speed" and w.engine_key() == "auto"   # 공용 예전 값 무시
    w.close(); w.device_thread.wait(15000); w.deleteLater()

    w = open_win("cuda")
    assert w.preset_key() == "quality"
    w.preset_btns["best"].setChecked(True)
    chosen = w.preset_key()
    w.close(); w.device_thread.wait(15000); w.deleteLater()

    w = open_win("cpu")
    assert w.preset_key() == "speed"                                  # CUDA 쪽 선택이 넘어오지 않음
    w.close(); w.device_thread.wait(15000); w.deleteLater()

    w = open_win("cuda")
    assert w.preset_key() == chosen                                   # 자기 선택은 유지
    w.close(); w.device_thread.wait(15000); w.deleteLater()
    s = QSettings(ini, QSettings.Format.IniFormat)
    assert s.value("cpu/preset") is None                              # 기본값은 저장하지 않음
