"""[실험] 화음 따기: 음정 곡선 → 음표 → MIDI, 메인/화음 분리 파이프라인 (네트워크·GPU 없이 실행)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from music_separator import transcribe  # noqa: E402
from music_separator.transcribe import Note  # noqa: E402

FPS = transcribe.FPS


def contour(spec, tuning=0.0, seed=0, vibrato=0.0, scoop=True):
    """[(MIDI 번호 또는 None, 초)] → 10ms 간격 f0(Hz). 음 시작에 아래에서 올라오는 슬라이드, 작은 흔들림 포함."""
    rng = np.random.default_rng(seed)
    out = []
    for p, sec in spec:
        n = int(round(sec * FPS))
        if p is None:
            out.append(np.zeros(n))
            continue
        m = np.full(n, float(p))
        if scoop:
            k = min(5, n // 3)
            m[:k] -= np.linspace(1.5, 0.0, k)
        if vibrato:
            m += vibrato * np.sin(2 * np.pi * 5.5 * np.arange(n) / FPS)
        m += rng.normal(0, 0.05, n) + tuning
        out.append(440.0 * 2 ** ((m - 69) / 12))
    return np.concatenate(out)


def pitches(notes):
    return [n.pitch for n in notes]


def test_notes_follow_melody_with_scoops_and_repeats():
    spec = [(60, 0.5), (62, 0.3), (64, 0.12), (65, 0.4), (None, 0.1), (65, 0.3), (67, 0.6), (None, 0.2)]
    notes = transcribe.notes_from_f0(contour(spec))
    assert pitches(notes) == [60, 62, 64, 65, 65, 67]
    starts = [0.0, 0.5, 0.8, 0.92, 1.42, 1.72]
    for nt, s in zip(notes, starts):
        assert abs(nt.start - s) <= 0.05, (nt, s)
    assert abs(notes[3].end - 1.32) <= 0.03           # 쉼표 앞에서 끝남 → 같은 음 두 번이 따로 남음


def test_vibrato_does_not_split_notes():
    f0 = contour([(57, 1.2), (60, 1.0)], vibrato=0.6)
    assert pitches(transcribe.notes_from_f0(f0)) == [57, 60]


def test_fast_runs_kept():
    """100ms 짜리 빠른 꾸밈음도 각각 음으로 남음."""
    spec = [(64, 0.3)] + [(p, 0.1) for p in (66, 67, 69, 67, 66)] + [(64, 0.4)]
    assert pitches(transcribe.notes_from_f0(contour(spec, scoop=False))) == [64, 66, 67, 69, 67, 66, 64]


def test_tuning_estimate_and_correction():
    spec = [(60, 0.6), (64, 0.6), (67, 0.6), (62, 0.6)]
    f0 = contour(spec, tuning=0.4)                     # 원곡이 40센트 높게 녹음된 경우
    m = transcribe.hz_to_midi(f0)
    t = transcribe.estimate_tuning(m)
    assert abs(t - 0.4) < 0.06
    assert pitches(transcribe.notes_from_f0(f0, tuning=t)) == [60, 64, 67, 62]
    assert transcribe.estimate_tuning(np.full(10, np.nan)) == 0.0


def test_gate_drops_leftover_voice_and_short_blips():
    f0 = contour([(60, 0.5), (67, 0.5), (60, 0.03), (None, 0.2), (62, 0.5)])
    db = np.full(len(f0), -20.0)
    db[50:100] = -50.0                                 # 다른 목소리가 조금 남은 부분 (기준 아래)
    notes = transcribe.notes_from_f0(f0, db, gate_db=-40.0)
    assert pitches(notes) == [60, 62]
    gate = np.full(len(f0), -40.0)
    gate[110:] = -10.0                                 # 프레임별 기준
    assert pitches(transcribe.notes_from_f0(f0, db, gate_db=gate)) == [60]
    assert transcribe.notes_from_f0(np.zeros(100)) == []
    assert transcribe.notes_from_f0(np.array([])) == []


def test_velocity_follows_loudness():
    f0 = contour([(60, 0.5), (None, 0.1), (62, 0.5)])
    db = np.concatenate([np.full(50, -12.0), np.full(10, -90.0), np.full(50, -30.0)])
    a, b = transcribe.notes_from_f0(f0, db)
    assert a.velocity > b.velocity >= 25


def test_frame_db_matches_rms():
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = np.zeros((len(t), 2), np.float32)
    x[sr // 2: 3 * sr // 2] = (0.5 * np.sin(2 * np.pi * 220 * t[sr // 2: 3 * sr // 2]))[:, None]
    db = transcribe.frame_db(x, sr, 201)
    assert abs(db[100] - 20 * np.log10(0.5 / np.sqrt(2))) < 0.2
    assert db[10] < -100


def test_midi_roundtrip_and_timing(tmp_path):
    notes = [Note(0.0, 0.5, 60, 90), Note(0.5, 0.75, 60, 70), Note(1.234, 2.0, 67, 110)]
    p = transcribe.write_midi(tmp_path / "x.mid", notes, "Lead vocal")
    data = p.read_bytes()
    assert data[:4] == b"MThd" and b"Lead vocal" in data
    back, bpm, ppq = transcribe.read_midi_notes(data)
    assert bpm == pytest.approx(120.0) and ppq == 480
    assert [(n.pitch, n.velocity) for n in back] == [(60, 90), (60, 70), (67, 110)]
    for a, b in zip(notes, back):
        assert abs(a.start - b.start) < 0.002 and abs(a.end - b.end) < 0.002
    assert not list(tmp_path.glob("*.part"))
    empty, _, _ = transcribe.read_midi_notes(transcribe.midi_bytes([], "Harmony"))
    assert empty == []
    assert [n.pitch for n in transcribe.transpose(notes, -2)] == [58, 58, 65]
    long = transcribe.midi_bytes([Note(600.0, 600.5, 72)])     # 10분 지점 (가변 길이 수치 여러 바이트)
    assert transcribe.read_midi_notes(long)[0][0].start == pytest.approx(600.0, abs=0.002)


# ------------------------------------------------- 전체 파이프라인 (torch 필요) ---

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_SEPARATOR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MUSIC_SEPARATOR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MUSIC_SEPARATOR_MODELS_DIR", raising=False)
    from music_separator import models

    models.clear_cache()
    yield tmp_path


def _song(tmp_path, sec=4.0, sr=44100):
    from music_separator import audio_io
    from music_separator.selftest import make_test_signal

    p = tmp_path / "듀엣.wav"
    audio_io.write_wav(p, (make_test_signal(sec, sr) * 0.8).astype(np.float32), 24, sr)
    return p


def test_harmony_outputs_audio_and_midi(isolated):
    """MR + 메인 + 화음 (오디오 3개, 메인 + 화음 = 분리된 보컬) + MIDI 2개, 진행률은 0→1 로만 증가."""
    pytest.importorskip("torch")
    from music_separator import audio_io, harmony, pipeline

    tmp = isolated
    spec = pipeline.JobSpec(str(_song(tmp)), tmp / "out", fmt="wav32f", model_override="demucs_unittest")
    steps = []
    res = harmony.run(spec, tmp / "work", progress=lambda f, m: steps.append(f), model_name="karaoke_unittest",
                      pitch_model="rmvpe_unittest")
    names = [p.name for p in res.files]
    assert len(names) == 5, names
    assert sum(n.endswith(".wav") for n in names) == 3 and sum(n.endswith(".mid") for n in names) == 2
    lead = next(p for p in res.files if "메인 보컬" in p.name and p.suffix == ".wav")
    back = next(p for p in res.files if "화음" in p.name and p.suffix == ".wav")
    mr = next(p for p in res.files if "MR" in p.name)
    vocals = np.load(next(s.path for s in res.separation.stems if s.key == "vocals"))
    a, b = audio_io.decode_audio(lead, None), audio_io.decode_audio(back, None)
    assert a.shape == b.shape == vocals.shape
    assert np.max(np.abs(a + b - vocals)) < 1e-5
    orig = audio_io.decode_audio(Path(spec.source), None)
    assert np.max(np.abs(audio_io.decode_audio(mr, None) + vocals - orig)) < 1e-4
    for kind in ("lead", "back"):
        notes, bpm, _ = transcribe.read_midi_notes(res.midi[kind].read_bytes())
        assert bpm == pytest.approx(120.0) and len(notes) == res.counts[kind]
        assert all(0 <= n.start < n.end <= 4.1 for n in notes)
    assert max(steps) <= 1.0 and steps == sorted(steps)
    assert not list((tmp / "work").glob("harmony_*"))          # 임시 파일 정리


def test_harmony_reuses_previous_separation_and_cancels(isolated):
    pytest.importorskip("torch")
    import threading

    from music_separator import harmony, pipeline

    tmp = isolated
    spec = pipeline.JobSpec(str(_song(tmp, 3.0)), tmp / "out", fmt="wav", model_override="demucs_unittest")
    first = pipeline.Pipeline(spec, tmp / "work").run()
    msgs = []
    res = harmony.run(spec, tmp / "work", lambda f, m: msgs.append(m), None, first.prepared, first.separation,
                      pipeline.Pipeline(spec, tmp / "work").separation_key(first.prepared),
                      model_name="karaoke_unittest", pitch_model="rmvpe_unittest")
    assert any("이전 분리 결과" in m for m in msgs)
    assert res.separation is first.separation
    ev = threading.Event()

    def cancel_at_split(f, m):
        if "메인 보컬 / 화음" in m:
            ev.set()
    with pytest.raises(pipeline.PipelineCancelled):
        harmony.run(spec, tmp / "work2", cancel_at_split, ev, model_name="karaoke_unittest",
                    pitch_model="rmvpe_unittest")


def test_obsolete_duet_model_removed(isolated):
    from music_separator import models

    old = models.user_models_dir() / "duet_male_female"
    old.mkdir(parents=True)
    (old / "model.ckpt").write_bytes(b"x" * 10)
    keep = models.user_models_dir() / "htdemucs_ft_vocals"
    keep.mkdir(parents=True)
    assert models.remove_obsolete() == ["duet_male_female"]
    assert not old.exists() and keep.exists()


def test_rmvpe_mel_and_decode():
    """멜 필터(librosa htk=True 와 같은 정의)와 확률 → f0 변환."""
    pytest.importorskip("torch")
    from music_separator import rmvpe

    w = rmvpe.mel_filters()
    assert w.shape == (128, 513) and np.all(w >= 0)
    assert np.all(w.sum(axis=1) > 0)                           # 비어 있는 필터 없음
    sal = np.full((3, 360), 0.001, np.float32)
    k = 150
    sal[0, k - 1:k + 2] = (0.3, 0.9, 0.3)
    sal[2, 10] = 0.02                                          # 기준(0.03) 이하 → 무성
    f0, conf = rmvpe.decode(sal)
    assert f0[0] == pytest.approx(10 * 2 ** (rmvpe.CENTS[k] / 1200), rel=1e-3)
    assert f0[1] == 0 and f0[2] == 0 and conf[0] == pytest.approx(0.9)


def test_rmvpe_chunked_shapes():
    """긴 입력을 나눠 계산해도 프레임 수가 그대로 (10ms 간격, 프레임 i 중심 = i/100초)."""
    pytest.importorskip("torch")
    from music_separator import models, rmvpe

    models.clear_cache()
    model = models.load_model("rmvpe_unittest")
    for sec in (0.01, 3.0, 40.0):
        n = int(sec * 16000)
        sal = rmvpe.salience_16k(model, np.zeros(n, np.float32))
        assert sal.shape == (1 + n // 160, 360)
    f0, conf = rmvpe.track(model, np.zeros((44100, 2), np.float32), 44100)
    assert f0.shape == conf.shape == (101,)
