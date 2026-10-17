"""[실험] 화음 따기: 보컬에서 주 선율과 화음을 찾아 음표(MIDI)로 받아 적기.

1) 기존 파이프라인으로 보컬/반주 분리 (같은 입력이면 이전 결과 재사용)
2) 보컬 트랙에서 가장 잘 들리는 목소리의 음정 추적 (RMVPE) → 주 선율 후보
3) 그 목소리의 배음 자리를 지운 나머지 소리에서 두 번째 목소리의 음정 추적 → 화음 후보
4) 화음은 진짜로 같이 부르는 소리만 남김
   - 나머지 소리 안에서 그 음의 배음 에너지가 보컬 전체에 비해 충분하고 (숨소리·자음·반주 잔향 제외;
     12dB 기준을 넘는 부분이 있어야 하고, 그 앞뒤는 15dB 기준까지 이어 붙임)
   - 0.15초 이상 이어지며
   - 주 선율과 1반음 안쪽이면 같은 목소리로 봄
   - 주 선율이 함께 날 때만 (주 선율이 없는 곳에서 잡힌 소리는 주 선율의 음량 기준을 넘지 못한 작은 소리 — 버림)
5) 같이 부르는 동안 가장 잘 들리는 목소리가 바뀌어 주 선율이 다른 사람에게 튀면, 이어지던 쪽을 주 선율로 유지
6) 저장: MR, 주 선율 / 화음 MIDI, 확인용 오디오 2개 (화음 = 같이 부르는 구간의 화음 배음 자리만,
   주 선율 = 보컬 − 화음 → 둘을 더하면 원래 보컬)

검증: scripts/probe_harmony.py (정답 트랙이 있는 남/여 예제로 만든 '솔로 → 솔로 → 함께' 듀엣, 잔향 포함).
"""
from __future__ import annotations

import gc
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio_io, engine, models, pipeline, pitch, system, transcribe
from .export import ExportCancelled, Exporter, ExportJob, ExportOptions, ExportSource

log = logging.getLogger(__name__)

MODEL_NAME = "rmvpe"
LABELS = {"mel": "주 선율", "har": "화음"}
TRACK_NAMES = {"mel": "Melody", "har": "Harmony"}       # MIDI 트랙 이름 (DAW 호환을 위해 영문)
STEM_KEYS = {"mel": "vocals_melody", "har": "vocals_harmony"}
SR16, HOP16, NFFT16 = 16000, 160, 1024
CANCEL_REL = 0.03        # 두 번째 목소리를 찾을 때 지우는 배음 폭 (주파수 비율)
MASK_REL, MASK_MAX = 0.03, 0.4     # 배음 에너지 측정·확인용 오디오 마스크 폭 (비율, f0 대비 상한)
SAME_VOICE = 1.0         # 반음: 두 번째 음정이 첫 번째와 이만큼 가까우면 같은 목소리로 봄
HARM_REL_DB = 12.0       # 화음: 그 음의 배음 에너지가 보컬 전체보다 이만큼 이상 작으면 버림
HARM_WEAK_DB = 15.0      # 화음 구간을 앞뒤로 늘릴 때의 느슨한 기준 (구간 안에 위 기준을 넘는 프레임이 충분해야 함)
HARM_MIN = 0.15          # 초: 화음이 이만큼 이어져야 인정 (위 기준을 넘는 프레임 기준)
# 세 값은 정답 트랙이 있는 시험곡(듀엣 / 번갈아 / 번갈아 + 잔향 / 작은 백보컬)에서 가짜 화음 비율과 화음 검출률을 함께 보고
# 고름 (scripts/probe_harmony.py 의 grid). '주 선율이 방금 떠난 음의 잔향' 규칙은 진짜 화음을 더 많이 지워서 뺐음
HARM_GAP = 0.03          # 초: 화음 안의 이 정도 끊김은 이어진 것으로 봄
ABS_GATE_DB = -55.0      # 이보다 작은 소리는 무음으로 봄
QUIET_DB = 40.0          # 주 선율: 곡의 큰 보컬(상위 5%)보다 이만큼 이상 작으면 무시 (반주 잔향 등)

# 전체 진행률 배분
P_SEP, P_DL, P_B, P_SPLIT, P_SAVE = 0.55, 0.60, 0.87, 0.91, 0.97

ProgressCb = Callable[[float, str], None]
PitchFn = Callable[[np.ndarray], tuple]     # 16kHz 모노 → (f0 Hz, 신뢰도), 10ms 간격


class _Cancelled(Exception):
    pass


@dataclass
class HarmonyResult:
    files: list                   # 저장한 오디오 + MIDI
    prepared: pipeline.Prepared
    separation: engine.SeparationResult
    separation_key: tuple
    seconds: float = 0.0
    notes: list = field(default_factory=list)       # 안내 문구
    counts: dict = field(default_factory=dict)      # {"mel": 음표 수, "har": 음표 수}
    midi: dict = field(default_factory=dict)        # {"mel": 경로, "har": 경로}
    together: float = 0.0                           # 노래하는 구간 중 화음이 같이 나는 비율


def missing_mb(model_name: str = MODEL_NAME) -> int:
    return models.missing_download_mb(model_name)


# ------------------------------------------------------- STFT (numpy) ---
# torch.stft / istft(center=True, 반사 패딩, 주기형 Hann 창, 창 제곱합 정규화)와 같은 계산.
# 프레임 묶음 단위로 처리해 긴 곡도 메모리를 적게 씀.

def _window(n_fft: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)


def n_frames(n: int, hop: int) -> int:
    return 1 + n // hop


def _frames(x: np.ndarray, n_fft: int, hop: int):
    """(패딩한 신호, 프레임 수). 프레임 t 는 원래 신호의 t·hop 을 중심으로 함."""
    pad = n_fft // 2
    xp = np.pad(np.asarray(x, dtype=np.float32), (pad, pad), mode="reflect" if len(x) > pad else "constant")
    return xp, n_frames(len(x), hop)


def stft_power(x: np.ndarray, n_fft: int, hop: int, block: int = 2048):
    """프레임 묶음마다 (시작 프레임, |X|² (주파수, 프레임)) 를 냄."""
    xp, T = _frames(x, n_fft, hop)
    win = _window(n_fft)
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        idx = (np.arange(t0, t1) * hop)[:, None] + np.arange(n_fft)[None, :]
        X = np.fft.rfft(xp[idx] * win, axis=1)
        yield t0, (X.real ** 2 + X.imag ** 2).T.astype(np.float32)


def masked_resynth(x: np.ndarray, n_fft: int, hop: int, mask_fn: Callable[[int, int], np.ndarray],
                   block: int = 2048) -> np.ndarray:
    """x 의 STFT 에 mask_fn(t0, t1) → (주파수, t1 − t0) 마스크를 곱해 다시 합성 (길이는 x 와 같음)."""
    xp, T = _frames(x, n_fft, hop)
    win = _window(n_fft)
    out = np.zeros(len(xp), np.float64)
    norm = np.zeros(len(xp), np.float64)
    w2 = (win.astype(np.float64)) ** 2
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        starts = np.arange(t0, t1) * hop
        idx = starts[:, None] + np.arange(n_fft)[None, :]
        X = np.fft.rfft(xp[idx] * win, axis=1) * mask_fn(t0, t1).T
        y = np.fft.irfft(X, n=n_fft, axis=1) * win
        for i, s in enumerate(starts):
            out[s:s + n_fft] += y[i]
            norm[s:s + n_fft] += w2
    pad = n_fft // 2
    out = out[pad:pad + len(x)]
    norm = norm[pad:pad + len(x)]
    return (out / np.where(norm > 1e-11, norm, 1.0)).astype(np.float32)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))


def midi_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.where(np.isfinite(m), 440.0 * 2.0 ** ((np.nan_to_num(m) - 69.0) / 12.0), 0.0)


# ------------------------------------------------------------ 신호 처리 ---

def cancel_harmonics(a16: np.ndarray, f0: np.ndarray, rel: float = CANCEL_REL) -> np.ndarray:
    """16kHz 모노에서 f0(10ms 간격)의 배음 자리를 지운 나머지 소리."""
    freqs = np.fft.rfftfreq(NFFT16, 1 / SR16)
    f = _fit(f0, n_frames(len(a16), HOP16))
    return masked_resynth(a16, NFFT16, HOP16, lambda t0, t1: 1.0 - transcribe.harmonic_mask(f[t0:t1], freqs, rel))


def harmonic_energy_db(x16: np.ndarray, f0: np.ndarray, rel: float = MASK_REL,
                       max_frac: float = MASK_MAX) -> np.ndarray:
    """프레임마다 f0 의 배음 자리에 있는 에너지 (dB, 10ms 간격). f0 ≤ 0 이면 -120."""
    freqs = np.fft.rfftfreq(NFFT16, 1 / SR16)
    T = n_frames(len(x16), HOP16)
    f = _fit(f0, T)
    out = np.full(T, -120.0)
    for t0, P in stft_power(x16, NFFT16, HOP16):
        t1 = t0 + P.shape[1]
        e = (P * transcribe.harmonic_mask(f[t0:t1], freqs, rel, max_frac)).sum(axis=0)
        out[t0:t1] = 10 * np.log10(np.maximum(e, 1e-12))
    out[f <= 0] = -120.0
    return out


def total_energy_db(x16: np.ndarray) -> np.ndarray:
    T = n_frames(len(x16), HOP16)
    out = np.full(T, -120.0)
    for t0, P in stft_power(x16, NFFT16, HOP16):
        out[t0:t0 + P.shape[1]] = 10 * np.log10(np.maximum(P.sum(axis=0), 1e-12))
    return out


def split_audio(vocals: np.ndarray, mel_hz: np.ndarray, har_hz: np.ndarray, rate: int) -> tuple:
    """보컬 (n, ch) → (주 선율, 화음) 확인용 오디오. 주 선율 + 화음 = 보컬 (샘플 단위로 정확히).

    화음이 나는 프레임에서만 화음의 배음 자리를 화음 쪽으로 (주 선율과 겹치는 칸은 비율대로).
    나머지 소리(주 선율, 자음·숨소리, 화음이 없는 구간)는 모두 주 선율 쪽."""
    n_fft, hop = 2048, rate // transcribe.FPS
    x = np.asarray(vocals, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    T = n_frames(len(x), hop)
    mel_hz, har_hz = _fit(mel_hz, T), _fit(har_hz, T)
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)

    def mask(t0, t1):
        mh = transcribe.harmonic_mask(har_hz[t0:t1], freqs, MASK_REL, MASK_MAX)
        mm = transcribe.harmonic_mask(np.where(har_hz[t0:t1] > 0, mel_hz[t0:t1], 0.0), freqs, MASK_REL, MASK_MAX)
        s = mh + mm
        return np.where(s > 1, mh / np.maximum(s, 1e-9), mh).astype(np.float32)
    har = np.empty_like(x)
    for ch in range(x.shape[1]):
        har[:, ch] = masked_resynth(x[:, ch], n_fft, hop, mask)
    return x - har, har


# ------------------------------------------------------- 두 목소리 찾기 ---

def _rmvpe_fn(model, device: str, progress: Callable[[float], None], cancel) -> PitchFn:
    from . import rmvpe

    def fn(a16):
        return rmvpe.decode(rmvpe.salience_16k(model, a16, device, progress, cancel))
    return fn


def validate_harmony(ma: np.ndarray, mb: np.ndarray, harm_db: np.ndarray, total_db: np.ndarray,
                     fps: int = transcribe.FPS) -> np.ndarray:
    """두 번째 목소리 후보(mb) 중 진짜 화음만 남김 (나머지는 NaN). 조건은 모듈 설명 4) 참고."""
    n = min(len(ma), len(mb), len(harm_db), len(total_db))
    ma, mb = np.asarray(ma[:n], float), np.asarray(mb[:n], float).copy()
    rel = harm_db[:n] - total_db[:n]
    strong = rel >= -HARM_REL_DB
    ok = np.isfinite(mb) & np.isfinite(ma) & (rel >= -max(HARM_WEAK_DB, HARM_REL_DB))
    ok &= np.abs(np.nan_to_num(ma) - np.nan_to_num(mb)) >= SAME_VOICE
    mb[~ok] = np.nan
    # 이어진 구간 (높이가 1.5반음 넘게 튀면 끊음, 짧은 끊김은 이음)
    gap = int(round(HARM_GAP * fps))
    runs: list = []
    for a, b in transcribe._runs(np.isfinite(mb)):
        if runs and a - runs[-1][-1][1] <= gap and abs(mb[a] - mb[runs[-1][-1][1] - 1]) < 1.5:
            runs[-1].append((a, b))
        else:
            runs.append([(a, b)])
    out = np.full(n, np.nan)
    min_len = int(round(HARM_MIN * fps))
    for parts in runs:
        s, e = parts[0][0], parts[-1][1]
        # 구간 안에서도 높이가 크게 튀면 나눔
        pieces, start = [], s
        for t in range(s + 1, e):
            if np.isfinite(mb[t]) and np.isfinite(mb[t - 1]) and abs(mb[t] - mb[t - 1]) >= 1.5:
                pieces.append((start, t))
                start = t
        pieces.append((start, e))
        for a, b in pieces:
            if np.sum(np.isfinite(mb[a:b]) & strong[a:b]) < min_len:
                continue
            out[a:b] = mb[a:b]
    return out


def keep_melody_line(ma: np.ndarray, mh: np.ndarray) -> tuple:
    """같이 부르는 동안 가장 잘 들리는 목소리가 바뀌어 주 선율 후보가 튀고, 화음이 직전 주 선율을 이어 가면
    둘을 바꿔 주 선율이 한 사람의 선을 따라가게 함. 반환: (주 선율, 화음)."""
    mel = np.asarray(ma, float).copy()
    har = np.asarray(mh, float).copy()
    last, last_t = np.nan, -100
    for t in range(len(mel)):
        a, b = mel[t], har[t]
        if np.isfinite(a) and np.isfinite(b) and t - last_t <= 10 and np.isfinite(last):
            if abs(a - last) >= 1.5 and abs(b - last) <= 0.75:
                mel[t], har[t] = b, a
        if np.isfinite(mel[t]):
            last, last_t = mel[t], t
    return mel, har


def voice_candidates(a16: np.ndarray, fn1: PitchFn, fn2: PitchFn, check: Callable[[], None] = lambda: None,
                     report: Callable[[float, str], None] = lambda f, m: None) -> dict:
    """16kHz 모노 보컬 → 주 선율 후보(ma), 두 번째 목소리 후보(mb), 그 배음 에너지(harm), 전체 에너지(total).
    음정은 MIDI 실수(NaN = 없음), 에너지는 dB, 모두 10ms 간격."""
    fa, _ = fn1(a16)
    check()
    report(0.45, "화음 찾는 중")
    res = cancel_harmonics(a16, fa)
    check()
    fb, _ = fn2(res)
    check()
    report(0.9, "화음 확인 중")
    n = min(len(fa), len(fb))
    fa, fb = _fit(fa, n), _fit(fb, n)
    total = _fit(total_energy_db(a16), n)
    loud = float(np.percentile(total[fa > 0], 95)) if (fa > 0).any() else 0.0
    ma = transcribe.hz_to_midi(np.where(total >= max(ABS_GATE_DB, loud - QUIET_DB), fa, 0.0))
    mb = transcribe.hz_to_midi(np.where(total >= ABS_GATE_DB, fb, 0.0))
    harm = _fit(harmonic_energy_db(res, np.where(np.isfinite(mb), fb, 0.0)), n)
    return {"ma": ma, "mb": mb, "harm": harm, "total": total, "residual": res}


def two_voices(model, vocals: np.ndarray, rate: int, device: str = "cpu",
               progress: Optional[Callable[[float, str], None]] = None,
               cancel: Optional[threading.Event] = None, pitch_fn: Optional[PitchFn] = None) -> tuple:
    """보컬 (n, ch) → (주 선율, 화음) 음정 (MIDI 실수, NaN = 없음, 10ms 간격).
    pitch_fn: 음정 추적 함수 (기본 RMVPE, 검증용으로 바꿔 끼울 수 있음)."""
    report = progress or (lambda f, m: None)

    def check():
        if cancel is not None and cancel.is_set():
            raise _Cancelled()
    x = np.asarray(vocals, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    a16 = audio_io.resample(mono[:, None], rate, SR16)[:, 0]
    fn1 = pitch_fn or _rmvpe_fn(model, device, lambda f: report(f * 0.45, "주 선율 음정 따는 중"), cancel)
    fn2 = pitch_fn or _rmvpe_fn(model, device, lambda f: report(0.5 + f * 0.4, "화음 음정 따는 중"), cancel)
    c = voice_candidates(a16, fn1, fn2, check, report)
    report(1.0, "화음 확인 완료")
    return pick_lines(c)


def pick_lines(c: dict) -> tuple:
    """후보 → (주 선율, 화음)."""
    mh = validate_harmony(c["ma"], c["mb"], c["harm"], c["total"])
    return keep_melody_line(c["ma"], mh)


# ------------------------------------------------------------------ 실행 ---

def run(spec: pipeline.JobSpec, work_dir: Path, progress: Optional[ProgressCb] = None,
        cancel: Optional[threading.Event] = None, previous_prep=None, previous_sep=None,
        previous_sep_key=None, model_name: str = MODEL_NAME) -> HarmonyResult:
    """spec 의 입력·처리 방식·장치·저장 형식·키로 실행. spec.target 은 무시하고
    MR + 주 선율 / 화음 확인용 오디오 + 주 선율 / 화음 MIDI 를 저장."""
    from . import rmvpe

    t0 = time.monotonic()
    cancel = cancel or threading.Event()
    report = progress or (lambda f, m: None)
    try:
        models.remove_obsolete()
    except Exception:  # noqa: BLE001
        pass

    def stage1(stage, frac, overall, msg):
        report(P_SEP * overall, msg)

    p = pipeline.Pipeline(replace(spec, target="mr"), Path(work_dir), progress=stage1, cancel=cancel)
    prep = p.prepare(previous_prep)
    sep, _reused = p.separate(prep, previous_sep, previous_sep_key)
    key = p.separation_key(prep)
    _ensure_model(model_name, report, cancel)
    run_dir = Path(work_dir) / f"harmony_{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        report(P_DL, "음정 추적 모델 불러오는 중…")
        model = models.load_model(model_name)
        vocals_path = next(s.path for s in sep.stems if s.key == "vocals")
        vocals = np.load(vocals_path, mmap_mode="r")
        rate = sep.model_rate
        device = "cuda" if spec.device == "cuda" else "cpu"
        try:
            mel, har = two_voices(model, vocals, rate, device, lambda f, m: report(
                P_DL + (P_B - P_DL) * f, f"{m} {f * 100:.0f}%"), cancel)
            if cancel.is_set():
                raise _Cancelled()
            report(P_B, "주 선율 / 화음 소리 나누는 중…")
            a_mel, a_har = split_audio(vocals, midi_to_hz(mel), midi_to_hz(har), rate)
        except (rmvpe.Cancelled, _Cancelled) as exc:
            raise pipeline.PipelineCancelled() from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("harmony tracking failed")
            raise pipeline.PipelineError(engine.friendly_error(exc, "음정 추적"), "separate", "pitch", True) from exc
        mel_path, har_path = run_dir / "melody.npy", run_dir / "harmony.npy"
        np.save(mel_path, a_mel)
        np.save(har_path, a_har)
        n = len(mel)
        db = {"mel": transcribe.frame_db(a_mel, rate, n), "har": transcribe.frame_db(a_har, rate, n)}
        del a_mel, a_har, vocals
        exp = _save(spec, prep, sep, run_dir, mel_path, har_path, report, cancel)
        files = list(exp.files)
        notes = []
        if exp.gain_db < 0:
            notes.append(f"0dBFS 를 넘는 부분이 있어 세 파일의 음량을 똑같이 {exp.gain_db:.1f}dB 낮췄습니다.")
        report(P_SAVE, "MIDI 파일 만드는 중…")
        tuning = transcribe.estimate_tuning(mel, har)
        song = audio_io.safe_filename(prep.title, 60)
        song_dir = Path(spec.out_dir) / song
        kl = f" [Key {spec.semitones:+d}]" if spec.semitones else ""
        midi, counts = {}, {}
        for kind, line in (("mel", mel), ("har", har)):
            nts = transcribe.notes_from_pitch(line, db[kind], -200.0, tuning)
            if spec.semitones:
                nts = transcribe.transpose(nts, spec.semitones)
            path = audio_io.unique_path(song_dir / audio_io.safe_filename(f"{song} - {LABELS[kind]} (실험){kl}.mid"))
            midi[kind] = transcribe.write_midi(path, nts, TRACK_NAMES[kind])
            counts[kind] = len(nts)
            files.append(midi[kind])
        if abs(tuning) >= 0.1:
            notes.append(f"원곡 음높이가 기준(A4 = 440Hz)보다 {tuning * 100:+.0f}센트 달라, 그만큼 보정해 음표를 정했습니다.")
        sung = np.isfinite(mel) | np.isfinite(har)
        together = float(np.isfinite(har)[sung].mean()) if sung.any() else 0.0
        if together < 0.02:
            notes.append("두 목소리가 함께 나는 부분을 거의 찾지 못해 화음 파일은 비어 있을 수 있습니다.")
    finally:
        models.forget(model_name)                 # 큰 모델은 메모리에서 내림 (다음에 다시 불러옴)
        gc.collect()
        shutil.rmtree(run_dir, ignore_errors=True)
    report(1.0, "완료")
    return HarmonyResult(files, prep, sep, key, time.monotonic() - t0, notes, counts, midi, together)


def _ensure_model(name: str, report: ProgressCb, cancel: threading.Event):
    if name not in models.MODELS or models.is_installed(name):
        return
    mb = models.MODELS[name].approx_mb
    if system.free_disk(models.user_models_dir()) < mb * system.MB * 1.2:
        raise pipeline.PipelineError(f"AI 모델을 저장할 공간이 부족합니다 (약 {mb}MB 필요).", "model", "disk", False)

    def dl(done, total, label):
        report(P_SEP + (P_DL - P_SEP) * done / max(1, total),
               f"음정 추적 모델 다운로드 중 (처음 한 번) {done / 1048576:.0f} / {total / 1048576:.0f} MB")
    try:
        models.download_model(name, dl, cancel)
    except models.DownloadCancelled as exc:
        models.remove_partial_downloads()
        raise pipeline.PipelineCancelled() from exc
    except Exception as exc:  # noqa: BLE001
        models.remove_partial_downloads()
        raise pipeline.PipelineError(engine.friendly_error(exc, "AI 모델 다운로드"), "model", "model_download",
                                     True) from exc


def _save(spec, prep, sep, run_dir: Path, mel_path: Path, har_path: Path, report: ProgressCb,
          cancel: threading.Event):
    """MR · 주 선율 / 화음 확인용 오디오를 기존 저장 방식(샘플레이트·키 조절·피크 처리)으로 저장."""
    report(P_SPLIT, "저장 준비 중…")
    residual = sep.stems[-1]
    src = ExportSource(stems={STEM_KEYS["mel"]: mel_path, STEM_KEYS["har"]: har_path, residual.key: residual.path},
                       residual_key=residual.key, mix44_path=prep.mix44_path, native_path=prep.native_path,
                       native_rate=prep.native_rate, model_rate=sep.model_rate)
    song = audio_io.safe_filename(prep.title, 60)
    song_dir = Path(spec.out_dir) / song
    try:
        song_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise pipeline.PipelineError(f"저장 폴더를 만들 수 없습니다:\n{song_dir}\n({exc})\n"
                                     "고급 옵션의 [설정]에서 다른 저장 위치를 선택해 주세요.", "encode", "output",
                                     False) from exc
    ext = audio_io.EXPORT_FORMATS[spec.fmt][1]
    kl = f" [Key {spec.semitones:+d}]" if spec.semitones else ""
    name = lambda label: audio_io.unique_path(song_dir / audio_io.safe_filename(f"{song} - {label}{kl}{ext}"))  # noqa
    jobs = [ExportJob(name("MR (보컬 제거)"), spec.fmt, "MR", "mr", excluded=list(STEM_KEYS.values()), gains={}),
            ExportJob(name(f"{LABELS['mel']} (확인용)"), spec.fmt, LABELS["mel"], "stem", key=STEM_KEYS["mel"]),
            ExportJob(name(f"{LABELS['har']} (확인용)"), spec.fmt, LABELS["har"], "stem", key=STEM_KEYS["har"])]
    rate = prep.native_rate if spec.keep_native_rate else engine.MODEL_RATE
    n = int(prep.n_samples * rate / engine.MODEL_RATE)
    need = sum(audio_io.estimate_export_bytes(n, j.fmt, rate) for j in jobs) + n * 8 * (len(jobs) + 4)
    if system.free_disk(song_dir) < need:
        raise pipeline.PipelineError(f"저장 공간이 부족합니다.\n필요: {system.human_bytes(need)}, "
                                     f"남은 공간: {system.human_bytes(system.free_disk(song_dir))}", "encode", "disk",
                                     False)
    opts = ExportOptions(keep_native_rate=spec.keep_native_rate, semitones=spec.semitones,
                         formant_vocals=spec.formant_vocals)
    try:
        return Exporter(src, opts, run_dir, progress=lambda f, m: report(P_SPLIT + (P_SAVE - P_SPLIT) * f, m),
                        cancel=cancel).run(jobs)
    except (ExportCancelled, pitch.PitchCancelled) as exc:
        for j in jobs:
            j.path.unlink(missing_ok=True)
        raise pipeline.PipelineCancelled() from exc
    except pitch.PitchError as exc:
        raise pipeline.PipelineError(f"키 변경에 실패했습니다.\n{exc}", "key", "key", True) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("harmony export failed")
        raise pipeline.PipelineError(engine.friendly_error(exc, "저장"), "encode", "output", True) from exc
