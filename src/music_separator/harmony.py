"""[실험] 화음 따기: 보컬에서 두 성부(위 / 아래)를 찾아 음표(MIDI)로 받아 적기.

1) 기존 파이프라인으로 보컬/반주 분리 (같은 입력이면 이전 결과 재사용)
2) 보컬 트랙에서 가장 큰 목소리의 음정 추적 (RMVPE)
3) 그 목소리의 배음 자리를 지운 나머지 소리에서 두 번째 목소리의 음정 추적
   (첫 번째와 1반음 안쪽이면 같은 목소리가 남은 것으로 보고 버림)
4) 프레임마다 높은 음 = 위 성부, 낮은 음 = 아래 성부 (한 목소리만 나는 곳은 위 성부)
5) 저장: MR, 위/아래 성부 확인용 오디오(보컬을 각 성부의 배음 자리로 나눈 것 — 둘을 더하면 원래 보컬),
   위/아래 성부 MIDI

남/여 합창 예제(정답 트랙 있음)로 잰 결과: 두 성부를 동시에 맞힌 프레임 87%, 찾은 음의 95% 가 정답
(scripts/probe_harmony.py). 같은 음을 옥타브로 겹쳐 부르면 위쪽 목소리를 찾지 못할 수 있다.
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
LABELS = {"up": "위 성부", "lo": "아래 성부"}
TRACK_NAMES = {"up": "Upper voice", "lo": "Lower voice"}     # MIDI 트랙 이름 (DAW 호환을 위해 영문)
STEM_KEYS = {"up": "vocals_upper", "lo": "vocals_lower"}
CANCEL_REL = 0.03        # 두 번째 목소리를 찾을 때 지우는 배음 폭 (주파수 비율)
PREVIEW_REL, PREVIEW_MAX = 0.03, 0.4     # 확인용 오디오 마스크 폭 (비율, f0 대비 상한)
SAME_VOICE = 1.0         # 반음: 두 번째 음정이 첫 번째와 이만큼 가까우면 같은 목소리로 봄
REL_GATE_DB = 18.0       # 두 번째 목소리: 보컬 전체보다 이만큼 이상 작으면 무시
ABS_GATE_DB = -55.0      # 이보다 작은 소리는 무음으로 봄
QUIET_DB = 40.0          # 첫 번째 목소리: 곡의 큰 보컬(상위 5%)보다 이만큼 이상 작으면 무시 (반주 잔향 등)

# 전체 진행률 배분
P_SEP, P_DL, P_A, P_CANCEL, P_B, P_SPLIT, P_SAVE = 0.55, 0.60, 0.72, 0.75, 0.87, 0.91, 0.97

ProgressCb = Callable[[float, str], None]


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
    counts: dict = field(default_factory=dict)      # {"up": 음표 수, "lo": 음표 수}
    midi: dict = field(default_factory=dict)        # {"up": 경로, "lo": 경로}
    together: float = 0.0                           # 노래하는 구간 중 두 목소리가 함께 나는 비율


def missing_mb(model_name: str = MODEL_NAME) -> int:
    return models.missing_download_mb(model_name)


# ------------------------------------------------------------ 신호 처리 ---

def cancel_harmonics(a16: np.ndarray, f0: np.ndarray, rel: float = CANCEL_REL) -> np.ndarray:
    """16kHz 모노에서 f0(10ms 간격)의 배음 자리를 지운 나머지 소리."""
    import torch

    from . import rmvpe

    x = torch.from_numpy(np.ascontiguousarray(a16, dtype=np.float32))
    win = torch.hann_window(rmvpe.N_FFT)
    X = torch.stft(x, rmvpe.N_FFT, rmvpe.HOP, rmvpe.N_FFT, win, center=True, return_complex=True)
    f = _fit(f0, X.shape[1])
    mask = 1.0 - transcribe.harmonic_mask(f, np.fft.rfftfreq(rmvpe.N_FFT, 1 / rmvpe.SAMPLE_RATE), rel)
    y = torch.istft(X * torch.from_numpy(mask), rmvpe.N_FFT, rmvpe.HOP, rmvpe.N_FFT, win, center=True,
                    length=len(a16))
    return y.numpy()


def split_audio(vocals: np.ndarray, up_hz: np.ndarray, lo_hz: np.ndarray, rate: int) -> tuple:
    """보컬 (n, ch) → (위 성부, 아래 성부) 확인용 오디오. 위 + 아래 = 보컬 (샘플 단위로 정확히).

    두 성부가 함께 나는 곳은 각 성부의 배음 자리를 나눠 갖고(겹치는 칸은 비율대로), 배음이 아닌 소리(자음·숨소리)는
    반씩. 한 성부만 나는 곳은 그 성부가 전부 가짐."""
    import torch

    n_fft, hop = 2048, rate // transcribe.FPS
    x = np.asarray(vocals, dtype=np.float32)
    T = 1 + len(x) // hop
    up_hz, lo_hz = _fit(up_hz, T), _fit(lo_hz, T)
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    mu = transcribe.harmonic_mask(up_hz, freqs, PREVIEW_REL, PREVIEW_MAX)
    ml = transcribe.harmonic_mask(lo_hz, freqs, PREVIEW_REL, PREVIEW_MAX)
    s = mu + ml
    scale = np.where(s > 1, 1.0 / np.maximum(s, 1e-9), 1.0).astype(np.float32)
    mu *= scale
    ml *= scale
    vu, vl = up_hz > 0, lo_hz > 0
    w = np.where(vu & ~vl, 1.0, np.where(vl & ~vu, 0.0, 0.5)).astype(np.float32)
    m_up = mu + (1.0 - mu - ml) * w[None, :]
    del mu, ml, s, scale
    M = torch.from_numpy(m_up)
    win = torch.hann_window(n_fft)
    up = np.empty_like(x)
    for ch in range(x.shape[1]):
        xc = torch.from_numpy(np.ascontiguousarray(x[:, ch]))
        X = torch.stft(xc, n_fft, hop, n_fft, win, center=True, return_complex=True)
        up[:, ch] = torch.istft(X * M[:, :X.shape[1]], n_fft, hop, n_fft, win, center=True, length=len(xc)).numpy()
    return up, x - up


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))


def midi_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.where(np.isfinite(m), 440.0 * 2.0 ** ((np.nan_to_num(m) - 69.0) / 12.0), 0.0)


def two_voices(model, vocals: np.ndarray, rate: int, device: str = "cpu",
               progress: Optional[Callable[[float, str], None]] = None,
               cancel: Optional[threading.Event] = None) -> tuple:
    """보컬 (n, ch) → (첫 번째 목소리, 두 번째 목소리) 음정 (MIDI 실수, NaN = 없음, 10ms 간격)."""
    from . import rmvpe

    report = progress or (lambda f, m: None)
    x = np.asarray(vocals, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    a16 = audio_io.resample(mono[:, None], rate, rmvpe.SAMPLE_RATE)[:, 0]
    sal = rmvpe.salience_16k(model, a16, device, lambda f: report(f * 0.45, "첫 번째 목소리 음정 따는 중"), cancel)
    fa, _ = rmvpe.decode(sal)
    del sal
    if cancel is not None and cancel.is_set():
        raise rmvpe.Cancelled()
    report(0.45, "두 번째 목소리 찾는 중")
    res = cancel_harmonics(a16, fa)
    sal = rmvpe.salience_16k(model, res, device, lambda f: report(0.5 + f * 0.5, "두 번째 목소리 음정 따는 중"),
                             cancel)
    fb, _ = rmvpe.decode(sal)
    n = min(len(fa), len(fb))
    fa, fb = fa[:n], fb[:n]
    db_total = transcribe.frame_db(a16, rmvpe.SAMPLE_RATE, n)
    db_res = transcribe.frame_db(res, rmvpe.SAMPLE_RATE, n)
    loud = float(np.percentile(db_total[fa > 0], 95)) if (fa > 0).any() else 0.0
    ma = transcribe.hz_to_midi(np.where(db_total >= max(ABS_GATE_DB, loud - QUIET_DB), fa, 0.0))
    mb = transcribe.hz_to_midi(np.where(db_res >= np.maximum(db_total - REL_GATE_DB, ABS_GATE_DB), fb, 0.0))
    mb[np.isfinite(ma) & np.isfinite(mb) & (np.abs(ma - mb) < SAME_VOICE)] = np.nan
    return ma, mb


# ------------------------------------------------------------------ 실행 ---

def run(spec: pipeline.JobSpec, work_dir: Path, progress: Optional[ProgressCb] = None,
        cancel: Optional[threading.Event] = None, previous_prep=None, previous_sep=None,
        previous_sep_key=None, model_name: str = MODEL_NAME) -> HarmonyResult:
    """spec 의 입력·처리 방식·장치·저장 형식·키로 실행. spec.target 은 무시하고
    MR + 위/아래 성부 확인용 오디오 + 위/아래 성부 MIDI 를 저장."""
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
            ma, mb = two_voices(model, vocals, rate, device, lambda f, m: report(
                P_DL + (P_B - P_DL) * f, f"{m} {f * 100:.0f}%"), cancel)
            up, lo = transcribe.assign_voices(ma, mb)
            if cancel.is_set():
                raise _Cancelled()
            report(P_B, "위 / 아래 성부 소리 나누는 중…")
            a_up, a_lo = split_audio(vocals, midi_to_hz(up), midi_to_hz(lo), rate)
        except (rmvpe.Cancelled, _Cancelled) as exc:
            raise pipeline.PipelineCancelled() from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("two-voice tracking failed")
            raise pipeline.PipelineError(engine.friendly_error(exc, "음정 추적"), "separate", "pitch", True) from exc
        up_path, lo_path = run_dir / "upper.npy", run_dir / "lower.npy"
        np.save(up_path, a_up)
        np.save(lo_path, a_lo)
        n = len(up)
        db = {"up": transcribe.frame_db(a_up, rate, n), "lo": transcribe.frame_db(a_lo, rate, n)}
        del a_up, a_lo, vocals
        exp = _save(spec, prep, sep, run_dir, up_path, lo_path, report, cancel)
        files = list(exp.files)
        notes = []
        if exp.gain_db < 0:
            notes.append(f"0dBFS 를 넘는 부분이 있어 세 파일의 음량을 똑같이 {exp.gain_db:.1f}dB 낮췄습니다.")
        report(P_SAVE, "MIDI 파일 만드는 중…")
        tuning = transcribe.estimate_tuning(up, lo)
        song = audio_io.safe_filename(prep.title, 60)
        song_dir = Path(spec.out_dir) / song
        kl = f" [Key {spec.semitones:+d}]" if spec.semitones else ""
        midi, counts = {}, {}
        for kind, line in (("up", up), ("lo", lo)):
            nts = transcribe.notes_from_pitch(line, db[kind], -200.0, tuning)
            if spec.semitones:
                nts = transcribe.transpose(nts, spec.semitones)
            path = audio_io.unique_path(song_dir / audio_io.safe_filename(f"{song} - {LABELS[kind]} (실험){kl}.mid"))
            midi[kind] = transcribe.write_midi(path, nts, TRACK_NAMES[kind])
            counts[kind] = len(nts)
            files.append(midi[kind])
        if abs(tuning) >= 0.1:
            notes.append(f"원곡 음높이가 기준(A4 = 440Hz)보다 {tuning * 100:+.0f}센트 달라, 그만큼 보정해 음표를 정했습니다.")
        sung = np.isfinite(up) | np.isfinite(lo)
        together = float((np.isfinite(up) & np.isfinite(lo))[sung].mean()) if sung.any() else 0.0
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


def _save(spec, prep, sep, run_dir: Path, up_path: Path, lo_path: Path, report: ProgressCb,
          cancel: threading.Event):
    """MR · 위/아래 성부 확인용 오디오를 기존 저장 방식(샘플레이트·키 조절·피크 처리)으로 저장."""
    report(P_SPLIT, "저장 준비 중…")
    residual = sep.stems[-1]
    src = ExportSource(stems={STEM_KEYS["up"]: up_path, STEM_KEYS["lo"]: lo_path, residual.key: residual.path},
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
            ExportJob(name(f"{LABELS['up']} (확인용)"), spec.fmt, LABELS["up"], "stem", key=STEM_KEYS["up"]),
            ExportJob(name(f"{LABELS['lo']} (확인용)"), spec.fmt, LABELS["lo"], "stem", key=STEM_KEYS["lo"])]
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
