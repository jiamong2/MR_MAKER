"""[실험] 화음 따기: 메인 보컬 / 화음(백보컬) 분리 + 음표(MIDI) 추출.

1) 기존 파이프라인으로 보컬/반주 분리 (같은 입력이면 이전 결과 재사용)
2) 보컬 트랙을 카라오케 모델(BS-RoFormer, frazer & becruily)에 넣어 메인 보컬 추정 → 화음 = 보컬 − 메인
   (같은 마스크의 나머지라 메인 + 화음 = 보컬, 둘 사이에 빠지거나 겹치는 소리 없음)
3) MR · 메인 보컬 · 화음 저장 (기존 저장 방식 그대로, 세 파일에 같은 음량 배율)
4) 메인 · 화음 트랙을 각각 음정 추적(RMVPE, 한 번에 한 음) → 음표로 정리 → MIDI 파일 2개
   화음 트랙에 조금 남은 메인 보컬처럼 보컬 전체보다 많이 작은 소리는 음표로 만들지 않음
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

MODEL_NAME = "karaoke_lead"
PITCH_MODEL = "rmvpe"
LABELS = {"lead": "메인 보컬", "back": "화음"}
TRACK_NAMES = {"lead": "Lead vocal", "back": "Harmony"}     # MIDI 트랙 이름 (DAW 호환을 위해 영문)
STEM_KEYS = {"lead": "vocals_lead", "back": "vocals_back"}
REL_GATE_DB = 18.0       # 보컬 전체보다 이만큼 이상 작은 소리는 음표로 만들지 않음 (분리 후 남은 다른 목소리)
ABS_GATE_DB = -55.0      # 이보다 작은 소리는 무음으로 봄

# 전체 진행률 배분
P_SEP, P_DL, P_SPLIT, P_SAVE, P_PITCH = 0.40, 0.45, 0.80, 0.86, 0.98

ProgressCb = Callable[[float, str], None]


@dataclass
class HarmonyResult:
    files: list                   # 저장한 오디오 + MIDI
    prepared: pipeline.Prepared
    separation: engine.SeparationResult
    separation_key: tuple
    seconds: float = 0.0
    notes: list = field(default_factory=list)       # 안내 문구
    counts: dict = field(default_factory=dict)      # {"lead": 음표 수, "back": 음표 수}
    midi: dict = field(default_factory=dict)        # {"lead": 경로, "back": 경로}


def missing_mb(model_name: str = MODEL_NAME, pitch_model: str = PITCH_MODEL) -> int:
    return sum(models.missing_download_mb(n) for n in (model_name, pitch_model))


def _ensure_models(names, report: ProgressCb, cancel: threading.Event, lo: float, hi: float):
    need = [n for n in names if n in models.MODELS and not models.is_installed(n)]
    if not need:
        return
    mb = sum(models.MODELS[n].approx_mb for n in need)
    if system.free_disk(models.user_models_dir()) < mb * system.MB * 1.2:
        raise pipeline.PipelineError(f"AI 모델을 저장할 공간이 부족합니다 (약 {mb}MB 필요).", "model", "disk", False)
    total = mb * 1048576
    base = 0
    for n in need:
        size = models.MODELS[n].approx_mb * 1048576

        def dl(done, tot, label, base=base):
            d = base + min(done, size)
            report(lo + (hi - lo) * d / max(1, total),
                   f"AI 모델 다운로드 중 (처음 한 번) {d / 1048576:.0f} / {total / 1048576:.0f} MB")
        try:
            models.download_model(n, dl, cancel)
        except models.DownloadCancelled as exc:
            models.remove_partial_downloads()
            raise pipeline.PipelineCancelled() from exc
        except Exception as exc:  # noqa: BLE001
            models.remove_partial_downloads()
            raise pipeline.PipelineError(engine.friendly_error(exc, "AI 모델 다운로드"), "model", "model_download",
                                         True) from exc
        base += size


def run(spec: pipeline.JobSpec, work_dir: Path, progress: Optional[ProgressCb] = None,
        cancel: Optional[threading.Event] = None, previous_prep=None, previous_sep=None,
        previous_sep_key=None, model_name: str = MODEL_NAME, pitch_model: str = PITCH_MODEL) -> HarmonyResult:
    """spec 의 입력·처리 방식·장치·저장 형식·키로 실행. spec.target 은 무시하고 MR + 메인 + 화음 + MIDI 2개를 저장."""
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

    _ensure_models([model_name, pitch_model], report, cancel, P_SEP, P_DL)
    if cancel.is_set():
        raise pipeline.PipelineCancelled()
    run_dir = Path(work_dir) / f"harmony_{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        lead_path, back_path, vocals_path = _split(spec, sep, run_dir, model_name, report, cancel)
        exp = _save(spec, prep, sep, run_dir, lead_path, back_path, report, cancel)
        files = list(exp.files)
        notes = []
        if exp.gain_db < 0:
            notes.append(f"0dBFS 를 넘는 부분이 있어 세 파일의 음량을 똑같이 {exp.gain_db:.1f}dB 낮췄습니다.")
        midi, counts, tuning = _transcribe(spec, prep, run_dir, lead_path, back_path, vocals_path, pitch_model,
                                           report, cancel)
        files += [midi["lead"], midi["back"]]
        if abs(tuning) >= 0.1:
            notes.append(f"원곡 음높이가 기준(A4 = 440Hz)보다 {tuning * 100:+.0f}센트 달라, 그만큼 보정해 음표를 정했습니다.")
    finally:
        for n in (model_name, pitch_model):
            models.forget(n)                      # 큰 모델은 메모리에서 내림 (다음에 다시 불러옴)
        gc.collect()
        shutil.rmtree(run_dir, ignore_errors=True)
    report(1.0, "완료")
    return HarmonyResult(files, prep, sep, key, time.monotonic() - t0, notes, counts, midi)


def _split(spec, sep, run_dir: Path, model_name: str, report: ProgressCb, cancel: threading.Event):
    """보컬 트랙 → 메인 보컬 · 화음 (.npy, 44.1kHz)."""
    from . import roformer

    report(P_DL, "메인/화음 분리 모델 불러오는 중…")
    model = models.load_model(model_name)
    vocals_path = next(s.path for s in sep.stems if s.key == "vocals")
    vocals = np.load(vocals_path, mmap_mode="r")
    device = spec.device if spec.device in ("cuda", "mps") else "cpu"
    preset = engine.PRESETS.get(spec.preset, engine.PRESETS[engine.DEFAULT_PRESET])
    overlap = max(1, min(int(getattr(model, "num_overlap", 2)), preset.roformer_overlap))
    try:
        out = roformer.demix(model, vocals, device, num_overlap=overlap,
                             chunk=getattr(model, "chunk", roformer.SW_CHUNK),
                             progress=lambda f: report(P_DL + (P_SPLIT - P_DL) * f,
                                                       f"메인 보컬 / 화음 나누는 중 {f * 100:.0f}%"),
                             cancel=cancel)
    except roformer.Cancelled as exc:
        raise pipeline.PipelineCancelled() from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("lead/back separation failed")
        raise pipeline.PipelineError(engine.friendly_error(exc), "separate", "separate", True) from exc
    lead = np.ascontiguousarray(out[0], dtype=np.float32)
    del out
    back = np.asarray(vocals, dtype=np.float32) - lead
    lead_path, back_path = run_dir / "lead.npy", run_dir / "back.npy"
    np.save(lead_path, lead)
    np.save(back_path, back)
    del lead, back, vocals
    return lead_path, back_path, Path(vocals_path)


def _save(spec, prep, sep, run_dir: Path, lead_path: Path, back_path: Path, report: ProgressCb,
          cancel: threading.Event):
    """MR · 메인 보컬 · 화음을 기존 저장 방식(샘플레이트·키 조절·피크 처리)으로 저장."""
    report(P_SPLIT, "저장 준비 중…")
    residual = sep.stems[-1]
    src = ExportSource(stems={STEM_KEYS["lead"]: lead_path, STEM_KEYS["back"]: back_path, residual.key: residual.path},
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
            ExportJob(name(f"{LABELS['lead']} (실험)"), spec.fmt, LABELS["lead"], "stem", key=STEM_KEYS["lead"]),
            ExportJob(name(f"{LABELS['back']} (실험)"), spec.fmt, LABELS["back"], "stem", key=STEM_KEYS["back"])]
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


def _transcribe(spec, prep, run_dir: Path, lead_path: Path, back_path: Path, vocals_path: Path, pitch_model: str,
                report: ProgressCb, cancel: threading.Event):
    """메인 · 화음 → MIDI 2개. 반환: ({kind: 경로}, {kind: 음표 수}, 튜닝(반음))."""
    from . import rmvpe

    report(P_SAVE, "음정 추적 모델 불러오는 중…")
    model = models.load_model(pitch_model)
    device = "cuda" if spec.device == "cuda" else "cpu"
    rate = engine.MODEL_RATE
    f0s, dbs = {}, {}
    for i, kind in enumerate(("lead", "back")):
        arr = np.load(lead_path if kind == "lead" else back_path, mmap_mode="r")
        try:
            f0, _conf = rmvpe.track(model, arr, rate, device, cancel=cancel,
                                    progress=lambda f, i=i, kind=kind: report(
                                        P_SAVE + (P_PITCH - P_SAVE) * (i + f) / 2,
                                        f"{LABELS[kind]} 음정 따는 중 {f * 100:.0f}%"))
        except rmvpe.Cancelled as exc:
            raise pipeline.PipelineCancelled() from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("pitch tracking failed")
            raise pipeline.PipelineError(engine.friendly_error(exc, "음정 추적"), "separate", "pitch", True) from exc
        f0s[kind] = f0
        dbs[kind] = transcribe.frame_db(arr, rate, len(f0))
        del arr
    n = max(len(f) for f in f0s.values())
    total = transcribe.frame_db(np.load(vocals_path, mmap_mode="r"), rate, n)
    gate = np.maximum(total - REL_GATE_DB, ABS_GATE_DB)
    heard = {k: transcribe.hz_to_midi(np.where(dbs[k] >= gate[:len(f)], f, 0.0)) for k, f in f0s.items()}
    tuning = transcribe.estimate_tuning(*heard.values())
    report(P_PITCH, "MIDI 파일 만드는 중…")
    song = audio_io.safe_filename(prep.title, 60)
    song_dir = Path(spec.out_dir) / song
    kl = f" [Key {spec.semitones:+d}]" if spec.semitones else ""
    midi, counts = {}, {}
    for kind, f0 in f0s.items():
        notes = transcribe.notes_from_f0(f0, dbs[kind], gate[:len(f0)], tuning)
        if spec.semitones:
            notes = transcribe.transpose(notes, spec.semitones)
        path = audio_io.unique_path(song_dir / audio_io.safe_filename(f"{song} - {LABELS[kind]} (실험){kl}.mid"))
        midi[kind] = transcribe.write_midi(path, notes, TRACK_NAMES[kind])
        counts[kind] = len(notes)
    return midi, counts, tuning
