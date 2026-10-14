"""원클릭 처리 파이프라인 (홈 화면의 [음원 추출 시작]).

1. 입력 유형 식별 (로컬 파일 / YouTube URL)
2. 파일 검증 또는 허용된 YouTube 오디오 획득 (원본 스트림 그대로, 재인코딩 없음)
3. 오디오 메타데이터 확인 (코덱·샘플레이트·채널·길이, 디스크 여유)
4. 고품질 PCM 디코딩 (원본 샘플레이트 float32 1회 + 필요할 때만 44.1kHz 로 soxr VHQ 변환 1회)
5. AI 분리 모델 준비 (처음 한 번 다운로드)
6. 2-Stem 음원 분리 (보컬 추정, 반주 = 원본 − 보컬)
7. 선택한 결과물(보컬 또는 MR)만 계산
8. 키 변경이 0 이 아닐 때만 피치 시프트 (선택한 결과물에 한 번)
9. 클리핑 검사 (필요할 때만 고정 배율로 음량 조정) · 출력 유효성 검사
10. 선택한 형식으로 단 한 번 최종 인코딩
11. 저장 폴더에 파일 제공

중간 데이터는 모두 float32 무손실(.npy)이며, 원본 파일은 읽기만 합니다.
GUI 와 독립적으로 동작하므로 테스트·자가 진단에서 그대로 실행할 수 있습니다.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio_io, engine, models, pitch, system, youtube
from .dsp import compute_peaks
from .export import ExportCancelled, Exporter, ExportJob, ExportOptions, ExportReport, ExportSource

log = logging.getLogger(__name__)

TARGETS = {"mr": "반주(MR)", "vocals": "보컬", "both": "보컬 + 반주"}

# (키, 표시 이름, 전체 진행률에서 차지하는 비중)
STAGES = [
    ("check", "입력 확인", 0.01),
    ("acquire", "오디오 가져오기", 0.10),
    ("decode", "디코딩", 0.04),
    ("model", "AI 모델 준비", 0.04),
    ("separate", "AI 음원 분리", 0.66),
    ("key", "키 조절", 0.08),
    ("encode", "저장", 0.07),
]
STAGE_LABEL = {k: v for k, v, _ in STAGES}


class PipelineCancelled(Exception):
    pass


class PipelineError(Exception):
    def __init__(self, message: str, stage: str = "", kind: str = "unknown", retryable: bool = True,
                 can_use_fast_model: bool = False, can_update_youtube: bool = False):
        super().__init__(message)
        self.stage = stage
        self.kind = kind
        self.retryable = retryable
        self.can_use_fast_model = can_use_fast_model
        self.can_update_youtube = can_update_youtube


@dataclass
class JobSpec:
    source: str                       # 로컬 파일 경로 또는 YouTube URL
    out_dir: Path
    target: str = "mr"                # mr / vocals / both
    semitones: int = 0
    fmt: str = audio_io.DEFAULT_FORMAT
    preset: str = engine.DEFAULT_PRESET
    engine: str = "auto"
    device: str = "cpu"
    formant_vocals: bool = True
    keep_native_rate: bool = True
    model_override: Optional[str] = None
    allow_download: bool = True


@dataclass
class Prepared:
    """디코딩까지 끝난 입력 (같은 입력으로 다시 실행할 때 재사용)."""
    source_id: str
    kind: str                         # file / youtube
    source_path: Path                 # 원본 파일 (YouTube 는 받은 스트림 파일)
    title: str
    info: audio_io.AudioInfo
    native_path: Path
    native_rate: int
    mix44_path: Path
    n_samples: int                    # 44.1kHz 기준 샘플 수
    peaks: np.ndarray
    youtube: Optional[youtube.AcquiredAudio] = None
    resample_steps: list = field(default_factory=list)


@dataclass
class JobResult:
    files: list
    prepared: Prepared
    separation: engine.SeparationResult
    export: ExportReport
    report: dict
    reused_separation: bool = False


def separation_key(source_id: str, models_used: str, preset: str) -> tuple:
    """같은 입력·같은 모델·같은 처리 방식이면 분리 결과를 재사용."""
    return (source_id, models_used, preset)


def source_kind(text: str) -> str:
    """'youtube' / 'file' / 'invalid_url' / ''"""
    t = (text or "").strip()
    if not t:
        return ""
    if youtube.video_id(t):
        return "youtube"
    if youtube.looks_like_url(t):
        return "invalid_url"
    return "file"


def source_id_for(text: str) -> str:
    t = (text or "").strip()
    vid = youtube.video_id(t)
    if vid:
        return f"yt:{vid}"
    p = Path(t)
    try:
        st = p.stat()
        return f"file:{p.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return f"file:{t}"


class Pipeline:
    def __init__(self, spec: JobSpec, work_dir: Path,
                 progress: Optional[Callable[[str, float, float, str], None]] = None,
                 notice: Optional[Callable[[str], None]] = None,
                 cancel: Optional[threading.Event] = None):
        self.spec = spec
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self._progress = progress or (lambda s, f, o, m: None)
        self._notice = notice or (lambda m: None)
        self.cancel = cancel or threading.Event()
        self.stage = "check"
        offs, acc = {}, 0.0
        for k, _, w in STAGES:
            offs[k] = acc
            acc += w
        self._off = offs
        self._w = {k: w for k, _, w in STAGES}
        self.t0 = time.monotonic()

    # ------------------------------------------------------------ helpers ---
    def report(self, frac: float, msg: str = ""):
        frac = max(0.0, min(1.0, frac))
        overall = self._off[self.stage] + self._w[self.stage] * frac
        self._progress(self.stage, frac, min(0.999, overall), msg or STAGE_LABEL[self.stage])

    def enter(self, stage: str, msg: str = ""):
        self.check_cancel()
        self.stage = stage
        self.report(0.0, msg)

    def check_cancel(self):
        if self.cancel.is_set():
            raise PipelineCancelled()

    # ----------------------------------------------------------- 1~4 단계 ---
    def prepare(self, previous: Optional[Prepared] = None) -> Prepared:
        spec = self.spec
        self.enter("check", "입력 확인 중…")
        kind = source_kind(spec.source)
        if kind == "":
            raise PipelineError("음원 파일을 선택하거나 YouTube URL을 입력해 주세요.", "check", "empty", False)
        if kind == "invalid_url":
            raise PipelineError("올바른 YouTube 영상 주소가 아닙니다.\n예: https://www.youtube.com/watch?v=…",
                                "check", "invalid_url", False)
        sid = source_id_for(spec.source)
        if previous is not None and previous.source_id == sid and previous.native_path.exists() \
                and previous.mix44_path.exists():
            self.stage = "decode"
            self.report(1.0, "이전에 불러온 음원을 사용합니다")
            return previous
        acquired = None
        if kind == "youtube":
            self.enter("acquire", "YouTube 영상 정보 확인 중…")
            dl_dir = self.work / "download"
            try:
                acquired = youtube.acquire(spec.source, dl_dir, lambda f, m: self.report(f, m), self.cancel)
            except youtube.YouTubeCancelled as exc:
                raise PipelineCancelled() from exc
            except youtube.YouTubeError as exc:
                raise PipelineError(str(exc), "acquire", exc.kind, exc.retryable,
                                    can_update_youtube=exc.kind in ("outdated", "unknown", "bot")) from exc
            path = acquired.path
            title = acquired.title
        else:
            path = Path(spec.source.strip().strip('"'))
            if not path.is_file():
                raise PipelineError(f"파일을 찾을 수 없습니다:\n{path}", "check", "not_found", False)
            title = path.stem
            if path.suffix.lower() not in audio_io.SUPPORTED_EXTENSIONS:
                log.info("unusual extension %s, trying anyway", path.suffix)
        self.check_cancel()
        self.enter("decode", "오디오 정보 확인 중…")
        info = audio_io.probe_audio(path)
        if not info.codec:
            raise PipelineError("오디오를 찾을 수 없거나 지원하지 않는 형식입니다.\n"
                                "MP3, WAV, FLAC, M4A, OGG, OPUS 등 일반적인 음원 파일을 사용해 주세요.",
                                "decode", "unsupported", False)
        if info.duration and info.duration > 60 * 60:
            raise PipelineError("음원이 너무 깁니다 (1시간 초과). 곡 단위로 잘라서 사용해 주세요.", "decode", "too_long", False)
        if info.duration:
            need = system.stem_bytes(system.samples_for(info.duration)) * 6 + 200 * system.MB
            free = system.free_disk(self.work)
            if free < need:
                raise PipelineError(f"임시 저장 공간이 부족합니다.\n필요: {system.human_bytes(need)}, "
                                    f"남은 공간: {system.human_bytes(free)}", "decode", "disk", False)
        self.report(0.15, f"디코딩 중… ({info.summary()})")
        try:
            native = audio_io.decode_audio(path, None)
        except audio_io.AudioError as exc:
            raise PipelineError(str(exc), "decode", "ffmpeg", False) from exc
        except MemoryError as exc:
            raise PipelineError("메모리가 부족해 음원을 읽지 못했습니다. 더 짧은 음원을 사용해 주세요.",
                                "decode", "memory", False) from exc
        self.check_cancel()
        rate = info.sample_rate or audio_io.SAMPLE_RATE
        stamp = int(time.time() * 1000)
        native_path = self.work / f"native_{rate}_{stamp}.npy"
        np.save(native_path, native)
        steps = []
        if rate != engine.MODEL_RATE:
            self.report(0.6, f"AI 입력용 변환 중 ({rate / 1000:g} → 44.1kHz, {audio_io.RESAMPLER})…")
            mix44 = audio_io.resample(native, rate, engine.MODEL_RATE)
            mix44_path = self.work / f"mix44_{stamp}.npy"
            np.save(mix44_path, mix44)
            steps.append(f"AI 입력: {rate / 1000:g}kHz → 44.1kHz ({audio_io.RESAMPLER}, 1회)")
        else:
            mix44, mix44_path = native, native_path
        n = len(mix44)
        peaks = compute_peaks(mix44)
        del native, mix44
        self.report(1.0, "디코딩 완료")
        return Prepared(sid, kind, path, title, info, native_path, rate, mix44_path, n, peaks, acquired, steps)

    # ----------------------------------------------------------- 5~6 단계 ---
    def separate(self, prep: Prepared, previous: Optional[engine.SeparationResult] = None,
                 previous_key: Optional[tuple] = None) -> tuple[engine.SeparationResult, bool]:
        spec = self.spec
        key = self.separation_key(prep)
        if previous is not None and previous_key == key and all(Path(s.path).exists() for s in previous.stems):
            self.stage = "separate"
            self.report(1.0, "이전 분리 결과를 사용합니다")
            return previous, True
        self.enter("model", "AI 모델 확인 중…")
        names = engine.required_models(spec.engine, "2", spec.preset, spec.model_override)
        for name in names:
            if models.is_installed(name):
                continue
            if not spec.allow_download:
                raise PipelineError(f"AI 모델({name})이 설치되어 있지 않습니다.", "model", "model_missing",
                                    False, can_use_fast_model=True)
            mb = models.MODELS[name].approx_mb if name in models.MODELS else 0
            free = system.free_disk(models.user_models_dir())
            if free < mb * system.MB * 1.2:
                raise PipelineError(f"AI 모델을 저장할 공간이 부족합니다.\n필요: 약 {mb}MB, "
                                    f"남은 공간: {system.human_bytes(free)}", "model", "disk", False)

            def dl(done, total, label, name=name):
                self.report(done / max(1, total),
                            f"AI 모델 다운로드 중 (처음 한 번) {done / 1048576:.0f} / {total / 1048576:.0f} MB")
            try:
                models.download_model(name, dl, self.cancel)
            except models.DownloadCancelled as exc:
                models.remove_partial_downloads()
                raise PipelineCancelled() from exc
            except Exception as exc:  # noqa: BLE001
                models.remove_partial_downloads()
                raise PipelineError(engine.friendly_error(exc, "AI 모델 다운로드"), "model", "model_download",
                                    True, can_use_fast_model=name != "htdemucs_ft_vocals") from exc
        self.report(1.0, "AI 모델 준비 완료")
        self.enter("separate", "AI 음원 분리 준비 중…")
        run_dir = self.work / f"sep_{int(time.time() * 1000)}"
        mix = np.load(prep.mix44_path, mmap_mode="r")
        try:
            res = engine.separate(mix, "2", run_dir, engine=spec.engine, preset=spec.preset, device=spec.device,
                                  model_name=spec.model_override,
                                  progress=lambda f, m: self.report(f, m), cancel=self.cancel,
                                  notice=self._notice)
        except engine.SeparationCancelled as exc:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise PipelineCancelled() from exc
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(run_dir, ignore_errors=True)
            log.exception("separation failed")
            raise PipelineError(engine.friendly_error(exc), "separate", "separate", True,
                                can_use_fast_model=engine.resolve_engine(spec.engine, spec.preset,
                                                                         spec.model_override) == "roformer") from exc
        finally:
            del mix
        return res, False

    def separation_key(self, prep: Prepared) -> tuple:
        s = self.spec
        return separation_key(prep.source_id, " + ".join(engine.required_models(s.engine, "2", s.preset,
                                                                                 s.model_override)), s.preset)

    # ---------------------------------------------------------- 7~11 단계 ---
    def export(self, prep: Prepared, res: engine.SeparationResult) -> ExportReport:
        spec = self.spec
        self.enter("key" if spec.semitones else "encode",
                   f"키 조절 중 ({pitch.key_label(spec.semitones)}, {pitch.engine_name()})…" if spec.semitones
                   else "저장 준비 중…")
        src = ExportSource(stems={st.key: st.path for st in res.stems}, residual_key=res.stems[-1].key,
                           mix44_path=prep.mix44_path, native_path=prep.native_path, native_rate=prep.native_rate,
                           model_rate=res.model_rate)
        opts = ExportOptions(keep_native_rate=spec.keep_native_rate, semitones=spec.semitones,
                             formant_vocals=spec.formant_vocals)
        jobs = self.jobs(prep, res)
        song_dir = Path(spec.out_dir) / audio_io.safe_filename(prep.title, 60)
        try:
            song_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PipelineError(f"저장 폴더를 만들 수 없습니다:\n{song_dir}\n({exc})\n"
                                "고급 옵션의 [설정]에서 다른 저장 위치를 선택해 주세요.", "encode", "output", False) from exc
        rate = prep.native_rate if spec.keep_native_rate else engine.MODEL_RATE
        n = int(prep.n_samples * rate / engine.MODEL_RATE)
        need = sum(audio_io.estimate_export_bytes(n, j.fmt, rate) for j in jobs) + n * 8 * (len(jobs) + 4)
        if system.free_disk(song_dir) < need:
            raise PipelineError(f"저장 공간이 부족합니다.\n필요: {system.human_bytes(need)}, "
                                f"남은 공간: {system.human_bytes(system.free_disk(song_dir))}", "encode", "disk", False)

        def prog(f, m):
            if spec.semitones and f < 0.6:
                if self.stage != "key":
                    self.stage = "key"
                self.report(f / 0.6, m)
            else:
                if self.stage != "encode":
                    self.stage = "encode"
                self.report((f - 0.6) / 0.4 if spec.semitones else f, m)
        try:
            return Exporter(src, opts, self.work, progress=prog, cancel=self.cancel).run(jobs)
        except (ExportCancelled, pitch.PitchCancelled) as exc:
            for j in jobs:
                try:
                    if j.path.exists():
                        j.path.unlink()
                except OSError:
                    pass
            raise PipelineCancelled() from exc
        except pitch.PitchError as exc:
            raise PipelineError(f"키 변경에 실패했습니다.\n{exc}", "key", "key", True) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("export failed")
            for j in jobs:
                try:
                    if j.path.exists() and j.path.stat().st_size < 1024:
                        j.path.unlink()
                except OSError:
                    pass
            raise PipelineError(engine.friendly_error(exc, "저장"), self.stage, "output", True) from exc

    def jobs(self, prep: Prepared, res: engine.SeparationResult) -> list[ExportJob]:
        spec = self.spec
        ext = audio_io.EXPORT_FORMATS[spec.fmt][1]
        song = audio_io.safe_filename(prep.title, 60)
        song_dir = Path(spec.out_dir) / song
        key = f" [Key {spec.semitones:+d}]" if spec.semitones else ""
        out = []
        if spec.target in ("vocals", "both"):
            p = audio_io.unique_path(song_dir / audio_io.safe_filename(f"{song} - 보컬 (Vocals){key}{ext}"))
            out.append(ExportJob(p, spec.fmt, "보컬", "stem", key="vocals"))
        if spec.target in ("mr", "both"):
            p = audio_io.unique_path(song_dir / audio_io.safe_filename(f"{song} - MR (보컬 제거){key}{ext}"))
            out.append(ExportJob(p, spec.fmt, "MR", "mr", excluded=["vocals"], gains={}))
        return out

    # ------------------------------------------------------------- 전체 ---
    def run(self, previous_prep: Optional[Prepared] = None,
            previous_sep: Optional[engine.SeparationResult] = None,
            previous_sep_key: Optional[tuple] = None) -> JobResult:
        prep = self.prepare(previous_prep)
        res, reused = self.separate(prep, previous_sep, previous_sep_key)
        exp = self.export(prep, res)
        self.stage = "encode"
        self._progress("done", 1.0, 1.0, "완료")
        return JobResult(exp.files, prep, res, exp, self.build_report(prep, res, exp, reused), reused)

    def build_report(self, prep: Prepared, res: engine.SeparationResult, exp: ExportReport, reused: bool) -> dict:
        info = prep.info
        r = {
            "input": {
                "kind": prep.kind, "title": prep.title, "codec": info.codec, "sample_rate": info.sample_rate,
                "channels": info.channels, "channel_layout": info.channel_layout, "bit_rate_kbps": info.bit_rate_kbps,
                "duration": round(info.duration, 2), "container": info.container, "lossless": info.lossless,
            },
            "decode": {"native_rate": prep.native_rate, "resample_steps": prep.resample_steps,
                       "intermediate": "float32 PCM (.npy, 무손실)"},
            "separation": {"engine": res.engine, "model": res.model, "preset": res.preset, "device": res.device,
                           "elapsed_sec": round(res.elapsed, 1), "model_rate": res.model_rate,
                           "reused": reused, "stems_sum_to_original": True},
            "outputs": [{
                "file": str(d.path), "format": d.fmt, "sample_rate": d.sample_rate, "seconds": round(d.seconds, 3),
                "channels": d.channels, "peak_dbfs": round(d.peak_db, 2),
                "true_peak_dbtp": None if d.true_peak_db is None else round(d.true_peak_db, 2),
                "resampled_from": d.resampled_from, "semitones": d.semitones, "formant": d.formant,
                "verified": d.verified} for d in exp.details],
            "gain_db": round(exp.gain_db, 2),
            "key_engine": exp.key_engine,
            "lossy_encodes": sum(1 for d in exp.details if d.fmt == "mp3"),
            "total_sec": round(time.monotonic() - self.t0, 1),
        }
        if prep.youtube is not None:
            y = prep.youtube
            r["youtube"] = {"video_id": y.video_id, "format_id": y.format_id, "codec": y.codec,
                            "codec_name": y.codec_name, "abr_kbps": round(y.abr_kbps, 1), "sample_rate": y.sample_rate,
                            "channels": y.channels, "container": y.container, "audio_only": y.audio_only,
                            "reencoded": y.reencoded, "reason": y.reason, "license": y.license,
                            "filesize": y.filesize, "candidates": y.table}
        return r


def estimate_seconds(n_samples: int, eng: str, device: str, preset: str) -> float:
    """대략적인 분리 시간 (초). 표시용."""
    sec = n_samples / engine.MODEL_RATE
    import os

    cores = max(1, (os.cpu_count() or 4))
    p = engine.PRESETS.get(preset, engine.PRESETS[engine.DEFAULT_PRESET])
    if eng == "roformer":
        factor = {"cuda": 0.15, "mps": 0.5}.get(device, 20.0 / cores * 2)
        factor *= p.roformer_overlap / 2
    else:
        factor = {"cuda": 0.05, "mps": 0.2}.get(device, 2.5 / cores * 2) * p.demucs_shifts
    return sec * factor
