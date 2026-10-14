"""백그라운드 작업 스레드 (UI 가 멈추지 않도록 모든 무거운 작업을 여기서 처리)."""
from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal

from .. import audio_io, engine, models, pipeline, pitch, youtube
from ..dsp import compute_peaks
from ..export import ExportCancelled, Exporter, ExportJob, ExportOptions, ExportSource  # noqa: F401

log = logging.getLogger(__name__)


@dataclass
class LoadedAudio:
    source: Path
    mix_path: Path          # 44.1kHz (AI 처리·재생용)
    n_samples: int
    peaks: np.ndarray
    native_rate: int = audio_io.SAMPLE_RATE
    native_path: Optional[Path] = None   # 원본 샘플레이트 사본 (44.1kHz 원본이면 mix_path)
    title: Optional[str] = None          # 표시·저장용 이름 (YouTube 제목 등)
    source_id: str = ""

    @property
    def seconds(self) -> float:
        return self.n_samples / audio_io.SAMPLE_RATE

    @property
    def name(self) -> str:
        return self.title or self.source.stem


class LoadThread(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, source: Path, target: Path, parent=None):
        super().__init__(parent)
        self.source = Path(source)
        self.target = Path(target)  # 파일마다 새 이름 (Windows 에서 열려 있는 memmap 덮어쓰기 방지)

    def run(self):
        try:
            # 원본 샘플레이트로 한 번만 디코딩 → 필요할 때만 soxr VHQ 로 44.1kHz 변환 (FFmpeg 리샘플러 미사용)
            native = audio_io.decode_audio(self.source, None)
            rate = audio_io.probe_audio(self.source).sample_rate or audio_io.SAMPLE_RATE
            native_path = self.target
            if rate != audio_io.SAMPLE_RATE:
                native_path = self.target.with_name(self.target.stem + f"_{rate}.npy")
                np.save(native_path, native)
                data = audio_io.resample(native, rate, audio_io.SAMPLE_RATE)
            else:
                data = native
            del native
            np.save(self.target, data)
            peaks = compute_peaks(data)
            n = len(data)
            del data
            self.done.emit(LoadedAudio(self.source, self.target, n, peaks, rate, native_path,
                                       source_id=pipeline.source_id_for(str(self.source))))
        except audio_io.AudioError as exc:
            self.failed.emit(str(exc))
        except MemoryError:
            self.failed.emit("파일이 너무 커서 메모리가 부족합니다. 더 짧은 파일을 사용해 주세요.")
        except Exception as exc:
            log.exception("load failed")
            self.failed.emit(f"파일을 불러오지 못했습니다.\n({type(exc).__name__}: {exc})")


class SeparationThread(QThread):
    progress = Signal(float, str)
    notice = Signal(str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, mix_path: Path, mode: str, preset: str, device: str, out_dir: Path,
                 model_name: Optional[str] = None, engine_key: str = "auto", parent=None):
        super().__init__(parent)
        self.mix_path = Path(mix_path)
        self.mode = mode
        self.preset = preset
        self.device = device
        self.out_dir = Path(out_dir)
        self.model_name = model_name
        self.engine_key = engine_key
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            need = [m for m in engine.required_models(self.engine_key, self.mode, self.preset,
                                                      self.model_name) if not models.is_installed(m)]
            for name in need:
                def dl(done, total, label):
                    self.progress.emit(done / max(1, total),
                                       f"{label}  {done / 1048576:.0f} / {total / 1048576:.0f} MB")
                self.progress.emit(0.0, "AI 모델 다운로드 준비 중…")
                models.download_model(name, dl, self.cancel_event)
            if self.cancel_event.is_set():
                raise engine.SeparationCancelled()
            mix = np.load(self.mix_path, mmap_mode="r")
            result = engine.separate(
                mix, self.mode, self.out_dir, engine=self.engine_key, preset=self.preset,
                device=self.device, model_name=self.model_name,
                progress=lambda f, m: self.progress.emit(f, m),
                cancel=self.cancel_event,
                notice=lambda m: self.notice.emit(m),
            )
            del mix
            self.done.emit(result)
        except (engine.SeparationCancelled, models.DownloadCancelled):
            models.remove_partial_downloads()
            self.cancelled.emit()
        except BaseException as exc:  # noqa: BLE001 - 어떤 오류든 UI 에 전달
            log.error("separation failed: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(engine.friendly_error(exc))


class ExportThread(QThread):
    progress = Signal(float, str)
    done = Signal(object)        # ExportReport
    failed = Signal(str)

    def __init__(self, source: ExportSource, jobs: list[ExportJob], options: ExportOptions, work_dir: Path,
                 parent=None):
        super().__init__(parent)
        self.source, self.jobs, self.options, self.work_dir = source, jobs, options, Path(work_dir)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            exp = Exporter(self.source, self.options, self.work_dir,
                           progress=lambda f, m: self.progress.emit(f, m), cancel=self.cancel_event)
            self.done.emit(exp.run(self.jobs))
        except ExportCancelled:
            for j in self.jobs:   # 덜 저장된 파일 정리
                try:
                    if j.path.exists() and j.path.stat().st_size < 1024:
                        j.path.unlink()
                except OSError:
                    pass
            self.failed.emit("저장을 취소했습니다.")
        except BaseException as exc:  # noqa: BLE001
            log.exception("export failed")
            self.failed.emit(engine.friendly_error(exc, "저장"))


class PipelineThread(QThread):
    """홈 화면 원클릭 작업."""

    progress = Signal(str, float, str)      # stage, overall, message
    notice = Signal(str)
    done = Signal(object)                   # pipeline.JobResult
    failed = Signal(object)                 # pipeline.PipelineError
    cancelled = Signal()

    def __init__(self, spec: pipeline.JobSpec, work_dir: Path, previous_prep=None, previous_sep=None,
                 previous_key=None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.work_dir = Path(work_dir)
        self.prev = (previous_prep, previous_sep, previous_key)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            p = pipeline.Pipeline(self.spec, self.work_dir,
                                  progress=lambda st, f, o, m: self.progress.emit(st, o, m),
                                  notice=lambda m: self.notice.emit(m), cancel=self.cancel_event)
            res = p.run(*self.prev)
            res.separation_key = p.separation_key(res.prepared)
            self.done.emit(res)
        except pipeline.PipelineCancelled:
            self.cancelled.emit()
        except pipeline.PipelineError as exc:
            log.warning("pipeline failed at %s: %s", exc.stage, exc)
            self.failed.emit(exc)
        except BaseException as exc:  # noqa: BLE001
            log.error("pipeline crashed: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(pipeline.PipelineError(engine.friendly_error(exc, "처리"), "", "unknown", True))


class YouTubeUpdateThread(QThread):
    progress = Signal(float, str)
    done = Signal(str)
    failed = Signal(str)

    def run(self):
        try:
            ver = youtube.update_downloader(
                lambda d, t: self.progress.emit(d / t if t else 0.0, f"YouTube 기능 업데이트 중… {d / 1048576:.1f} MB"))
            self.done.emit(ver)
        except Exception as exc:  # noqa: BLE001
            log.exception("yt-dlp update failed")
            self.failed.emit(str(exc) if isinstance(exc, youtube.YouTubeError)
                             else f"업데이트를 받지 못했습니다. 인터넷 연결을 확인해 주세요.\n({exc})")


class YouTubeLoadThread(QThread):
    """고급 옵션 화면: YouTube 오디오를 받아서 불러오기 (분리는 하지 않음)."""

    progress = Signal(float, str)
    done = Signal(object, object)           # LoadedAudio, AcquiredAudio
    failed = Signal(str)

    def __init__(self, url: str, work_dir: Path, target: Path, parent=None):
        super().__init__(parent)
        self.url, self.work_dir, self.target = url, Path(work_dir), Path(target)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            acq = youtube.acquire(self.url, self.work_dir / "download", lambda f, m: self.progress.emit(f, m),
                                  self.cancel_event)
            lt = LoadThread(acq.path, self.target)
            out: dict = {}
            lt.done.connect(lambda la: out.setdefault("la", la), Qt.ConnectionType.DirectConnection)
            lt.failed.connect(lambda m: out.setdefault("err", m), Qt.ConnectionType.DirectConnection)
            lt.run()
            if "err" in out:
                self.failed.emit(out["err"])
                return
            la = out["la"]
            la.title = acq.title
            la.source_id = pipeline.source_id_for(self.url)
            self.done.emit(la, acq)
        except youtube.YouTubeCancelled:
            self.failed.emit("취소했습니다.")
        except youtube.YouTubeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("youtube load failed")
            self.failed.emit(str(youtube.friendly_error(exc)))


class KeyThread(QThread):
    """재생용 키 변경: 모든 트랙(44.1kHz)을 미리 변환해 임시 파일로 저장."""

    progress = Signal(float, str)
    done = Signal(int, dict)       # semitones, {key: npy 경로}
    failed = Signal(str)

    def __init__(self, tracks: dict, semitones: int, keep_drums: bool, out_dir: Path, parent=None,
                 formant_vocals: bool = True):
        super().__init__(parent)
        self.tracks = dict(tracks)          # key -> 원래 44.1kHz npy 경로
        self.semitones = semitones
        self.keep_drums = keep_drums
        self.formant_vocals = formant_vocals
        self.out_dir = Path(out_dir)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            result: dict = {}
            todo = []
            for key, path in self.tracks.items():
                if self.semitones == 0 or (key == "drums" and self.keep_drums):
                    result[key] = Path(path)
                else:
                    todo.append((key, Path(path)))
            self.out_dir.mkdir(parents=True, exist_ok=True)
            done = [0]
            lock = threading.Lock()
            label = pitch.key_label(self.semitones)

            def work(item):
                key, path = item
                if self.cancel_event.is_set():
                    return None
                data = np.load(path, mmap_mode="r")
                out = pitch.shift(data, self.semitones, audio_io.SAMPLE_RATE,
                                  formant=self.formant_vocals and key == "vocals", cancel=self.cancel_event)
                dst = self.out_dir / f"{key}_k{self.semitones:+d}.npy"
                np.save(dst, out)
                with lock:
                    done[0] += 1
                    self.progress.emit(done[0] / max(1, len(todo)), f"키 변경 중 ({label}) {done[0]}/{len(todo)}")
                return key, dst

            self.progress.emit(0.0, f"키 변경 중 ({label})…")
            if pitch.find_rbshift() and len(todo) > 1:
                # Rubber Band 는 별도 프로세스라 트랙을 동시에 처리 (코어 수만큼)
                import os
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=max(1, min(len(todo), (os.cpu_count() or 2) - 1))) as ex:
                    for r in ex.map(work, todo):
                        if r is not None:
                            result[r[0]] = r[1]
            else:
                # (python-stretch 는 GIL 을 잡고 있어 여러 스레드로 나눠도 빨라지지 않음 → 순서대로 처리)
                for item in todo:
                    r = work(item)
                    if r is not None:
                        result[r[0]] = r[1]
            if self.cancel_event.is_set():
                return
            self.done.emit(self.semitones, result)
        except pitch.PitchCancelled:
            return
        except BaseException as exc:  # noqa: BLE001
            log.exception("key shift failed")
            self.failed.emit(engine.friendly_error(exc, "키 변경"))


class UpdateCheckThread(QThread):
    found = Signal(object)        # updater.UpdateInfo 또는 None
    failed = Signal(str)

    def run(self):
        from .. import updater

        try:
            self.found.emit(updater.check())
        except Exception as exc:  # noqa: BLE001
            log.info("update check failed: %s", exc)
            self.failed.emit(str(exc))


class UpdateDownloadThread(QThread):
    progress = Signal(float)
    done = Signal(str)            # setup.exe 경로
    failed = Signal(str)

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.cancel = threading.Event()

    def run(self):
        from .. import updater

        try:
            path = updater.download(self.info, lambda d, t: self.progress.emit(d / t if t else 0.0), self.cancel)
            self.done.emit(str(path))
        except InterruptedError:
            self.failed.emit("")
        except Exception as exc:  # noqa: BLE001
            log.exception("update download failed")
            self.failed.emit(f"업데이트 파일을 받지 못했습니다. 인터넷 연결을 확인해 주세요.\n({exc})")
