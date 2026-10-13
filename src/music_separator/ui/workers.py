"""백그라운드 작업 스레드 (UI 가 멈추지 않도록 모든 무거운 작업을 여기서 처리)."""
from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from .. import audio_io, engine, models
from ..dsp import compute_peaks, mixdown

log = logging.getLogger(__name__)


@dataclass
class LoadedAudio:
    source: Path
    mix_path: Path
    n_samples: int
    peaks: np.ndarray

    @property
    def seconds(self) -> float:
        return self.n_samples / audio_io.SAMPLE_RATE


class LoadThread(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, source: Path, target: Path, parent=None):
        super().__init__(parent)
        self.source = Path(source)
        self.target = Path(target)  # 파일마다 새 이름 (Windows 에서 열려 있는 memmap 덮어쓰기 방지)

    def run(self):
        try:
            data = audio_io.decode_audio(self.source)
            path = self.target
            np.save(path, data)
            peaks = compute_peaks(data)
            n = len(data)
            del data
            self.done.emit(LoadedAudio(self.source, path, n, peaks))
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

    def __init__(self, mix_path: Path, mode: str, high_quality: bool, device: str, out_dir: Path,
                 model_name: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.mix_path = Path(mix_path)
        self.mode = mode
        self.high_quality = high_quality
        self.device = device
        self.out_dir = Path(out_dir)
        self.model_name = model_name or engine.model_for(mode, high_quality)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            if self.model_name != "demucs_unittest" and not models.is_installed(self.model_name):
                def dl(done, total, label):
                    self.progress.emit(done / max(1, total),
                                       f"{label}  {done / 1048576:.0f} / {total / 1048576:.0f} MB")
                self.progress.emit(0.0, "AI 모델 다운로드 준비 중…")
                models.download_model(self.model_name, dl, self.cancel_event)
            if self.cancel_event.is_set():
                raise engine.SeparationCancelled()
            mix = np.load(self.mix_path, mmap_mode="r")
            result = engine.separate(
                mix, self.mode, self.out_dir, high_quality=self.high_quality, device=self.device,
                model_name=self.model_name,
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


@dataclass
class ExportJob:
    path: Path
    sources: list            # [(np.ndarray, gain)]
    fmt: str
    title: str


class ExportThread(QThread):
    progress = Signal(float, str)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, jobs: list[ExportJob], parent=None):
        super().__init__(parent)
        self.jobs = jobs

    def run(self):
        written: list[Path] = []
        try:
            n = len(self.jobs)
            for i, job in enumerate(self.jobs):
                base = i / n

                def prog(f, i=i, job=job, base=base):
                    self.progress.emit(base + f / n, f"저장 중 ({i + 1}/{n}) {job.title}")

                if len(job.sources) == 1 and job.sources[0][1] == 1.0:
                    data = job.sources[0][0]
                else:
                    data = mixdown([s for s, _ in job.sources], [g for _, g in job.sources])
                job.path.parent.mkdir(parents=True, exist_ok=True)
                audio_io.write_audio(job.path, data, job.fmt, progress=prog)
                written.append(job.path)
            self.progress.emit(1.0, "저장 완료")
            self.done.emit(written)
        except BaseException as exc:  # noqa: BLE001
            log.exception("export failed")
            self.failed.emit(engine.friendly_error(exc, "저장"))
