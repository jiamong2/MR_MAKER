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

from .. import audio_io, engine, models, pitch
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
            np.save(self.target, data)
            peaks = compute_peaks(data)
            n = len(data)
            del data
            rate = audio_io.probe_sample_rate(self.source) or audio_io.SAMPLE_RATE
            native_path = self.target
            if rate != audio_io.SAMPLE_RATE and 8000 <= rate <= 384000:
                native = audio_io.decode_audio(self.source, sample_rate=rate)
                native_path = self.target.with_name(self.target.stem + f"_{rate}.npy")
                np.save(native_path, native)
                del native
            else:
                rate = audio_io.SAMPLE_RATE
            self.done.emit(LoadedAudio(self.source, self.target, n, peaks, rate, native_path))
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
                 model_name: Optional[str] = None, engine_key: str = "demucs", parent=None):
        super().__init__(parent)
        self.mix_path = Path(mix_path)
        self.mode = mode
        self.high_quality = high_quality
        self.device = device
        self.out_dir = Path(out_dir)
        self.model_name = model_name
        self.engine_key = engine_key
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            need = [m for m in engine.required_models(self.engine_key, self.mode, self.high_quality,
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
                mix, self.mode, self.out_dir, engine=self.engine_key, high_quality=self.high_quality,
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


class KeyThread(QThread):
    """재생용 키 변경: 모든 트랙(44.1kHz)을 미리 변환해 임시 파일로 저장."""

    progress = Signal(float, str)
    done = Signal(int, dict)       # semitones, {key: npy 경로}
    failed = Signal(str)

    def __init__(self, tracks: dict, semitones: int, keep_drums: bool, out_dir: Path, parent=None):
        super().__init__(parent)
        self.tracks = dict(tracks)          # key -> 원래 44.1kHz npy 경로
        self.semitones = semitones
        self.keep_drums = keep_drums
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
                out = pitch.shift(data, self.semitones, audio_io.SAMPLE_RATE)
                dst = self.out_dir / f"{key}_k{self.semitones:+d}.npy"
                np.save(dst, out)
                with lock:
                    done[0] += 1
                    self.progress.emit(done[0] / max(1, len(todo)), f"키 변경 중 ({label}) {done[0]}/{len(todo)}")
                return key, dst

            self.progress.emit(0.0, f"키 변경 중 ({label})…")
            # (python-stretch 는 GIL 을 잡고 있어 여러 스레드로 나눠도 빨라지지 않음 → 순서대로 처리)
            for item in todo:
                r = work(item)
                if r is not None:
                    result[r[0]] = r[1]
            if self.cancel_event.is_set():
                return
            self.done.emit(self.semitones, result)
        except BaseException as exc:  # noqa: BLE001
            log.exception("key shift failed")
            self.failed.emit(engine.friendly_error(exc, "키 변경"))
