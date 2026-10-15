"""[실험] 듀엣 가수별(남/여) 분리.

1) 기존 파이프라인으로 보컬/반주 분리 (같은 입력이면 이전 결과 재사용) → MR 저장
2) 보컬 트랙을 남/여 분리 모델(BS-RoFormer, Sucial)로 다시 분리 → 남자 보컬 / 여자 보컬 저장

남녀가 동시에 부르는 구간(화음)을 위한 모델이라, 번갈아 부르는 곡이나 같은 성별 듀엣은 잘 나뉘지 않을 수 있다.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio_io, engine, models, pipeline, system

log = logging.getLogger(__name__)

MODEL_NAME = "duet_male_female"
LABELS = {"male": "남자 보컬", "female": "여자 보컬"}
PEAK_CEILING = 10 ** (-0.1 / 20)

ProgressCb = Callable[[float, str], None]


@dataclass
class DuetResult:
    files: list
    prepared: pipeline.Prepared
    separation: engine.SeparationResult
    separation_key: tuple
    seconds: float = 0.0
    notes: list = field(default_factory=list)


def run(spec: pipeline.JobSpec, work_dir: Path, progress: Optional[ProgressCb] = None,
        cancel: Optional[threading.Event] = None, previous_prep=None, previous_sep=None,
        previous_sep_key=None, model_name: str = MODEL_NAME) -> DuetResult:
    """spec 의 입력·처리 방식·장치·저장 형식으로 실행. spec.target 은 무시하고 MR + 남/여 보컬을 저장."""
    t0 = time.monotonic()
    cancel = cancel or threading.Event()
    report = progress or (lambda f, m: None)

    def stage1(stage, frac, overall, msg):           # 보컬/반주 분리 + MR 저장 → 전체의 0 ~ 45%
        report(0.45 * overall, msg)

    spec = replace(spec, target="mr")
    p = pipeline.Pipeline(spec, Path(work_dir), progress=stage1, cancel=cancel)
    prep = p.prepare(previous_prep)
    sep, _reused = p.separate(prep, previous_sep, previous_sep_key)
    key = p.separation_key(prep)
    mr = p.export(prep, sep)
    files = list(mr.files)

    # 2) 남/여 분리 모델
    if not models.is_installed(model_name):
        mb = models.MODELS[model_name].approx_mb
        if system.free_disk(models.user_models_dir()) < mb * system.MB * 1.2:
            raise pipeline.PipelineError(f"AI 모델을 저장할 공간이 부족합니다 (약 {mb}MB 필요).", "model", "disk", False)
        try:
            models.download_model(model_name, lambda d, t, lbl: report(
                0.45 + 0.05 * d / max(1, t), f"남/여 분리 모델 다운로드 중 (처음 한 번) {d / 1048576:.0f} / "
                                            f"{t / 1048576:.0f} MB"), cancel)
        except models.DownloadCancelled as exc:
            models.remove_partial_downloads()
            raise pipeline.PipelineCancelled() from exc
        except Exception as exc:  # noqa: BLE001
            models.remove_partial_downloads()
            raise pipeline.PipelineError(engine.friendly_error(exc, "AI 모델 다운로드"), "model", "model_download",
                                         True) from exc
    report(0.5, "남/여 분리 모델 불러오는 중…")
    from . import roformer

    model = models.load_model(model_name)
    vocals_path = next(s.path for s in sep.stems if s.key == "vocals")
    vocals = np.load(vocals_path, mmap_mode="r")
    device = spec.device if spec.device in ("cuda", "mps") else "cpu"
    try:
        out = roformer.demix(model, vocals, device, num_overlap=getattr(model, "num_overlap", 2),
                             chunk=getattr(model, "chunk", roformer.SW_CHUNK),
                             progress=lambda f: report(0.5 + 0.45 * f, f"남/여 보컬 분리 중 {f * 100:.0f}%"),
                             cancel=cancel)
    except roformer.Cancelled as exc:
        raise pipeline.PipelineCancelled() from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("duet separation failed")
        raise pipeline.PipelineError(engine.friendly_error(exc), "separate", "separate", True) from exc
    finally:
        del vocals

    # 3) 저장 (보컬 분리 결과와 같은 44.1kHz, 두 트랙에 같은 음량 배율 → 둘을 더하면 원래 보컬)
    report(0.96, "저장 중…")
    song = audio_io.safe_filename(prep.title, 60)
    song_dir = Path(spec.out_dir) / song
    song_dir.mkdir(parents=True, exist_ok=True)
    ext = audio_io.EXPORT_FORMATS[spec.fmt][1]
    peak = max(float(np.max(np.abs(o))) if len(o) else 0.0 for o in out)
    gain = 1.0 if spec.fmt in audio_io.FLOAT_FORMATS or peak <= PEAK_CEILING else PEAK_CEILING / peak
    notes = []
    if gain < 1.0:
        notes.append(f"0dBFS 를 넘는 부분이 있어 남/여 보컬 음량을 {20 * np.log10(gain):.1f}dB 낮췄습니다.")
    for inst, data in zip(model.instruments, out):
        label = LABELS.get(inst.lower(), inst)
        path = audio_io.unique_path(song_dir / audio_io.safe_filename(f"{song} - {label} (실험){ext}"))
        audio_io.write_audio(path, data * np.float32(gain) if gain != 1.0 else data, spec.fmt, engine.MODEL_RATE)
        files.append(path)
    report(1.0, "완료")
    return DuetResult(files, prep, sep, key, time.monotonic() - t0, notes)
