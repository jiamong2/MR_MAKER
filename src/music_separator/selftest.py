"""자가 진단 (--selftest).

빌드된 실행 파일이 각 OS 에서 실제로 동작하는지 CI 에서 확인하기 위한 헤드리스 테스트:
FFmpeg 디코딩 → (모델 다운로드) → Demucs 분리 → WAV/MP3 저장 → 다시 읽어서 검증.
"""
from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np


def _say(msg: str) -> None:
    if sys.stdout is not None:
        try:
            print(msg, flush=True)
        except Exception:
            pass


def make_test_signal(seconds: float = 12.0, sr: int = 44100) -> np.ndarray:
    """보컬 비슷한 하모닉 + 베이스 + 킥 드럼 + 하이햇 노이즈."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 220 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))  # 비브라토
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 8)) * 0.15
    bass = 0.25 * np.sin(2 * np.pi * 55 * t)
    beat = (t % 0.5)
    kick = 0.6 * np.sin(2 * np.pi * 60 * beat) * np.exp(-beat * 25)
    hat = 0.05 * rng.standard_normal(len(t)) * np.exp(-((t + 0.25) % 0.25) * 60)
    left = voice + bass + kick + hat
    right = voice * 0.9 + bass + kick + hat * 1.1
    return np.stack([left, right], 1).astype(np.float32)


def run(model: Optional[str] = None, allow_download: bool = True, report: Optional[str] = None,
        device_pref: str = "auto", mode: str = "4", engine_key: str = "demucs", seconds: float = 12.0) -> int:
    from . import __version__, audio_io, engine, models, pitch, system
    from .export import Exporter, ExportJob, ExportOptions, ExportSource

    rep: dict = {"version": __version__, "platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version(), "checks": {}, "ok": False}
    checks = rep["checks"]
    work = Path(tempfile.mkdtemp(prefix="music-separator-selftest-"))
    t_all = time.monotonic()

    def check(name: str, fn, critical: bool = True):
        t0 = time.monotonic()
        try:
            value = fn()
            checks[name] = {"ok": True, "value": value, "sec": round(time.monotonic() - t0, 2)}
            _say(f"[OK]   {name}: {value}")
            return value
        except Exception as exc:  # noqa: BLE001
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "trace": traceback.format_exc(), "critical": critical}
            _say(f"[FAIL] {name}: {exc}")
            if critical:
                raise
            return None

    try:
        def versions():
            import demucs
            import torch
            from PySide6 import QtCore

            return {"torch": torch.__version__, "demucs": getattr(demucs, "__version__", "?"),
                    "numpy": np.__version__, "qt": QtCore.qVersion()}
        check("versions", versions)
        check("ffmpeg", lambda: audio_io.find_ffmpeg())

        sig = make_test_signal(seconds)
        wav_in = work / "input.wav"
        mp3_in = work / "input.mp3"

        def io_roundtrip():
            audio_io.write_wav(wav_in, sig, 16)
            audio_io.write_mp3(mp3_in, sig)
            a = audio_io.decode_audio(wav_in)
            b = audio_io.decode_audio(mp3_in)
            assert abs(len(a) - len(sig)) == 0, f"wav length {len(a)} != {len(sig)}"
            assert abs(len(b) - len(sig)) < 44100 * 0.2, f"mp3 length {len(b)}"
            dur = audio_io.probe_duration(mp3_in)
            assert dur and abs(dur - seconds) < 0.3, f"probe {dur}"
            return {"wav_samples": len(a), "mp3_samples": len(b), "probe_sec": dur}
        check("audio_io", io_roundtrip)

        check("devices", lambda: [f"{d.key}: {d.label}" for d in system.detect_devices()])
        dev = system.choose_device(device_pref).key
        if model and model.startswith("roformer"):
            engine_key = "roformer"
        names = engine.required_models(engine_key, mode, False, model)
        rep["engine"], rep["models"] = engine_key, names
        real = not any(n.endswith(("_unittest", "_unittest6")) for n in names)

        if real:
            def ensure():
                out = {}
                for name in names:
                    if models.is_installed(name):
                        out[name] = f"installed at {models.find_model_dir(name)}"
                        continue
                    if not allow_download:
                        raise RuntimeError(f"model {name} not installed and download disabled")
                    last = [0.0]

                    def prog(done, total, label):
                        if time.monotonic() - last[0] > 2:
                            last[0] = time.monotonic()
                            _say(f"       {label} {done / 1048576:.0f}/{total / 1048576:.0f}MB")
                    out[name] = f"downloaded to {models.download_model(name, prog)}"
                return out
            check("model", ensure)

        mix = audio_io.decode_audio(wav_in)

        def separate():
            res = engine.separate(mix, mode, work / "stems", engine=engine_key, device=dev, model_name=model)
            stems = {s.key: np.load(s.path) for s in res.stems}
            for k, a in stems.items():
                assert a.shape == mix.shape, f"{k} shape {a.shape}"
                assert np.isfinite(a).all(), f"{k} has NaN"
            total = sum(stems.values())
            sum_err = float(np.max(np.abs(total - mix)))
            assert sum_err < 1e-4, f"stems must sum to the mix exactly (max err {sum_err})"
            out = {"models": names, "engine": engine_key, "device": res.device, "elapsed": round(res.elapsed, 2),
                   "stems": list(stems), "raw_reconstruction_error": round(res.raw_recon_error, 4),
                   "rms": {k: round(float(np.sqrt((a ** 2).mean())), 4) for k, a in stems.items()}}
            if real:
                # 실제 모델이면 모델 출력의 합이 원본과 거의 같아야 함 (가중치가 제대로 로드됐는지 확인)
                assert res.raw_recon_error < 0.35, f"model outputs do not sum to the mix ({res.raw_recon_error:.2f})"
            holder["res"] = res
            return out
        holder: dict = {}
        check("separate", separate)
        res = holder["res"]

        def export():
            # 48kHz 원본을 가정한 사본 → 원본 샘플레이트 저장 경로까지 확인
            native48 = audio_io.resample(mix, 44100, 48000)
            np.save(work / "mix44.npy", mix)
            np.save(work / "native48.npy", native48)
            src = ExportSource({st.key: st.path for st in res.stems}, res.stems[-1].key, work / "mix44.npy",
                               work / "native48.npy", 48000)
            out_dir = work / "export"
            jobs = [ExportJob(out_dir / f"{st.key}_{fmt}{audio_io.EXPORT_FORMATS[fmt][1]}", fmt, st.key, "stem",
                              key=st.key) for st in res.stems for fmt in ("wav24", "wav32f", "flac", "mp3")]
            excluded = [res.stems[0].key]
            jobs.append(ExportJob(out_dir / "MR.wav", "wav32f", "MR", "mr", excluded=excluded,
                                  gains={st.key: 1.0 for st in res.stems if st.key not in excluded}))
            jobs.append(ExportJob(out_dir / "FULL.wav", "wav32f", "FULL", "mr", excluded=[], gains={}))
            rep_ = Exporter(src, ExportOptions(), work).run(jobs)
            for j in jobs:
                d = audio_io.decode_audio(j.path, sample_rate=48000)
                assert abs(len(d) - len(native48)) < 48000 * 0.05, f"{j.path.name}: {len(d)}"
            full = audio_io.decode_audio(out_dir / "FULL.wav", sample_rate=48000)
            full_err = float(np.max(np.abs(full - native48)))
            assert full_err < 1e-6, f"full mix should equal the original exactly ({full_err})"
            mp3 = audio_io.decode_audio(out_dir / f"{res.stems[1].key}_mp3.mp3", sample_rate=48000)
            ref = audio_io.decode_audio(out_dir / f"{res.stems[1].key}_wav32f.wav", sample_rate=48000)
            n = min(len(mp3), len(ref), 48000 * 3)
            c = np.correlate(mp3[:n, 0], ref[:n, 0], "full")
            lag = int(np.argmax(np.abs(c)) - (n - 1)) if np.abs(ref[:n, 0]).max() > 1e-4 else 0
            assert abs(lag) < 5, f"mp3 is not sample-aligned (lag {lag})"
            return {"files": len(jobs), "rate": rep_.sample_rate, "full_mix_max_err": full_err, "mp3_lag": lag}
        check("export", export)

        def key_shift():
            if not pitch.available():
                raise RuntimeError("python-stretch 를 불러올 수 없습니다")
            sr = 44100
            t = np.arange(sr * 2) / sr
            tone = np.stack([0.4 * np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
            y = pitch.shift(tone, 3, sr)
            seg = y[sr // 2: sr + sr // 2, 0]
            f = np.fft.rfftfreq(len(seg), 1 / sr)[np.argmax(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))))]
            assert abs(f - 440 * 2 ** (3 / 12)) < 3, f"pitch shift wrong: {f:.1f}Hz"
            assert len(y) == len(tone)
            return {"440Hz +3 ->": round(float(f), 1)}
        check("key_shift", key_shift)

        def audio_device():
            import sounddevice as sd

            return {"portaudio": sd.get_portaudio_version()[1],
                    "outputs": sum(1 for d in sd.query_devices() if d["max_output_channels"] > 0)}
        check("sounddevice", audio_device, critical=False)
        rep["ok"] = all(c["ok"] for c in checks.values() if c.get("critical", True))
    except Exception:
        rep["ok"] = False
    rep["total_sec"] = round(time.monotonic() - t_all, 1)
    _say(f"SELFTEST {'PASSED' if rep['ok'] else 'FAILED'} in {rep['total_sec']}s")
    if report:
        Path(report).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0 if rep["ok"] else 1
