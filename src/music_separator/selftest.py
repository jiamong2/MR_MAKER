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
        device_pref: str = "auto", mode: str = "4") -> int:
    from . import __version__, audio_io, engine, models, system
    from .dsp import mixdown

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

        sig = make_test_signal()
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
            assert dur and abs(dur - 12.0) < 0.3, f"probe {dur}"
            return {"wav_samples": len(a), "mp3_samples": len(b), "probe_sec": dur}
        check("audio_io", io_roundtrip)

        check("devices", lambda: [f"{d.key}: {d.label}" for d in system.detect_devices()])
        dev = system.choose_device(device_pref).key
        name = model or engine.model_for(mode, False)

        if name != "demucs_unittest":
            def ensure():
                if models.is_installed(name):
                    return f"installed at {models.find_model_dir(name)}"
                if not allow_download:
                    raise RuntimeError("model not installed and download disabled")
                last = [0.0]

                def prog(done, total, label):
                    if time.monotonic() - last[0] > 2:
                        last[0] = time.monotonic()
                        _say(f"       {label} {done / 1048576:.0f}/{total / 1048576:.0f}MB")
                folder = models.download_model(name, prog)
                return f"downloaded to {folder}"
            check("model", ensure)

        mix = audio_io.decode_audio(wav_in)

        def separate():
            res = engine.separate(mix, mode, work / "stems", device=dev, model_name=name)
            stems = {s.key: np.load(s.path) for s in res.stems}
            for k, a in stems.items():
                assert a.shape == mix.shape, f"{k} shape {a.shape}"
                assert np.isfinite(a).all(), f"{k} has NaN"
            out = {"model": name, "device": res.device, "elapsed": round(res.elapsed, 2),
                   "stems": list(stems), "rms": {k: round(float(np.sqrt((a ** 2).mean())), 4)
                                                 for k, a in stems.items()}}
            if name != "demucs_unittest":
                recon = sum(stems.values())
                err = float(np.linalg.norm(recon - mix) / (np.linalg.norm(mix) + 1e-9))
                out["reconstruction_error"] = round(err, 4)
                assert err < 0.35, f"stems do not sum to the mix (rel. error {err:.2f})"
            holder["res"] = res
            return out
        holder: dict = {}
        check("separate", separate)
        res = holder["res"]

        def export():
            written = []
            stems = {s.key: np.load(s.path, mmap_mode="r") for s in res.stems}
            for k, a in stems.items():
                for fmt in ("wav16", "wav24", "mp3"):
                    p = work / "export" / f"{k}{audio_io.EXPORT_FORMATS[fmt][1]}".replace(".", f"_{fmt}.", 1)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    audio_io.write_audio(p, a, fmt)
                    written.append(p)
            mr = mixdown([a for k, a in stems.items() if k != "vocals"], [1.0] * (len(stems) - 1))
            p = work / "export" / "MR.wav"
            audio_io.write_wav(p, mr, 16)
            written.append(p)
            for p in written:
                d = audio_io.decode_audio(p)
                assert abs(len(d) - len(mix)) < 44100 * 0.2, f"{p.name}: {len(d)}"
            return {"files": len(written)}
        check("export", export)

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
