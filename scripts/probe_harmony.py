"""화음 따기(두 성부 음정 추적) 점검 (CI 전용, .github/workflows/probe-harmony.yml).

1. RMVPE 음정 추적: 파일 크기·SHA-256, 우리 구현 vs 기준 구현(RVC) — 짧은 곡(한 번에) / 긴 곡(나눠서) 비교
2. 실제 듀엣 보컬 예제(Sucial 저장소: 섞인 보컬 + 남/여 정답 트랙)로 앱과 같은 코드(harmony.two_voices →
   transcribe.assign_voices)를 돌려, 정답 트랙을 따로 추적한 결과와 프레임 단위로 비교
3. 확인용 오디오(split_audio): 위/아래 성부 소리가 정답 남/여 트랙과 얼마나 닮았는지 (SDR)

결과는 ::notice:: 주석으로 남긴다 (단계마다 주석 수 제한이 있어 마지막에 묶어서).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import types
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_separator import audio_io, harmony, rmvpe, transcribe  # noqa: E402

HF = "https://huggingface.co"
RMVPE_URL = f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
SUCIAL = f"{HF}/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/examples"
RESULTS: list = []


def note(msg: str):
    print(msg, flush=True)
    RESULTS.append(msg)


def flush_notes():
    buf = ""
    for line in RESULTS:
        if buf and len(buf) + len(line) > 3500:
            print(f"::notice::{buf}", flush=True)
            buf = ""
        buf += (" ‖ " if buf else "") + line
    if buf:
        print(f"::notice::{buf}", flush=True)


def err(msg: str):
    print(f"::error::{msg}", flush=True)


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sdr(ref, est) -> float:
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-9) / (np.sum((ref - est) ** 2) + 1e-9)))


def load_wav(p: Path, rate: int = 44100) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(p, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return audio_io.resample(x[:, :2], sr, rate)


def hit(x, r):
    return np.isfinite(x) & np.isfinite(r) & (np.abs(x - r) < 0.5)


def score(name, p1, p2, gm, gf):
    """두 정답 성부(gm, gf)와 예측 두 줄(p1, p2) 비교 (프레임 단위, ±0.5반음)."""
    n = min(len(p1), len(p2), len(gm), len(gf))
    p1, p2, gm, gf = p1[:n], p2[:n], gm[:n], gf[:n]
    harm = np.isfinite(gm) & np.isfinite(gf) & (np.abs(gm - gf) >= 1.5)
    both = ((hit(p1, gm) & hit(p2, gf)) | (hit(p1, gf) & hit(p2, gm)))[harm].mean()
    rec_m = (hit(p1, gm) | hit(p2, gm))[np.isfinite(gm)].mean()
    rec_f = (hit(p1, gf) | hit(p2, gf))[np.isfinite(gf)].mean()
    vals = np.concatenate([p1, p2])
    ok = np.isfinite(vals)
    prec = (hit(vals, np.concatenate([gm, gm])) | hit(vals, np.concatenate([gf, gf])))[ok].mean() if ok.any() else 0
    note(f"[{name}] BOTH {both * 100:.0f}% (of {int(harm.sum())} harmony frames) · recall male {rec_m * 100:.0f}% "
         f"female {rec_f * 100:.0f}% · precision {prec * 100:.0f}%")


def main(work: Path, rvc_root: Path):
    import torch

    work.mkdir(parents=True, exist_ok=True)
    note(f"cpu threads={torch.get_num_threads()} torch={torch.__version__}")
    try:
        with urllib.request.urlopen(f"{HF}/api/models/lj1995/VoiceConversionWebUI?blobs=true") as r:
            info = json.loads(r.read().decode())
        s = next(x for x in info.get("siblings", []) if x["rfilename"] == "rmvpe.pt")
        note(f"rmvpe repo license={(info.get('cardData') or {}).get('license')} size={s.get('size')} "
             f"sha256={(s.get('lfs') or {}).get('sha256')}")
    except Exception as exc:  # noqa: BLE001
        err(f"repo info: {exc}")

    ex = {}
    for k in ("vocals_raw", "vocals_male", "vocals_female"):
        try:
            ex[k] = load_wav(fetch(f"{SUCIAL}/{k}.wav", work / f"{k}.wav"))
        except Exception as exc:  # noqa: BLE001
            err(f"example {k}: {exc}")

    # ----------------------------------------------------- 1. RMVPE ---
    pm = None
    try:
        p = fetch(RMVPE_URL, work / "rmvpe.pt")
        note(f"rmvpe bytes={p.stat().st_size} sha256={sha(p)}")
        pm = rmvpe.load(p)
        import librosa

        lm = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
        note(f"mel filters vs librosa: max abs diff {float(np.max(np.abs(lm - rmvpe.mel_filters()))):.2e}")
        stub = types.ModuleType("tools.cuda_graph")
        stub.run_cuda_graph = lambda model, name, fn, x: fn(x)
        sys.modules["tools"] = types.ModuleType("tools")
        sys.modules["tools.cuda_graph"] = stub
        spec = importlib.util.spec_from_file_location("rvc_rmvpe", rvc_root / "infer" / "rmvpe.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        refp = mod.RMVPE(str(p), is_half=False, device="cpu")
        base16 = audio_io.resample(ex["vocals_male"].mean(1, keepdims=True), 44100, 16000)[:, 0]
        long16 = np.concatenate([base16] * int(np.ceil(16000 * 75 / len(base16))))[: 16000 * 75]
        for label, a16 in (("short 10s (one pass)", base16[: 16000 * 10]), ("long 75s (chunked)", long16)):
            f_ref = refp.infer_from_audio(a16, thred=0.03)
            f_ours, _ = rmvpe.decode(rmvpe.salience_16k(pm, a16))
            n = min(len(f_ref), len(f_ours))
            both = (f_ref[:n] > 0) & (f_ours[:n] > 0)
            agree = float(np.mean((f_ref[:n] > 0) == (f_ours[:n] > 0)))
            cents = np.abs(1200 * np.log2(f_ours[:n][both] / f_ref[:n][both])) if both.any() else np.array([0.0])
            note(f"rmvpe OURS vs REFERENCE {label}: frames {len(f_ref)}/{len(f_ours)} voicing agree "
                 f"{agree * 100:.2f}% · cents diff 99th {np.percentile(cents, 99):.2f} max {cents.max():.1f}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        err(f"rmvpe FAILED: {type(exc).__name__}: {str(exc)[:900]}")
    if pm is None or len(ex) < 3:
        return

    # ------------------------------------- 2·3. 실제 듀엣 예제 (앱과 같은 코드) ---
    try:
        mix, gt_m, gt_f = ex["vocals_raw"], ex["vocals_male"], ex["vocals_female"]
        t0 = time.time()
        ma, mb = harmony.two_voices(pm, mix, 44100)
        up, lo = transcribe.assign_voices(ma, mb)
        t_track = time.time() - t0
        t0 = time.time()
        a_up, a_lo = harmony.split_audio(mix, harmony.midi_to_hz(up), harmony.midi_to_hz(lo), 44100)
        t_split = time.time() - t0
        n = len(up)
        gm = transcribe.hz_to_midi(np.where(transcribe.frame_db(gt_m, 44100, n) > -55,
                                            rmvpe.track(pm, gt_m, 44100)[0][:n], 0))
        gf = transcribe.hz_to_midi(np.where(transcribe.frame_db(gt_f, 44100, n) > -55,
                                            rmvpe.track(pm, gt_f, 44100)[0][:n], 0))
        note(f"{len(mix) / 44100:.1f}s duet: two_voices {t_track:.1f}s, split_audio {t_split:.1f}s (CPU)")
        score("APP voices a/b", ma, mb, gm, gf)
        score("APP upper/lower", up, lo, gm, gf)
        k = min(len(gm), n)
        harm = np.isfinite(gm[:k]) & np.isfinite(gf[:k])
        note(f"upper=female {hit(up[:k], gf[:k])[harm].mean() * 100:.0f}% · lower=male "
             f"{hit(lo[:k], gm[:k])[harm].mean() * 100:.0f}% (frames with both voices)")
        m = min(len(a_up), len(gt_f))
        note(f"preview SDR: upper vs female {sdr(gt_f[:m], a_up[:m]):.1f} dB (mix vs female "
             f"{sdr(gt_f[:m], mix[:m]):.1f}), lower vs male {sdr(gt_m[:m], a_lo[:m]):.1f} dB (mix vs male "
             f"{sdr(gt_m[:m], mix[:m]):.1f})")
        tuning = transcribe.estimate_tuning(up, lo)
        res = {"female": transcribe.notes_from_pitch(gf, None, -200, tuning),
               "upper": transcribe.notes_from_pitch(up, transcribe.frame_db(a_up, 44100, n), -200, tuning),
               "male": transcribe.notes_from_pitch(gm, None, -200, tuning),
               "lower": transcribe.notes_from_pitch(lo, transcribe.frame_db(a_lo, 44100, n), -200, tuning)}
        note("notes: " + ", ".join(f"{kk}={len(v)}" for kk, v in res.items()) + f" tuning={tuning:+.2f}")
        for kk, v in res.items():
            note(f"{kk}: " + " ".join(f"{x.start:.2f}/{x.end - x.start:.2f}/{x.pitch}" for x in v[:30]))
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        err(f"end-to-end FAILED: {type(exc).__name__}: {str(exc)[:900]}")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
    finally:
        flush_notes()
