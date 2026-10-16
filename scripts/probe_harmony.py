"""화음 따기 모델 점검 (CI 전용, .github/workflows/probe-harmony.yml).

1. 카라오케(메인/화음) BS-RoFormer: 파일 크기·SHA-256, 우리 구현 vs 기준 구현(ZFTurbo) 같은 입력 비교, CPU 속도
2. RMVPE 음정 추적: 파일 크기·SHA-256, 우리 구현 vs 기준 구현(RVC) — 짧은 곡(한 번에) / 긴 곡(나눠서) 비교
3. 실제 듀엣 보컬 예제(Sucial 저장소: 섞인 보컬 + 남/여 정답 트랙)로 전체 흐름 확인
   정답 트랙을 따로 음정 추적한 결과를 기준으로, 섞인 보컬 → 메인/화음 분리 → 음정 추적 결과가 두 성부를 얼마나
   따라가는지 프레임 단위로 측정 (분리 없이 섞인 보컬을 바로 추적한 경우와 비교)

결과는 ::notice:: 주석으로 남긴다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import types
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_separator import audio_io, rmvpe, roformer, transcribe  # noqa: E402

HF = "https://huggingface.co"
KARAOKE = f"{HF}/becruily/bs-roformer-karaoke/resolve/main"
RMVPE_URL = f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
SUCIAL = f"{HF}/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/examples"


RESULTS: list = []


def note(msg: str):
    """결과 한 줄 (GitHub 는 단계마다 주석을 10개까지만 보여 주므로 마지막에 묶어서 냄)."""
    print(msg, flush=True)
    RESULTS.append(msg)


def flush_notes():
    buf = ""
    for line in RESULTS:
        if len(buf) + len(line) > 3500:
            print(f"::notice::{buf}", flush=True)
            buf = ""
        buf += (" ‖ " if buf else "") + line
    if buf:
        print(f"::notice::{buf}", flush=True)


def err(msg: str):
    print(f"::error::{msg}", flush=True)


class _Skip(Exception):
    pass


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


def repo_info(repo: str, files: list):
    try:
        with urllib.request.urlopen(f"{HF}/api/models/{repo}?blobs=true") as r:
            info = json.loads(r.read().decode())
        lic = (info.get("cardData") or {}).get("license")
        rows = []
        for s in info.get("siblings", []):
            if s["rfilename"] in files:
                lfs = s.get("lfs") or {}
                rows.append(f"{s['rfilename']} size={s.get('size')} lfs_sha256={lfs.get('sha256')}")
        note(f"{repo}: license={lic} tags={[t for t in info.get('tags', []) if 'license' in t]} | " + " | ".join(rows))
    except Exception as exc:  # noqa: BLE001
        err(f"{repo} info: {exc}")


def load_wav(p: Path, rate: int = 44100) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(p, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return audio_io.resample(x[:, :2], sr, rate)


def main(work: Path, mss_root: Path, rvc_root: Path):
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    work.mkdir(parents=True, exist_ok=True)
    note(f"cpu threads={torch.get_num_threads()} torch={torch.__version__}")
    repo_info("becruily/bs-roformer-karaoke", ["bs_roformer_karaoke_frazer_becruily.ckpt",
                                               "config_karaoke_frazer_becruily.yaml"])
    repo_info("lj1995/VoiceConversionWebUI", ["rmvpe.pt"])

    # ---------------------------------------------------------------- 예제 ---
    ex = {}
    for k in ("vocals_raw", "vocals_male", "vocals_female"):
        try:
            ex[k] = load_wav(fetch(f"{SUCIAL}/{k}.wav", work / f"{k}.wav"))
        except Exception as exc:  # noqa: BLE001
            err(f"example {k}: {exc}")
    if ex:
        note("examples: " + ", ".join(f"{k} {len(v) / 44100:.1f}s" for k, v in ex.items()))
    mix = ex.get("vocals_raw")

    # ------------------------------------------------- 1. 카라오케 모델 ---
    kmodel = None
    try:
        y = fetch(f"{KARAOKE}/config_karaoke_frazer_becruily.yaml", work / "karaoke.yaml")
        c = fetch(f"{KARAOKE}/bs_roformer_karaoke_frazer_becruily.ckpt", work / "karaoke.ckpt")
        print("---- karaoke config ----\n" + y.read_text(encoding="utf-8") + "\n------------------------", flush=True)
        note(f"karaoke ckpt bytes={c.stat().st_size} sha256={sha(c)}")
        cfg = roformer.read_config(y)
        note(f"karaoke parsed: instruments={cfg['instruments']} all={cfg['all_instruments']} chunk={cfg['chunk']} "
             f"overlap={cfg['num_overlap']} model={ {k: v for k, v in cfg['model'].items() if k != 'freqs_per_bands'} }")
        kmodel = roformer.load_checkpoint(c, cfg)
        note(f"karaoke LOADED OK (strict): stems={kmodel.num_stems} instruments={kmodel.instruments}")
        if os.environ.get("SKIP_KARAOKE_REF"):
            raise _Skip()
        sys.path.insert(0, str(mss_root))
        from models.bs_roformer.bs_roformer import BSRoformer as RefBS

        ref = RefBS(**dict(cfg["model"]))
        st = torch.load(str(c), map_location="cpu", weights_only=True)
        ref.load_state_dict(st.get("state_dict", st) if isinstance(st, dict) and "state_dict" in st else st)
        ref.eval()
        src = mix if mix is not None else np.random.default_rng(0).normal(0, 0.1, (cfg["chunk"], 2)).astype(np.float32)
        seg = np.zeros((cfg["chunk"], 2), np.float32)
        part = src[: cfg["chunk"]]
        seg[: len(part)] = part
        x = torch.from_numpy(np.ascontiguousarray(seg.T))[None]
        with torch.no_grad():
            t0 = time.time()
            a = ref(x).numpy()
            t_ref = time.time() - t0
            t0 = time.time()
            b = kmodel(x).numpy()
            t_ours = time.time() - t0
        a = a.reshape(b.shape)
        note(f"karaoke OURS vs REFERENCE (same {cfg['chunk'] / 44100:.1f}s chunk): {sdr(a, b):.1f} dB "
             f"(max abs diff {float(np.max(np.abs(a - b))):.2e}); chunk time ours={t_ours:.1f}s ref={t_ref:.1f}s; "
             f"→ CPU 1분 곡 약 {60 / (cfg['chunk'] / 44100) * 2 * t_ours:.0f}s (overlap 2) / "
             f"{60 / (cfg['chunk'] / 44100) * 4 * t_ours:.0f}s (overlap 4)")
    except _Skip:
        note("karaoke reference comparison skipped (done before: 43.0 dB)")
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        err(f"karaoke FAILED: {type(exc).__name__}: {str(exc)[:900]}")

    # ----------------------------------------------------- 2. RMVPE ---
    pm = None
    try:
        p = fetch(RMVPE_URL, work / "rmvpe.pt")
        note(f"rmvpe bytes={p.stat().st_size} sha256={sha(p)}")
        pm = rmvpe.load(p)
        note(f"rmvpe LOADED OK (strict): params={sum(t.numel() for t in pm.parameters()) / 1e6:.1f}M")
        import librosa

        lm = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
        note(f"mel filters vs librosa: max abs diff {float(np.max(np.abs(lm - rmvpe.mel_filters()))):.2e} "
             f"(max value {float(lm.max()):.3e})")
        stub = types.ModuleType("tools.cuda_graph")
        stub.run_cuda_graph = lambda model, name, fn, x: fn(x)
        sys.modules["tools"] = types.ModuleType("tools")
        sys.modules["tools.cuda_graph"] = stub
        spec = importlib.util.spec_from_file_location("rvc_rmvpe", rvc_root / "infer" / "rmvpe.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        refp = mod.RMVPE(str(p), is_half=False, device="cpu")
        base = ex.get("vocals_male") if ex else None
        if base is None:
            t = np.arange(16000 * 12) / 16000
            base16 = (0.3 * np.sin(2 * np.pi * 220 * t * (1 + 0.02 * np.sin(2 * np.pi * 0.5 * t)))).astype(np.float32)
        else:
            base16 = audio_io.resample(base.mean(1, keepdims=True), 44100, 16000)[:, 0]
        long16 = np.concatenate([base16] * max(1, int(np.ceil(16000 * 75 / len(base16)))))[: 16000 * 75]
        for label, a16 in (("short 10s (one pass)", base16[: 16000 * 10]), ("long 75s (chunked)", long16)):
            t0 = time.time()
            f_ref = refp.infer_from_audio(a16, thred=0.03)
            t_ref = time.time() - t0
            t0 = time.time()
            f_ours, _ = rmvpe.decode(rmvpe.salience_16k(pm, a16))
            t_ours = time.time() - t0
            n = min(len(f_ref), len(f_ours))
            both = (f_ref[:n] > 0) & (f_ours[:n] > 0)
            agree = float(np.mean((f_ref[:n] > 0) == (f_ours[:n] > 0)))
            cents = np.abs(1200 * np.log2(f_ours[:n][both] / f_ref[:n][both])) if both.any() else np.array([0.0])
            note(f"rmvpe OURS vs REFERENCE {label}: frames ref={len(f_ref)} ours={len(f_ours)} voicing agree "
                 f"{agree * 100:.2f}% · cents diff max {cents.max():.2f} / 99th {np.percentile(cents, 99):.2f} · "
                 f"time ours={t_ours:.1f}s ref={t_ref:.1f}s")
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        err(f"rmvpe FAILED: {type(exc).__name__}: {str(exc)[:900]}")

    # ------------------------------------------- 3. 실제 듀엣 예제 전체 흐름 ---
    if kmodel is None or pm is None or len(ex) < 3:
        return
    try:
        t0 = time.time()
        lead = roformer.demix(kmodel, mix, "cpu", num_overlap=2, chunk=kmodel.chunk)[0]
        t_split = time.time() - t0
        back = mix - lead
        e = lambda x: 10 * np.log10(np.mean(x ** 2) + 1e-12)  # noqa: E731
        note(f"split {len(mix) / 44100:.1f}s in {t_split:.1f}s (CPU, overlap 2): energy dB mix={e(mix):.1f} "
             f"lead={e(lead):.1f} back={e(back):.1f}")
        track = {}
        t0 = time.time()
        for k, x in (("lead", lead), ("back", back), ("male", ex["vocals_male"]), ("female", ex["vocals_female"]),
                     ("mix", mix)):
            f0, _ = rmvpe.track(pm, x, 44100)
            track[k] = (f0, transcribe.frame_db(x, 44100, len(f0)))
        note(f"rmvpe 5 tracks in {time.time() - t0:.1f}s")
        n = min(len(v[0]) for v in track.values())
        total = transcribe.frame_db(mix, 44100, n)
        gate = np.maximum(total - 18.0, -55.0)

        def midi(k, gated=True):
            f0, db = track[k]
            f0, db = f0[:n], db[:n]
            return transcribe.hz_to_midi(np.where((db >= gate) if gated else (db > -55), f0, 0.0))
        gm, gf = midi("male", False), midi("female", False)
        lm, bm, mm = midi("lead"), midi("back"), midi("mix", False)
        harm = np.isfinite(gm) & np.isfinite(gf) & (np.abs(gm - gf) >= 1.5)    # 두 성부가 함께 다른 음을 부르는 프레임
        solo = np.isfinite(gm) ^ np.isfinite(gf)

        def hit(x, ref):
            return np.isfinite(x) & np.isfinite(ref) & (np.abs(x - ref) < 0.5)
        if harm.any():
            lm_m, lm_f = hit(lm, gm)[harm].mean(), hit(lm, gf)[harm].mean()
            bm_m, bm_f = hit(bm, gm)[harm].mean(), hit(bm, gf)[harm].mean()
            both = ((hit(lm, gm) & hit(bm, gf)) | (hit(lm, gf) & hit(bm, gm)))[harm].mean()
            mix_any = (hit(mm, gm) | hit(mm, gf))[harm].mean()
            note(f"duet frames with harmony={int(harm.sum())} ({harm.mean() * 100:.0f}% of song), solo={int(solo.sum())}. "
                 f"lead=male {lm_m * 100:.0f}% / female {lm_f * 100:.0f}%; back=male {bm_m * 100:.0f}% / "
                 f"female {bm_f * 100:.0f}%; BOTH voices recovered {both * 100:.0f}% "
                 f"(no separation: one voice only, {mix_any * 100:.0f}%)")
        tuning = transcribe.estimate_tuning(lm, bm)
        res = {}
        for k in ("lead", "back", "male", "female"):
            f0, db = track[k]
            g = gate[: len(f0)] if k in ("lead", "back") else -55.0
            res[k] = transcribe.notes_from_f0(f0[:n], db[:n], g if np.ndim(g) == 0 else g[:n], tuning)
        rng = {k: (min(x.pitch for x in v), max(x.pitch for x in v)) if v else None for k, v in res.items()}
        note("notes: " + ", ".join(f"{k}={len(v)} range={rng[k]}" for k, v in res.items()) + f" tuning={tuning:+.2f}")
        for k in ("male", "female", "lead", "back"):
            note(f"{k} notes: " + " ".join(f"{x.start:.2f}/{x.end - x.start:.2f}/{x.pitch}" for x in res[k][:45]))
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        err(f"end-to-end FAILED: {type(exc).__name__}: {str(exc)[:900]}")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    finally:
        flush_notes()
