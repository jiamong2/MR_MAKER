"""화음 따기(주 선율 + 화음) 점검 (CI 전용, .github/workflows/probe-harmony.yml).

1. RMVPE 음정 추적: 우리 구현 vs 기준 구현(RVC) — 짧은 곡(한 번에) / 긴 곡(나눠서)
2. 정답 트랙이 있는 남/여 예제(Sucial)로 만든 시험곡에서 v1.3.5 방식(위/아래 성부)과 새 방식(주 선율/화음) 비교
   - 듀엣: 원래 예제 (두 사람이 계속 같이 부름)
   - 번갈아: 남자 솔로 13초 → 여자 솔로 13초 → 함께 26초 (City of Stars 같은 구성)
   - 번갈아 + 잔향: 위 곡에 긴 잔향(RT60 1.5초)을 섞음 (잔향이 가짜 두 번째 목소리를 만드는지)
   지표: 솔로 구간에서 솔로 목소리가 첫 줄에 있는 비율 / 둘째 줄로 새는 비율 / 가짜 화음 비율,
        함께 부르는 구간에서 두 목소리를 동시에 맞힌 비율, 음표 단위 일치율(F, 시작 ±0.1초·같은 음)
3. 새 방식 조건별 영향 (배음 에너지 기준, 최소 길이, 잔향 판정, 주 선율 유지)
4. 확인용 오디오: 솔로 구간의 화음 파일 음량, 함께 부르는 구간의 SDR

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

    try:
        evaluate(lambda a: rmvpe.decode(rmvpe.salience_16k(pm, a)), ex)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        err(f"evaluation FAILED: {type(exc).__name__}: {str(exc)[:900]}")


# --------------------------------------------------- v1.3.5 방식 (비교용) ---

def assign_voices_v135(a, b, fps=100, link=0.05, hold=0.3):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(len(a), len(b))
    up, lo = np.full(n, np.nan), np.full(n, np.nan)
    last = {"up": (-10 ** 9, np.nan), "lo": (-10 ** 9, np.nan)}
    link_f, hold_f = max(1, int(round(link * fps))), int(round(hold * fps))
    for t in range(n):
        x, y = a[t], b[t]
        fx, fy = np.isfinite(x), np.isfinite(y)
        if fx and fy:
            hi, low = (x, y) if x >= y else (y, x)
            up[t], lo[t] = hi, low
            last["up"], last["lo"] = (t, hi), (t, low)
            continue
        if not (fx or fy):
            continue
        p = x if fx else y
        cont = {k: t - last[k][0] <= link_f and abs(p - last[k][1]) < 1.0 for k in ("up", "lo")}
        if cont["up"] != cont["lo"]:
            line = "up" if cont["up"] else "lo"
        elif cont["up"] or (t - last["up"][0] <= hold_f and t - last["lo"][0] <= hold_f):
            line = "up" if abs(p - last["up"][1]) <= abs(p - last["lo"][1]) else "lo"
        else:
            line = "up"
        (up if line == "up" else lo)[t] = p
        last[line] = (t, p)
    return up, lo


def old_method(c, a16):
    n = len(c["ma"])
    dbt = transcribe.frame_db(a16, 16000, n)
    dbr = transcribe.frame_db(c["residual"], 16000, n)
    ma = c["ma"]
    mb = np.where(dbr >= np.maximum(dbt - 18.0, -55.0), c["mb"], np.nan)
    mb[np.isfinite(ma) & np.isfinite(mb) & (np.abs(ma - mb) < 1.0)] = np.nan
    return assign_voices_v135(ma, mb)


def new_method(c, **over):
    saved = {k: getattr(harmony, k) for k in over if k.isupper()}
    for k, v in saved.items():
        setattr(harmony, k, over[k])
    try:
        mh = harmony.validate_harmony(c["ma"], c["mb"], c["harm"], c["total"])
        if over.get("keep", True):
            return harmony.keep_melody_line(c["ma"], mh)
        return np.asarray(c["ma"], float), mh
    finally:
        for k, v in saved.items():
            setattr(harmony, k, v)


# ------------------------------------------------------------------ 평가 ---

def reverb(x: np.ndarray, sr: int = 44100, rt60: float = 1.5, wet: float = 0.35, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    out = np.array(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        ir = rng.normal(0, 1, n) * np.exp(-6.9 * t / rt60)
        ir[: int(0.01 * sr)] = 0                         # 10ms 프리딜레이
        ir /= np.sqrt(np.sum(ir ** 2))
        m = 1 << int(np.ceil(np.log2(len(x) + n)))
        y = np.fft.irfft(np.fft.rfft(x[:, ch], m) * np.fft.rfft(ir, m), m)[: len(x)]
        out[:, ch] += wet * y
    return out.astype(np.float32)


def note_f(est, ref, tol=0.1):
    used, hit = set(), 0
    for r in ref:
        for i, e in enumerate(est):
            if i not in used and e.pitch == r.pitch and abs(e.start - r.start) <= tol:
                used.add(i)
                hit += 1
                break
    p = hit / max(1, len(est))
    r = hit / max(1, len(ref))
    return 2 * p * r / max(1e-9, p + r)


def semis(ns):
    st = [abs(ns[i + 1].pitch - ns[i].pitch) for i in range(len(ns) - 1) if abs(ns[i + 1].start - ns[i].end) < 0.035]
    return float(np.mean(np.array(st) == 1)) if st else 0.0


def metrics(l1, l2, gm, gf):
    n = min(len(l1), len(l2), len(gm), len(gf))
    l1, l2, gm, gf = l1[:n], l2[:n], gm[:n], gf[:n]
    vm, vf = np.isfinite(gm), np.isfinite(gf)
    solo = vm ^ vf
    singer = np.where(vm, gm, gf)
    harm = vm & vf & (np.abs(np.nan_to_num(gm) - np.nan_to_num(gf)) >= 1.5)
    r = {}
    if solo.any():
        r["solo_in_line1"] = hit(l1, singer)[solo].mean()
        r["solo_in_line2"] = hit(l2, singer)[solo].mean()
        r["solo_false_line2"] = (np.isfinite(l2) & ~hit(l2, singer))[solo].mean()
    if harm.any():
        r["duet_both"] = ((hit(l1, gm) & hit(l2, gf)) | (hit(l1, gf) & hit(l2, gm)))[harm].mean()
        r["duet_line2_present"] = np.isfinite(l2)[harm].mean()
        v2 = np.isfinite(l2) & (vm | vf)
        r["line2_precision"] = (hit(l2, gm) | hit(l2, gf))[v2].mean() if v2.any() else 0.0
    return r


def fmt(r):
    return " ".join(f"{k}={v * 100:.0f}%" for k, v in r.items())


def evaluate(fn, ex):
    sr = 44100
    raw, gm_a, gf_a = ex["vocals_raw"], ex["vocals_male"], ex["vocals_female"]
    h = len(raw) // 2
    z = np.zeros_like(raw[:h])
    alt_m = np.concatenate([gm_a[:h], z[: len(raw) - h], gm_a])
    alt_f = np.concatenate([z, gf_a[h:], gf_a])
    alt_mix = np.concatenate([gm_a[:h], gf_a[h:], raw])
    songs = {"duet": (raw, gm_a, gf_a), "alternating": (alt_mix, alt_m, alt_f),
             "alternating+reverb": (reverb(alt_mix), alt_m, alt_f)}

    def gt(x, n):
        f0, _ = fn(audio_io.resample(x.mean(1, keepdims=True), sr, 16000)[:, 0])
        return transcribe.hz_to_midi(np.where(transcribe.frame_db(x, sr, len(f0)) > -55, f0, 0))[:n]
    for name, (mix, xm, xf) in songs.items():
        t0 = time.time()
        a16 = audio_io.resample(mix.mean(1, keepdims=True), sr, 16000)[:, 0]
        c = harmony.voice_candidates(a16, fn, fn)
        n = len(c["ma"])
        gm, gf = gt(xm, n), gt(xf, n)
        t_c = time.time() - t0
        up, lo = old_method(c, a16)
        mel, har = new_method(c)
        note(f"[{name}] {len(mix) / sr:.0f}s, candidates {t_c:.1f}s")
        note(f"[{name}] OLD 위/아래: {fmt(metrics(up, lo, gm, gf))}")
        note(f"[{name}] NEW 주선율/화음: {fmt(metrics(mel, har, gm, gf))}")
        # 음표 단위 (솔로 목소리 = 첫 줄 기대). 정답 음표는 정답 트랙을 같은 방식으로 정리한 것
        solo = np.isfinite(gm) ^ np.isfinite(gf)
        if solo.any():
            singer = np.where(np.isfinite(gm), gm, gf)
            ref = transcribe.notes_from_pitch(np.where(solo, singer, np.nan), None, -200, 0.0)
            seg = lambda line: transcribe.notes_from_pitch(np.where(solo, line, np.nan), None, -200, 0.0)  # noqa
            o1, n1 = seg(up), seg(mel)
            note(f"[{name}] solo notes: ref {len(ref)} | OLD line1 {len(o1)} F={note_f(o1, ref):.2f} | "
                 f"NEW melody {len(n1)} F={note_f(n1, ref):.2f} | semitone steps ref {semis(ref) * 100:.0f}% "
                 f"OLD {semis(o1) * 100:.0f}% NEW {semis(n1) * 100:.0f}%")
        # 조건별 영향 (기본값 대비) + 조합 격자
        rows = []
        for label, over in (("no harm gate", {"HARM_REL_DB": 200.0}), ("no min", {"HARM_MIN": 0.0}),
                            ("no echo rule", {"ECHO_START": 0.0}), ("no melody keep", {"keep": False})):
            r = metrics(*new_method(c, **over), gm, gf)
            rows.append(f"{label}: " + " ".join(f"{k.split('_', 1)[1]}={v * 100:.0f}" for k, v in r.items()
                                               if k in ("solo_false_line2", "duet_both", "line2_precision")))
        note(f"[{name}] ablation (default {harmony.HARM_REL_DB:g}dB/{harmony.HARM_MIN:g}s/echo/keep): " + " | ".join(rows))
        grid = []
        for rel in (12.0, 15.0, 18.0, 25.0):
            for mn in (0.1, 0.15, 0.25):
                for es in (0.0, 0.12):
                    r = metrics(*new_method(c, HARM_REL_DB=rel, HARM_MIN=mn, ECHO_START=es), gm, gf)
                    grid.append(f"{rel:g}/{mn:g}/{'E' if es else '-'}:"
                                f"{r.get('solo_false_line2', 0) * 100:.0f},{r.get('duet_both', 0) * 100:.0f},"
                                f"{r.get('line2_precision', 0) * 100:.0f}")
        note(f"[{name}] grid rel/min/echo: false,both,prec  " + " ".join(grid))
        # 확인용 오디오
        a_mel, a_har = harmony.split_audio(mix, harmony.midi_to_hz(mel), harmony.midi_to_hz(har), sr)
        dv = transcribe.frame_db(mix, sr, n)
        dh = transcribe.frame_db(a_har, sr, n)
        msg = []
        if solo.any():
            msg.append(f"solo: harmony audio {np.median((dh - dv)[solo]):.1f} dB vs vocals")
        both = np.isfinite(gm) & np.isfinite(gf)
        if both.any():
            idx = np.repeat(both, sr // 100)[: len(mix)]
            idx = np.pad(idx, (0, len(mix) - len(idx)))
            mm, ff = xm[idx], xf[idx]
            hh, me = a_har[idx], a_mel[idx]
            best = max(sdr(mm, me) + sdr(ff, hh), sdr(ff, me) + sdr(mm, hh)) / 2
            base = (sdr(mm, mix[idx]) + sdr(ff, mix[idx])) / 2
            msg.append(f"together: mean SDR {best:.1f} dB (mix as-is {base:.1f} dB)")
        note(f"[{name}] preview audio: " + "; ".join(msg))


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
    finally:
        flush_notes()
