"""2-Stem(보컬 / 반주) 분리 품질 비교 — MUSDB18 7초 샘플 테스트 세트(50곡, 정답 트랙 포함).

사용:
  python scripts/evaluate_separation.py --data build/musdb7 --out eval.json
  python scripts/evaluate_separation.py --configs htdemucs,sw --tracks 20

측정 항목 (곡마다 계산 후 평균·중앙값)
- SDR 보컬 / SDR 반주(MR): 10·log10(Σs² / Σ(s−ŝ)²) — MVSEP·MDX 챌린지와 같은 방식 (스테레오 전체)
  반주는 앱과 똑같이 '원본 − 보컬 추정'으로 계산
- MR 보컬 잔류(dB): 1초 단위로 MR ≈ a·반주 + b·보컬 로 분해했을 때 b·보컬 / a·반주 에너지 비 (낮을수록 좋음)
- 보컬 내 반주 누출(dB): 보컬추정 ≈ c·보컬 + d·반주 로 분해했을 때 d·반주 / c·보컬 (낮을수록 좋음)
- 스테레오 폭 오차(dB): MR 과 정답 반주의 Side/Mid 에너지 비 차이 (0 에 가까울수록 원래 스테레오 이미지)
- 클리핑: 결과 최대 피크
- 처리 속도: 실시간 대비 배수 (이 컴퓨터 기준)

MUSDB18 샘플은 각 트랙이 AAC 256kbps 로 들어 있어 '원본'은 디코딩한 정답 트랙의 합으로 만듭니다.
주의: BS-RoFormer SW 는 학습 데이터가 공개되지 않아 MUSDB18 테스트 곡이 학습에 쓰였을 가능성을 배제할 수 없습니다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

URL = "https://github.com/sigsep/sigsep-mus-db/releases/download/v0.4.0/MUSDB18-7-STEMS.zip"
SR = 44100
GAP = SR  # 곡 사이 1초 무음

CONFIGS = {
    "htdemucs": dict(model="htdemucs", preset="speed", label="htdemucs (이전 버전 기본)"),
    "ft_speed": dict(model="htdemucs_ft_vocals", preset="speed", label="htdemucs_ft 보컬 · 속도 우선"),
    "ft_quality": dict(model="htdemucs_ft_vocals", preset="quality", label="htdemucs_ft 보컬 · shifts 2"),
    "sw": dict(model="bs_roformer_sw", preset="quality", label="BS-RoFormer SW · 품질 우선 (기본)"),
    "sw_best": dict(model="bs_roformer_sw", preset="best", label="BS-RoFormer SW · 최고 품질"),
}


def fetch(data: Path) -> Path:
    data.mkdir(parents=True, exist_ok=True)
    test = data / "test"
    if test.is_dir() and len(list(test.glob("*.mp4"))) >= 50:
        return test
    zp = data / "MUSDB18-7-STEMS.zip"
    if not zp.exists():
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(data)
    return test


def decode_stream(path: Path, idx: int) -> np.ndarray:
    from music_separator import audio_io

    cmd = [audio_io.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", f"0:a:{idx}",
           "-f", "f32le", "-ac", "2", "-ar", str(SR), "pipe:1"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(out, "<f4").reshape(-1, 2).astype(np.float32)


def load_tracks(test: Path, n: int) -> list:
    tracks = []
    for p in sorted(test.glob("*.stem.mp4"))[:n]:
        stems = [decode_stream(p, i) for i in (1, 2, 3, 4)]   # drums, bass, other, vocals
        m = min(len(s) for s in stems)
        drums, bass, other, vocals = (s[:m] for s in stems)
        inst = drums + bass + other
        tracks.append({"name": p.name.replace(".stem.mp4", ""), "vocals": vocals, "inst": inst,
                       "mix": vocals + inst})
    return tracks


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    num = np.sum(ref.astype(np.float64) ** 2)
    den = np.sum((ref.astype(np.float64) - est) ** 2)
    return float(10 * np.log10((num + 1e-8) / (den + 1e-8)))


def leak_db(est: np.ndarray, target: np.ndarray, other: np.ndarray, frame: int = SR) -> float:
    """est ≈ a·target + b·other 로 1초씩 분해 → Σ|b·other|² / Σ|a·target|² (dB)."""
    num = den = 0.0
    for s in range(0, len(est) - frame + 1, frame):
        e = est[s:s + frame].reshape(-1).astype(np.float64)
        t = target[s:s + frame].reshape(-1).astype(np.float64)
        o = other[s:s + frame].reshape(-1).astype(np.float64)
        if np.sum(t ** 2) < 1e-6 or np.sum(o ** 2) < 1e-6:
            continue
        a_mat = np.stack([t, o], 1)
        coef, *_ = np.linalg.lstsq(a_mat, e, rcond=None)
        num += np.sum((coef[1] * o) ** 2)
        den += np.sum((coef[0] * t) ** 2)
    if den <= 0:
        return float("nan")
    return float(10 * np.log10((num + 1e-12) / den))


def width_db(x: np.ndarray) -> float:
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2
    return float(10 * np.log10((np.sum(side.astype(np.float64) ** 2) + 1e-9) / (np.sum(mid.astype(np.float64) ** 2) + 1e-9)))


def run_config(key: str, tracks: list, device: str, work: Path) -> dict:
    from music_separator import engine

    cfg = CONFIGS[key]
    parts, spans = [], []
    pos = 0
    for t in tracks:
        parts += [t["mix"], np.zeros((GAP, 2), np.float32)]
        spans.append((pos, pos + len(t["mix"])))
        pos += len(t["mix"]) + GAP
    mix = np.concatenate(parts)
    t0 = time.monotonic()
    res = engine.separate(mix, "2", work / key, engine="auto", preset=cfg["preset"], device=device,
                          model_name=cfg["model"],
                          progress=lambda f, m: None)
    elapsed = time.monotonic() - t0
    voc_all = np.load(res.stems[0].path)
    rows = []
    for t, (a, b) in zip(tracks, spans):
        v_hat = voc_all[a:b]
        mr = t["mix"] - v_hat
        rows.append({
            "track": t["name"],
            "sdr_vocals": sdr(t["vocals"], v_hat),
            "sdr_inst": sdr(t["inst"], mr),
            "mr_vocal_residue_db": leak_db(mr, t["inst"], t["vocals"]),
            "vocal_bleed_db": leak_db(v_hat, t["vocals"], t["inst"]),
            "width_err_db": abs(width_db(mr) - width_db(t["inst"])),
            "peak": float(np.max(np.abs(mr))),
        })

    def agg(k, fn):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        return round(float(fn(vals)), 3) if vals else None
    audio_sec = sum(b - a for a, b in spans) / SR
    return {
        "config": key, "label": cfg["label"], "model": res.model, "preset": res.preset, "device": res.device,
        "tracks": len(rows),
        "sdr_vocals_mean": agg("sdr_vocals", np.mean), "sdr_vocals_median": agg("sdr_vocals", np.median),
        "sdr_inst_mean": agg("sdr_inst", np.mean), "sdr_inst_median": agg("sdr_inst", np.median),
        "mr_vocal_residue_db_mean": agg("mr_vocal_residue_db", np.mean),
        "vocal_bleed_db_mean": agg("vocal_bleed_db", np.mean),
        "width_err_db_mean": agg("width_err_db", np.mean),
        "max_peak": agg("peak", np.max),
        "realtime_factor": round(elapsed / audio_sec, 3), "elapsed_sec": round(elapsed, 1),
        "per_track": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "build" / "musdb7"))
    ap.add_argument("--configs", default="htdemucs,ft_speed,ft_quality,sw")
    ap.add_argument("--tracks", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="separation_eval.json")
    ap.add_argument("--annotate", action="store_true", help="GitHub Actions 주석으로 요약 출력")
    a = ap.parse_args()
    from music_separator import engine, models

    engine.configure_threads()
    test = fetch(Path(a.data))
    tracks = load_tracks(test, a.tracks)
    print(f"{len(tracks)} tracks, {sum(len(t['mix']) for t in tracks) / SR:.0f}s audio", flush=True)
    work = Path(a.out).resolve().parent / "eval_work"
    results = []
    for key in [k.strip() for k in a.configs.split(",") if k.strip()]:
        cfg = CONFIGS[key]
        if not models.is_installed(cfg["model"]):
            print(f"downloading model {cfg['model']}", flush=True)
            models.download_model(cfg["model"])
        print(f"== {key}: {cfg['label']}", flush=True)
        r = run_config(key, tracks, a.device, work)
        results.append(r)
        summary = {k: v for k, v in r.items() if k != "per_track"}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        if a.annotate:
            print(f"::notice title=separation {key}::{r['label']} | SDR 보컬 {r['sdr_vocals_mean']} "
                  f"(중앙 {r['sdr_vocals_median']}) | SDR 반주 {r['sdr_inst_mean']} (중앙 {r['sdr_inst_median']}) | "
                  f"MR 보컬 잔류 {r['mr_vocal_residue_db_mean']}dB | 보컬 내 반주 누출 {r['vocal_bleed_db_mean']}dB | "
                  f"스테레오 폭 오차 {r['width_err_db_mean']}dB | 최대 피크 {r['max_peak']} | "
                  f"실시간 대비 {r['realtime_factor']}배 ({r['device']})", flush=True)
    print("\n| 설정 | SDR 보컬 평균/중앙 | SDR 반주 평균/중앙 | MR 보컬 잔류 | 보컬 내 반주 누출 | 스테레오 폭 오차 | 속도 |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['label']} | {r['sdr_vocals_mean']} / {r['sdr_vocals_median']} | {r['sdr_inst_mean']} / "
              f"{r['sdr_inst_median']} | {r['mr_vocal_residue_db_mean']} dB | {r['vocal_bleed_db_mean']} dB | "
              f"{r['width_err_db_mean']} dB | ×{r['realtime_factor']} |")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.exit(main())
