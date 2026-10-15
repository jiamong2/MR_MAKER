"""듀엣(남/여) 분리 후보 모델 점검: 다운로드 → 설정 읽기 → 우리 BS-RoFormer 구현으로 불러오기 → 짧게 실행.

CI(.github/workflows/probe-duet.yml)에서 실행하며 결과를 ::notice:: 주석으로 남긴다.
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_separator import roformer  # noqa: E402

HF = "https://huggingface.co"
SUCIAL = f"{HF}/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main"
CANDIDATES = {
    "chorus_sucial_ep267": (f"{SUCIAL}/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
                            f"{SUCIAL}/config_chorus_male_female_bs_roformer.yaml"),
}


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    num = np.sum(ref ** 2)
    den = np.sum((ref - est) ** 2)
    return float(10 * np.log10((num + 1e-9) / (den + 1e-9)))


def note(msg: str):
    print(f"::notice::{msg}", flush=True)


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def list_repo(repo: str) -> list:
    import json

    with urllib.request.urlopen(f"{HF}/api/models/{repo}") as r:
        info = json.loads(r.read().decode())
    files = [x["rfilename"] for x in info.get("siblings", [])]
    note(f"{repo} files: {files} license={info.get('cardData', {}).get('license')}")
    return files


def main(work: Path):
    work.mkdir(parents=True, exist_ok=True)
    import json

    for repo in ("DryPaintMan/MelBandRoformer-Duet", "oulianov/MelBandRoformer-Duet"):
        try:
            with urllib.request.urlopen(f"{HF}/api/models/{repo}") as r:
                info = json.loads(r.read().decode())
            card = ""
            try:
                with urllib.request.urlopen(f"{HF}/{repo}/raw/main/README.md") as r:
                    card = r.read().decode("utf-8", "replace").replace("\n", " | ")[:700]
            except Exception:  # noqa: BLE001
                pass
            note(f"{repo}: files={[x['rfilename'] for x in info.get('siblings', [])]} "
                 f"license={(info.get('cardData') or {}).get('license')} card={card}")
        except Exception as exc:  # noqa: BLE001
            print(f"::error::{repo}: {exc}", flush=True)
    # 예제 (원본 보컬 + 정답 남/여) 로 실제 분리 품질(SDR) 측정
    import soundfile as sf

    ex = {}
    for k in ("vocals_raw", "vocals_male", "vocals_female"):
        try:
            ex[k], ex_sr = sf.read(fetch(f"{SUCIAL}/examples/{k}.wav", work / f"{k}.wav"), dtype="float32",
                                   always_2d=True)
        except Exception as exc:  # noqa: BLE001
            print(f"::error::example {k}: {exc}", flush=True)
    if ex:
        note("examples: " + ", ".join(f"{k} {v.shape}" for k, v in ex.items()) + f" sr={ex_sr}")
    sr = 44100
    t = np.arange(sr * 8) / sr
    # 낮은 목소리(약 130Hz)와 높은 목소리(약 330Hz)를 흉내 낸 배음 신호 (실행 확인용, 품질 평가 아님)
    low = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 12)) * 0.08
    high = sum(np.sin(2 * np.pi * 330 * k * t) / k for k in range(1, 8)) * 0.08
    mix = np.stack([low + high, low + high], axis=1).astype(np.float32)
    for name, (ckpt_url, yaml_url) in CANDIDATES.items():
        try:
            y = fetch(yaml_url, work / f"{name}.yaml")
            c = fetch(ckpt_url, work / f"{name}.ckpt")
            sha = hashlib.sha256(c.read_bytes()).hexdigest()
            note(f"{name} ckpt size={c.stat().st_size} sha256={sha}")
            cfg = roformer.read_config(y)
            print(f"{name} parsed: instruments={cfg['instruments']} all={cfg['all_instruments']} chunk={cfg['chunk']} "
                 f"overlap={cfg['num_overlap']} model_keys={sorted(cfg['model'])}")
            model = roformer.load_checkpoint(c, cfg)
            t0 = time.time()
            out = roformer.demix(model, mix, "cpu", num_overlap=cfg["num_overlap"], chunk=cfg["chunk"])
            e = [float(np.sqrt(np.mean(o ** 2))) for o in out]
            note(f"{name} LOADED OK: stems={model.num_stems} out={out.shape} rms={['%.4f' % v for v in e]} "
                 f"input_rms={float(np.sqrt(np.mean(mix ** 2))):.4f} time={time.time() - t0:.1f}s")
            if len(ex) == 3:
                raw = ex["vocals_raw"]
                if raw.shape[1] == 1:
                    raw = np.repeat(raw, 2, axis=1)
                t0 = time.time()
                sep = roformer.demix(model, raw, "cpu", num_overlap=cfg["num_overlap"], chunk=cfg["chunk"])
                res = []
                for i, inst in enumerate(model.instruments):
                    ref_key = "vocals_male" if "male" == inst.lower() or inst.lower().startswith("male") else \
                        ("vocals_female" if "female" in inst.lower() else None)
                    if ref_key is None:
                        continue
                    ref = ex[ref_key]
                    if ref.shape[1] == 1:
                        ref = np.repeat(ref, 2, axis=1)
                    n = min(len(ref), len(sep[i]))
                    res.append(f"{inst}: SDR {sdr(ref[:n], sep[i][:n]):.2f}dB")
                # 기준선: 아무것도 안 하고 원본을 반씩 나눴을 때
                n = min(len(raw), len(ex["vocals_male"]))
                base = sdr(np.repeat(ex["vocals_male"], 2, 1)[:n] if ex["vocals_male"].shape[1] == 1
                           else ex["vocals_male"][:n], raw[:n] * 0.5)
                note(f"{name} EXAMPLE QUALITY: {', '.join(res)} (baseline half-mix {base:.2f}dB) "
                     f"time={time.time() - t0:.1f}s for {len(raw) / ex_sr:.1f}s audio")
        except Exception as exc:  # noqa: BLE001
            print(f"::error::{name} FAILED: {type(exc).__name__}: {str(exc)[:900]}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "probe_work"))
