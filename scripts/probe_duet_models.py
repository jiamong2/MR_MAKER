"""듀엣(남/여) 분리 후보 모델 점검: 다운로드 → 설정 읽기 → 우리 BS-RoFormer 구현으로 불러오기 → 짧게 실행.

CI(.github/workflows/probe-duet.yml)에서 실행하며 결과를 ::notice:: 주석으로 남긴다.
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_separator import roformer  # noqa: E402

HF = "https://huggingface.co"
CANDIDATES = {
    "male_female_aufr33": (
        f"{HF}/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
        f"{HF}/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.yaml"),
    "chorus_sucial_ep267": (
        f"{HF}/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
        f"{HF}/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/model_chorus_bs_roformer.yaml"),
}


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
    for repo in ("Sucial/Chorus_Male_Female_BS_RoFormer",):
        try:
            list_repo(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"::error::list {repo}: {exc}", flush=True)
    sr = 44100
    t = np.arange(sr * 8) / sr
    # 낮은 목소리(약 130Hz)와 높은 목소리(약 330Hz)를 흉내 낸 배음 신호 (실행 확인용, 품질 평가 아님)
    low = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 12)) * 0.08
    high = sum(np.sin(2 * np.pi * 330 * k * t) / k for k in range(1, 8)) * 0.08
    mix = np.stack([low + high, low + high], axis=1).astype(np.float32)
    for name, (ckpt_url, yaml_url) in CANDIDATES.items():
        try:
            y = fetch(yaml_url, work / f"{name}.yaml")
            note(f"{name} yaml: " + y.read_text(encoding="utf-8").replace("\n", " | ")[:1500])
            c = fetch(ckpt_url, work / f"{name}.ckpt")
            sha = hashlib.sha256(c.read_bytes()).hexdigest()
            note(f"{name} ckpt size={c.stat().st_size} sha256={sha}")
            cfg = roformer.read_config(y)
            note(f"{name} parsed: instruments={cfg['instruments']} all={cfg['all_instruments']} chunk={cfg['chunk']} "
                 f"overlap={cfg['num_overlap']} model_keys={sorted(cfg['model'])}")
            model = roformer.load_checkpoint(c, cfg)
            t0 = time.time()
            out = roformer.demix(model, mix, "cpu", num_overlap=cfg["num_overlap"], chunk=cfg["chunk"])
            e = [float(np.sqrt(np.mean(o ** 2))) for o in out]
            note(f"{name} LOADED OK: stems={model.num_stems} out={out.shape} rms={['%.4f' % v for v in e]} "
                 f"input_rms={float(np.sqrt(np.mean(mix ** 2))):.4f} time={time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"::error::{name} FAILED: {type(exc).__name__}: {str(exc)[:900]}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "probe_work"))
