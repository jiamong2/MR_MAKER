"""테스트용: 실제 Demucs 와 동일한 구조(무작위 가중치)의 모델 파일 생성.

네트워크 없이 모델 로드 경로(HuggingFace safetensors 형식 / 기존 .th 형식)와
실제 크기 모델의 연산을 검증하기 위해 사용합니다. 분리 품질은 의미 없습니다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from demucs.htdemucs import HTDemucs
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_separator.models import MODELS  # noqa: E402


def make(root: Path, name: str, fmt: str = "safetensors") -> None:
    spec = MODELS[name]
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.yaml").write_text(spec.yaml_text, encoding="utf-8")
    for sig, legacy in zip(spec.sigs, spec.legacy_files):
        torch.manual_seed(0)
        kwargs = {"sources": list(spec.sources)}
        model = HTDemucs(**kwargs)
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        if fmt == "safetensors":
            meta = {
                "klass": "demucs.htdemucs.HTDemucs",
                "args": json.dumps([]),
                "kwargs": json.dumps({**kwargs, "segment": {"_type": "fraction", "numerator": 39,
                                                            "denominator": 5}}),
            }
            save_file(state, str(folder / f"{sig}.safetensors"), metadata=meta)
        else:
            from fractions import Fraction

            pkg = {"klass": HTDemucs, "args": (), "kwargs": {**kwargs, "segment": Fraction(39, 5)},
                   "state": state}
            torch.save(pkg, folder / f"{sig}.th")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    fmt = sys.argv[2] if len(sys.argv) > 2 else "safetensors"
    for n in sys.argv[3:] or ["htdemucs", "htdemucs_6s"]:
        make(out, n, fmt)
        print("created", n, fmt)
