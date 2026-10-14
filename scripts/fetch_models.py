"""AI 모델을 미리 내려받아 빌드에 포함(번들)하거나 오프라인 PC 로 옮길 때 사용합니다.

사용:
  python scripts/fetch_models.py build/models                       # htdemucs_ft_vocals (기본, 약 80MB)
  python scripts/fetch_models.py build/models bs_roformer_sw        # 최고 품질 모델(약 667MB)

만들어진 폴더를
  - 빌드에 포함:  MS_BUNDLE_MODELS=build/models pyinstaller packaging/music_separator.spec
  - 오프라인 PC:  프로그램의 [도움말] → [AI 모델 가져오기] 에서 이 폴더 선택
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1]).resolve()
    names = sys.argv[2:] or ["htdemucs_ft_vocals"]
    # 사용자 데이터 폴더를 target 의 부모로 돌려 models/<name> 구조로 저장
    staging = target.parent / (target.name + ".staging")
    os.environ["MUSIC_SEPARATOR_DATA_DIR"] = str(staging)
    os.environ.pop("MUSIC_SEPARATOR_MODELS_DIR", None)
    from music_separator import models

    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        if models._installed_in(target / name, models.MODELS[name]):
            print(f"{name}: already present")
            continue
        last = [0.0]

        def prog(done, total, label, name=name):
            if time.monotonic() - last[0] > 1.5:
                last[0] = time.monotonic()
                print(f"  {name}: {done / 1048576:.0f}/{total / 1048576:.0f} MB", flush=True)

        print(f"downloading {name} …", flush=True)
        folder = models.download_model(name, prog)
        dest = target / name
        dest.mkdir(exist_ok=True)
        for f in folder.iterdir():
            shutil.copy2(f, dest / f.name)
        print(f"{name}: ok -> {dest}")
    # 로드 검증
    os.environ["MUSIC_SEPARATOR_MODELS_DIR"] = str(target)
    models.clear_cache()
    for name in names:
        m = models.load_model(name)
        print(f"verified {name}: sources={list(getattr(m, 'sources', None) or getattr(m, 'instruments', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
