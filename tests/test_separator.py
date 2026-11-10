"""MR SEPARATOR (6-Stem 에디션) — 에디션 구분, 업데이트 파일 선택, 골라 듣기·믹스 저장."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _py(code: str, edition: str) -> str:
    env = dict(os.environ, MUSIC_SEPARATOR_EDITION=edition, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8",
                       timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


def test_edition_constants():
    code = ("import json, music_separator as m; print(json.dumps([m.EDITION, m.APP_NAME, m.APP_SHORT, m.EXE_NAME, "
            "m.BUNDLE_ID, m.SETTINGS_ID, m.APP_ID, m.DATA_DIR_NAME, m.LOG_NAME]))")
    mk = json.loads(_py(code, "maker"))
    sp = json.loads(_py(code, "separator"))
    assert mk[:6] == ["maker", "MR MAKER for ui.clover", "MR MAKER", "MRMaker", "io.github.musicseparator.app",
                      "MusicSeparator"]                       # 기존 프로그램은 그대로 (설정·업데이트 유지)
    assert sp[:4] == ["separator", "MR SEPARATOR for ui.clover", "MR SEPARATOR", "MRSeparator"]
    assert sp[4] != mk[4] and sp[5] != mk[5] and sp[8] != mk[8]      # 번들 ID·설정·로그는 따로
    assert sp[6:8] == mk[6:8]                                 # 데이터 폴더(내려받은 AI 모델)는 함께 씀
    assert json.loads(_py(code, "nonsense"))[0] == "maker"    # 모르는 값이면 기본 프로그램


def test_edition_defaults():
    code = ("import json; from music_separator import engine, pipeline; "
            "print(json.dumps([engine.app_default_preset(), pipeline.JobSpec('x', '.').mode]))")
    assert json.loads(_py(code, "separator"))[0] == "quality"       # 피아노·기타 품질이 목적 → 항상 품질 우선
    assert json.loads(_py(code, "maker"))[1] == "2"


def test_updater_picks_own_edition_files():
    code = '''
import json
from music_separator import updater
def rel(names): return {"tag_name": "v99.0.0", "html_url": "u", "assets": [
    {"name": n, "browser_download_url": "https://x/" + n, "size": 1} for n in names]}
both = ["MRMaker-99.0.0-windows-x64-cpu-setup.exe", "MRMaker-99.0.0-windows-x64-cpu.msi",
        "MRSeparator-99.0.0-windows-x64-cpu-setup.exe", "MRSeparator-99.0.0-windows-x64-cpu.msi",
        "MRSeparator-99.0.0-macos-arm64.dmg", "MRMaker-99.0.0-macos-arm64.dmg"]
a = updater.parse_release(rel(both), "cpu", auto=True)
m = updater.parse_release(rel(both), "cpu", auto=True, msi=True)
only_maker = updater.parse_release(rel([n for n in both if n.startswith("MRMaker")]), "cpu", auto=True)
only_sep = updater.parse_release(rel([n for n in both if n.startswith("MRSeparator")]), "cpu", auto=True)
print(json.dumps([[x.name for x in a.assets], [x.name for x in m.assets], only_maker is not None,
                  only_sep is not None, updater.MSI_DIR_NAME]))
'''
    mk = json.loads(_py(code, "maker"))
    sp = json.loads(_py(code, "separator"))
    assert mk[0] == ["MRMaker-99.0.0-windows-x64-cpu-setup.exe"] and mk[1] == ["MRMaker-99.0.0-windows-x64-cpu.msi"]
    assert sp[0] == ["MRSeparator-99.0.0-windows-x64-cpu-setup.exe"]
    assert sp[1] == ["MRSeparator-99.0.0-windows-x64-cpu.msi"]
    assert mk[2:4] == [True, False] and sp[2:4] == [False, True]      # 다른 프로그램만 올라온 릴리스는 알리지 않음
    assert mk[4] == "MR MAKER (MSI)" and sp[4] == "MR SEPARATOR (MSI)"


def test_separation_key_keeps_modes_apart():
    from music_separator import pipeline

    two = pipeline.separation_key("file:a", "bs_roformer_sw", "quality")
    assert two == ("file:a", "bs_roformer_sw", "quality") == pipeline.separation_key("file:a", "bs_roformer_sw",
                                                                                      "quality", "2")
    assert pipeline.separation_key("file:a", "bs_roformer_sw", "quality", "6") != two   # 같은 모델이라도 섞이지 않게


def _ui(tmp_path, edition: str) -> str:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "separator_ui_check.py"), str(tmp_path / edition), edition],
                       env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), (r.stdout[-1500:], r.stderr[-2500:])
    return r.stdout


def test_separator_listen_and_mix_export(tmp_path):
    """칩 ↔ 음소거 동기화, 솔로·볼륨이 반영된 '지금 들리는 대로 저장', 저장된 소리 = 켜진 트랙의 합."""
    out = _ui(tmp_path, "separator")
    assert "믹스 (피아노 제외)" in out and "믹스 (피아노만)" in out and "믹스 (드럼+베이스, 볼륨 조절)" in out


def test_maker_edition_unchanged(tmp_path):
    """MR MAKER 는 그대로: 홈 화면으로 시작, 2-Stem 트랙, 믹스 카드 없음."""
    _ui(tmp_path, "maker")
