"""MR MAKER / MR SEPARATOR for ui.clover - AI 음원 분리 데스크톱 애플리케이션.

같은 코드로 두 가지 프로그램(에디션)을 만든다.
- maker     : MR MAKER for ui.clover — 보컬 담당자용. 2-Stem(보컬 / 반주), 원클릭 홈 화면
- separator : MR SEPARATOR for ui.clover — 악기를 다루는 사람용. 6-Stem(보컬·드럼·베이스·기타·피아노·나머지),
              트랙 화면이 곧 첫 화면

에디션은 빌드할 때 build_info.json 에 기록되고, 소스 실행 때는 MUSIC_SEPARATOR_EDITION 환경 변수로 고른다.
"""
import json as _json
import os as _os
import sys as _sys
from pathlib import Path as _Path

EDITIONS = ("maker", "separator")


def _detect_edition() -> str:
    e = _os.environ.get("MUSIC_SEPARATOR_EDITION", "").strip().lower()
    if e in EDITIONS:
        return e
    if getattr(_sys, "frozen", False):
        try:
            base = _Path(getattr(_sys, "_MEIPASS", _Path(_sys.executable).parent))
            e = str(_json.loads((base / "build_info.json").read_text(encoding="utf-8")).get("edition", ""))
            if e in EDITIONS:
                return e
        except Exception:  # noqa: BLE001
            pass
    return "maker"


EDITION = _detect_edition()
SEPARATOR = EDITION == "separator"

APP_BRAND = "for ui.clover"
if SEPARATOR:
    APP_NAME = "MR SEPARATOR for ui.clover"
    APP_SHORT = "MR SEPARATOR"
    EXE_NAME = "MRSeparator"
    BUNDLE_ID = "io.github.musicseparator.separator"
    SETTINGS_ID = "MRSeparator"            # 설정(QSettings)은 MR MAKER 와 따로 저장
    LOG_NAME = "mr_separator.log"
    ICON_STEM = "icon_separator"           # assets/icon_separator.png · .ico · .icns
else:
    APP_NAME = "MR MAKER for ui.clover"   # 화면·설치 프로그램에 표시되는 이름
    APP_SHORT = "MR MAKER"                 # 좁은 곳(macOS 메뉴 막대 등)에 쓰는 이름
    EXE_NAME = "MRMaker"                   # 실행 파일 / 설치 파일 이름 (공백 없이)
    BUNDLE_ID = "io.github.musicseparator.app"   # macOS 번들 ID (이전 버전 설정 유지 위해 그대로)
    SETTINGS_ID = "MusicSeparator"
    LOG_NAME = "music_separator.log"
    ICON_STEM = "icon"
# 내부 식별자 — 이전 버전(Music Separator)의 설정·다운로드한 AI 모델을 그대로 쓰도록 유지.
# 두 에디션이 같은 데이터 폴더를 쓰므로 내려받은 AI 모델(BS-RoFormer SW 667MB 등)을 함께 쓴다.
APP_ID = "MusicSeparator"
ORG_NAME = "MusicSeparator"
DATA_DIR_NAME = "Music Separator"      # macOS ~/Library/Application Support/<이름>
# 두 프로그램은 버전을 따로 매기고 따로 릴리스한다.
#   MR MAKER     : 태그 v<버전>            (GitHub '최신 릴리스' = MR MAKER)
#   MR SEPARATOR : 태그 separator-v<버전>  ('최신 릴리스'로 표시하지 않음)
MAKER_VERSION = "1.4.2"
SEPARATOR_VERSION = "1.0.1"
__version__ = SEPARATOR_VERSION if SEPARATOR else MAKER_VERSION
RELEASE_TAG_PREFIX = "separator-v" if SEPARATOR else "v"
