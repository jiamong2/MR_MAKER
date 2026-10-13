"""디자인 시스템: 밝은 화이트 테마 색상, 폰트(Pretendard), 스타일시트."""
from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

log = logging.getLogger(__name__)

# ---- 색상 (은은한 흰색 계열) -------------------------------------------------
BG = "#f3f5f9"            # 창 배경 (아주 옅은 회청색)
SURFACE = "#ffffff"       # 카드/패널
SURFACE_2 = "#f7f9fc"     # 카드 안쪽 박스
SURFACE_3 = "#f1f4f8"     # 버튼, 입력칸
TRACK_PANEL = "#ffffff"   # 트랙/타임라인 패널
BORDER = "#e4e8ef"
BORDER_2 = "#d9dee7"
TEXT = "#1d2433"
TEXT_DIM = "#687083"
TEXT_FAINT = "#9aa1b0"
ACCENT = "#3478f6"
ACCENT_2 = "#2f6ee8"
ACCENT_SOFT = "#eaf2ff"
DANGER = "#e5484d"
DANGER_SOFT = "#fff1f2"
SUCCESS = "#22a55b"
WARN = "#d9822b"
MUTE_ON = "#3478f6"
SOLO_ON = "#f5b400"
TOGGLE_OFF = "#d6dce6"
SHADOW = "#1d2433"

FONT_FAMILY = "Pretendard"


def load_fonts() -> str:
    """번들 폰트(Pretendard) 등록. 실패하면 OS 기본 한글 폰트 사용."""
    global FONT_FAMILY
    from ..paths import assets_dir

    families = set()
    for f in sorted((assets_dir() / "fonts").glob("*.otf")):
        fid = QFontDatabase.addApplicationFont(str(f))
        if fid >= 0:
            families.update(QFontDatabase.applicationFontFamilies(fid))
    if "Pretendard" in families:
        FONT_FAMILY = "Pretendard"
    else:
        FONT_FAMILY = {"darwin": "Apple SD Gothic Neo", "win32": "Malgun Gothic"}.get(sys.platform,
                                                                                   "Noto Sans CJK KR")
        log.warning("Pretendard 폰트를 불러오지 못해 %s 사용", FONT_FAMILY)
    return FONT_FAMILY


def font(px: int = 13, weight: QFont.Weight = QFont.Weight.Normal, tabular: bool = False) -> QFont:
    f = QFont(FONT_FAMILY)
    f.setPixelSize(px)
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    if tabular:
        try:
            f.setFeature(QFont.Tag("tnum"), 1)
        except Exception:
            pass
    return f


def rgba(color: str, alpha: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha:.3f})"


def tint(color: str, amount: float) -> QColor:
    """color 를 흰색과 섞은 파스텔 색 (amount: 0 = 흰색, 1 = 원래 색)."""
    c = QColor(color)
    return QColor(int(255 + (c.red() - 255) * amount), int(255 + (c.green() - 255) * amount),
                  int(255 + (c.blue() - 255) * amount))


def shade(color: str, amount: float) -> QColor:
    """color 를 어둡게 (amount: 0 = 그대로, 1 = 검정)."""
    c = QColor(color)
    k = 1 - amount
    return QColor(int(c.red() * k), int(c.green() * k), int(c.blue() * k))


def apply(app) -> None:
    load_fonts()
    app.setStyle("Fusion")
    app.setFont(font(13))
    try:  # 창 제목 표시줄도 밝게 (Qt 6.8+)
        from PySide6.QtCore import Qt

        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except Exception:
        pass
    pal = QPalette()
    for role, col in [
        (QPalette.ColorRole.Window, BG), (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Base, SURFACE), (QPalette.ColorRole.AlternateBase, SURFACE_2),
        (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, SURFACE),
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, SURFACE),
        (QPalette.ColorRole.ToolTipText, TEXT), (QPalette.ColorRole.PlaceholderText, TEXT_FAINT),
        (QPalette.ColorRole.Light, "#ffffff"), (QPalette.ColorRole.Mid, BORDER_2),
    ]:
        pal.setColor(role, QColor(col))
    for role in (QPalette.ColorRole.ButtonText, QPalette.ColorRole.Text, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_FAINT))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET.replace("@FONT@", FONT_FAMILY))


STYLESHEET = f"""
* {{ font-family: "@FONT@"; }}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER_2}; padding: 6px 8px;
           border-radius: 6px; font-size: 12px; }}
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel[role="dim"] {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel[role="faint"] {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel[role="caps"] {{ color: {TEXT_FAINT}; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}
QLabel[role="cardTitle"] {{ font-size: 17px; font-weight: 700; }}
QLabel[role="sectionLabel"] {{ font-size: 14px; font-weight: 600; }}
QLabel[role="warn"] {{ color: {WARN}; font-size: 12px; }}

/* 헤더 */
QWidget#Header {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QLabel#AppTitle {{ font-size: 22px; font-weight: 800; color: {TEXT}; }}
QLabel#AppBrand {{ font-size: 13px; font-weight: 600; color: #3f7a55; }}
QLabel#Tagline {{ color: {TEXT_DIM}; font-size: 13px; }}
QFrame#Lcd {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 12px; }}

/* 버튼 */
QPushButton {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER_2}; border-radius: 10px;
              padding: 8px 14px; font-size: 13px; font-weight: 500; }}
QPushButton:hover {{ background: {SURFACE_2}; border-color: #c9d0dc; }}
QPushButton:pressed {{ background: {SURFACE_3}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: {SURFACE_2}; border-color: {BORDER}; }}
QPushButton[kind="primary"] {{ color: white; font-weight: 600; border: 1px solid #2f6ee8;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4b8bff, stop:1 #2f6ee8); }}
QPushButton[kind="primary"]:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5a96ff, stop:1 #3a78f2); }}
QPushButton[kind="primary"]:pressed {{ background: #2a63d6; }}
QPushButton[kind="primary"]:disabled {{ background: #c9d9f7; border-color: #c9d9f7; color: #ffffff; }}
QPushButton[kind="danger"] {{ background: {DANGER_SOFT}; border-color: #f3b4b8; color: {DANGER}; }}
QPushButton[kind="danger"]:hover {{ background: #ffe5e7; }}
QPushButton[kind="square"] {{ padding: 0; border-radius: 12px; }}
QPushButton[kind="ghost"] {{ background: transparent; border: none; padding: 4px; border-radius: 8px; }}
QPushButton[kind="ghost"]:hover {{ background: {SURFACE_3}; }}
QPushButton[kind="step"] {{ padding: 0; border-radius: 10px; font-size: 18px; font-weight: 600; }}

/* 카드 */
QFrame#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; }}
QFrame#InfoBox {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#NumBadge {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 9px;
                  font-size: 16px; font-weight: 700; color: {TEXT}; qproperty-alignment: AlignCenter; }}
QScrollArea#LeftScroll, QWidget#LeftPanel {{ background: transparent; border: none; }}
QLabel#KeyValue {{ font-size: 26px; font-weight: 800; color: {TEXT}; }}

QComboBox {{ background: {SURFACE}; border: 1px solid {BORDER_2}; border-radius: 10px; padding: 8px 12px;
            color: {TEXT}; font-size: 13px; min-height: 22px; }}
QComboBox:hover {{ border-color: #c3cad6; }}
QComboBox:disabled {{ color: {TEXT_FAINT}; background: {SURFACE_2}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: url("@CHEVRON@"); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORDER_2}; border-radius: 8px;
                              selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
                              outline: 0; padding: 4px; }}
QCheckBox {{ color: {TEXT}; spacing: 9px; font-size: 13px; }}
QCheckBox:disabled {{ color: {TEXT_FAINT}; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid #c3cad6; background: #ffffff; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("@CHECK@"); }}

QSlider::groove:horizontal {{ height: 4px; background: #e3e7ee; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 10px; margin: -5px 0; border-radius: 7px; background: #ffffff;
                             border: 2px solid {ACCENT}; }}
QSlider::groove:horizontal:disabled {{ background: #eceff4; }}
QSlider::sub-page:horizontal:disabled {{ background: #c9d0dc; }}
QSlider::handle:horizontal:disabled {{ border-color: #c9d0dc; }}
QSlider#Seek::handle:horizontal {{ border: 2px solid {ACCENT}; width: 12px; margin: -6px 0; border-radius: 8px; }}

QProgressBar {{ background: #edf1f6; border: none; border-radius: 4px; height: 8px; color: transparent; }}
QProgressBar::chunk {{ border-radius: 4px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3478f6, stop:1 #6aa6ff); }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #d3d9e3; border-radius: 3px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #d3d9e3; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* 트랙 */
QFrame#TrackCard {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#TrackNum {{ color: {TEXT_FAINT}; font-size: 13px; font-weight: 600; }}
QLabel#TrackName {{ font-size: 15px; font-weight: 700; }}
QLabel#TrackSub {{ color: {TEXT_DIM}; font-size: 12px; }}
QPushButton#MuteBtn, QPushButton#SoloBtn {{ background: {SURFACE_3}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 0; font-size: 12px; font-weight: 600; color: #5b6475; }}
QPushButton#MuteBtn:hover, QPushButton#SoloBtn:hover {{ background: #e8ecf2; }}
QPushButton#MuteBtn:checked {{ background: {MUTE_ON}; border-color: {MUTE_ON}; color: white; }}
QPushButton#SoloBtn:checked {{ background: {SOLO_ON}; border-color: {SOLO_ON}; color: #3a2a00; }}
QLabel#Gain {{ color: {TEXT_DIM}; font-size: 12px; }}

/* 하단 바 */
QFrame#BottomBar {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
QLabel#StatusOk {{ color: {SUCCESS}; font-weight: 700; font-size: 13px; }}
QLabel#StatusBusy {{ color: {ACCENT}; font-weight: 700; font-size: 13px; }}
QLabel#StatusErr {{ color: {DANGER}; font-weight: 700; font-size: 13px; }}

QMenu {{ background: {SURFACE}; border: 1px solid {BORDER_2}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 18px; border-radius: 6px; color: {TEXT}; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}
QMenuBar {{ background: {SURFACE}; color: {TEXT}; }}
QMessageBox, QDialog {{ background: {SURFACE}; }}
QLineEdit {{ background: #ffffff; border: 1px solid {BORDER_2}; border-radius: 8px; padding: 7px 10px; color: {TEXT}; }}
"""


def finalize_stylesheet(app) -> None:
    """에셋 경로가 필요한 부분(체크, 화살표 이미지)을 채움."""
    from ..paths import assets_dir

    a = assets_dir()
    ss = app.styleSheet().replace("@CHECK@", (a / "check.png").as_posix())
    ss = ss.replace("@CHEVRON@", (a / "chevron.png").as_posix())
    app.setStyleSheet(ss)
