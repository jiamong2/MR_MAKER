"""디자인 시스템: 네이비 다크 테마 색상, 폰트(Pretendard), 스타일시트."""
from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

log = logging.getLogger(__name__)

# ---- 색상 -------------------------------------------------------------------
BG = "#0b1020"            # 창 배경
SURFACE = "#11182a"       # 카드/패널
SURFACE_2 = "#161e33"     # 카드 안쪽 박스
SURFACE_3 = "#1b2439"     # 버튼, 입력칸
TRACK_PANEL = "#0f1526"   # 트랙/타임라인 패널
BORDER = "#222c44"
BORDER_2 = "#2c3752"
TEXT = "#eef1f8"
TEXT_DIM = "#8f99b2"
TEXT_FAINT = "#5e6883"
ACCENT = "#3d7bff"
ACCENT_2 = "#5b93ff"
ACCENT_DEEP = "#2552d9"
DANGER = "#ef5a62"
SUCCESS = "#34c77b"
WARN = "#f5b543"
MUTE_ON = "#3d7bff"
SOLO_ON = "#f5b400"

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


def apply(app) -> None:
    load_fonts()
    app.setStyle("Fusion")
    app.setFont(font(13))
    try:  # 창 제목 표시줄도 다크로 (Qt 6.8+)
        from PySide6.QtCore import Qt

        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except Exception:
        pass
    pal = QPalette()
    for role, col in [
        (QPalette.ColorRole.Window, BG), (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Base, SURFACE_3), (QPalette.ColorRole.AlternateBase, SURFACE),
        (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, SURFACE_3),
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, SURFACE_2),
        (QPalette.ColorRole.ToolTipText, TEXT), (QPalette.ColorRole.PlaceholderText, TEXT_FAINT),
    ]:
        pal.setColor(role, QColor(col))
    for role in (QPalette.ColorRole.ButtonText, QPalette.ColorRole.Text, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_FAINT))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET.replace("@FONT@", FONT_FAMILY))


STYLESHEET = f"""
* {{ font-family: "@FONT@"; }}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER_2}; padding: 6px 8px;
           border-radius: 6px; font-size: 12px; }}
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel[role="dim"] {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel[role="faint"] {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel[role="caps"] {{ color: {TEXT_DIM}; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}
QLabel[role="cardTitle"] {{ font-size: 17px; font-weight: 700; }}
QLabel[role="sectionLabel"] {{ font-size: 14px; font-weight: 600; }}

/* 헤더 */
QWidget#Header {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #10172b, stop:1 #0c1222);
                 border-bottom: 1px solid {BORDER}; }}
QLabel#AppTitle {{ font-size: 22px; font-weight: 700; }}
QLabel#AiBadge {{ color: {ACCENT_2}; border: 1px solid {rgba(ACCENT_2, .7)}; border-radius: 5px;
                 padding: 0px 5px; font-size: 11px; font-weight: 700; }}
QLabel#Tagline {{ color: {TEXT_DIM}; font-size: 13px; }}
QFrame#Lcd {{ background: #0a0e1a; border: 1px solid {BORDER_2}; border-radius: 12px; }}

/* 버튼 */
QPushButton {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER_2}; border-radius: 10px;
              padding: 8px 14px; font-size: 13px; font-weight: 500; }}
QPushButton:hover {{ background: #222c45; border-color: #36425f; }}
QPushButton:pressed {{ background: #161e33; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: #141a2b; border-color: #1d2539; }}
QPushButton[kind="primary"] {{ color: white; font-weight: 600; border: 1px solid #4a86ff;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4a88ff, stop:1 #2f66f0); }}
QPushButton[kind="primary"]:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5b95ff, stop:1 #3a72f7); }}
QPushButton[kind="primary"]:pressed {{ background: #2a5ad6; }}
QPushButton[kind="primary"]:disabled {{ background: #1c2a4d; border-color: #24345c; color: #6b7ca3; }}
QPushButton[kind="danger"] {{ background: {rgba(DANGER, .12)}; border-color: {rgba(DANGER, .55)}; color: #ffc3c6; }}
QPushButton[kind="danger"]:hover {{ background: {rgba(DANGER, .2)}; }}
QPushButton[kind="square"] {{ padding: 0; border-radius: 12px; }}
QPushButton[kind="ghost"] {{ background: transparent; border: none; padding: 4px; border-radius: 8px; }}
QPushButton[kind="ghost"]:hover {{ background: {SURFACE_3}; }}

/* 카드 */
QFrame#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; }}
QFrame#InfoBox {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#NumBadge {{ background: {SURFACE_3}; border: 1px solid {BORDER_2}; border-radius: 9px;
                  font-size: 16px; font-weight: 700; qproperty-alignment: AlignCenter; }}
QScrollArea#LeftScroll, QWidget#LeftPanel {{ background: transparent; border: none; }}

QComboBox {{ background: #0e1426; border: 1px solid {BORDER_2}; border-radius: 10px; padding: 8px 12px;
            color: {TEXT}; font-size: 13px; min-height: 22px; }}
QComboBox:hover {{ border-color: #3a4766; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: url("@CHEVRON@"); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER_2}; border-radius: 8px;
                              selection-background-color: {ACCENT}; outline: 0; padding: 4px; }}
QCheckBox {{ color: {TEXT}; spacing: 9px; font-size: 13px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid #45526f; background: #0e1426; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("@CHECK@"); }}

QSlider::groove:horizontal {{ height: 4px; background: #27304a; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 10px; margin: -5px 0; border-radius: 7px; background: #f4f6fb;
                             border: 2px solid {ACCENT}; }}
QSlider#Seek::handle:horizontal {{ border: none; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }}
QSlider#Seek::sub-page:horizontal {{ background: {ACCENT}; }}

QProgressBar {{ background: #0c1222; border: 1px solid {BORDER}; border-radius: 4px; height: 8px;
               color: transparent; }}
QProgressBar::chunk {{ border-radius: 3px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2f66f0, stop:1 #6aa0ff); }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2a3450; border-radius: 3px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2a3450; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* 트랙 */
QFrame#TrackCard {{ background: #131a2c; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#TrackNum {{ color: {TEXT_FAINT}; font-size: 13px; font-weight: 600; }}
QLabel#TrackName {{ font-size: 15px; font-weight: 700; }}
QLabel#TrackSub {{ color: {TEXT_DIM}; font-size: 12px; }}
QPushButton#MuteBtn, QPushButton#SoloBtn {{ background: #1b2338; border: 1px solid {BORDER_2}; border-radius: 6px;
    padding: 0; font-size: 12px; font-weight: 600; color: #c6cde0; }}
QPushButton#MuteBtn:checked {{ background: {MUTE_ON}; border-color: {MUTE_ON}; color: white; }}
QPushButton#SoloBtn:checked {{ background: {SOLO_ON}; border-color: {SOLO_ON}; color: #1a1400; }}
QLabel#Gain {{ color: {TEXT_DIM}; font-size: 12px; }}

/* 하단 바 */
QFrame#BottomBar {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
QLabel#StatusOk {{ color: {SUCCESS}; font-weight: 700; font-size: 13px; }}
QLabel#StatusBusy {{ color: {ACCENT_2}; font-weight: 700; font-size: 13px; }}
QLabel#StatusErr {{ color: {DANGER}; font-weight: 700; font-size: 13px; }}

QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER_2}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 18px; border-radius: 6px; color: {TEXT}; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}
QMenuBar {{ background: {BG}; color: {TEXT}; }}
QMessageBox, QDialog {{ background: {SURFACE}; }}
QLineEdit {{ background: #0e1426; border: 1px solid {BORDER_2}; border-radius: 8px; padding: 7px 10px; color: {TEXT}; }}
"""


def finalize_stylesheet(app) -> None:
    """에셋 경로가 필요한 부분(체크, 화살표 이미지)을 채움."""
    from ..paths import assets_dir

    a = assets_dir()
    ss = app.styleSheet().replace("@CHECK@", (a / "check.png").as_posix())
    ss = ss.replace("@CHEVRON@", (a / "chevron.png").as_posix())
    app.setStyleSheet(ss)
