"""다크 테마 (Logic Pro 풍 회색 톤) 와 벡터 아이콘."""
from __future__ import annotations

import sys

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap

BG = "#1b1b1d"          # 창 배경
PANEL = "#232326"       # 인스펙터/패널
PANEL_2 = "#2b2b2f"     # 카드, 트랙 헤더
TRACK_BG = "#1f1f22"    # 트랙 영역
BORDER = "#3a3a3f"
BORDER_SOFT = "#2f2f34"
TEXT = "#e6e6e9"
TEXT_DIM = "#9b9ba3"
TEXT_FAINT = "#6c6c74"
ACCENT = "#3d8bfd"
ACCENT_HOVER = "#5a9dff"
WARN = "#f5a623"
DANGER = "#ff5c5c"
SUCCESS = "#4cd27a"
MUTE_ON = "#2f9bff"     # Logic: Mute = 파랑
SOLO_ON = "#f5c518"     # Logic: Solo = 노랑
LCD_BG = "#111214"


def ui_font_family() -> str:
    if sys.platform == "darwin":
        return ".AppleSystemUIFont"
    if sys.platform == "win32":
        return "Segoe UI"
    return "Noto Sans CJK KR"


def mono_font(size: int = 20, bold: bool = False) -> QFont:
    fam = "Menlo" if sys.platform == "darwin" else ("Consolas" if sys.platform == "win32" else "DejaVu Sans Mono")
    f = QFont(fam, size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setBold(bold)
    return f


def apply_palette(app) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANEL_2))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(PANEL_2))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL_2))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_FAINT))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(TEXT_FAINT))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TEXT_FAINT))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(TEXT_FAINT))
    app.setPalette(pal)
    from ..paths import assets_dir

    check = (assets_dir() / "check.png").as_posix()
    app.setStyleSheet(STYLESHEET.replace("@CHECK@", check))


STYLESHEET = f"""
* {{ font-size: 13px; }}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}

/* 상단 컨트롤 바 */
QWidget#ControlBar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2d2d31, stop:1 #252528);
                     border-bottom: 1px solid #111; }}
QLabel#AppTitle {{ font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#FileTitle {{ color: {TEXT_DIM}; }}
QFrame#Lcd {{ background: {LCD_BG}; border: 1px solid #000; border-radius: 6px; }}
QLabel#LcdTime {{ color: #f2f2f2; }}
QLabel#LcdSub {{ color: {TEXT_FAINT}; font-size: 10px; letter-spacing: 1px; }}

QPushButton {{ background: #36363b; color: {TEXT}; border: 1px solid #45454b; border-radius: 6px;
              padding: 6px 12px; }}
QPushButton:hover {{ background: #404046; }}
QPushButton:pressed {{ background: #2c2c30; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: #2a2a2e; border-color: #333338; }}
QPushButton#Primary {{ background: {ACCENT}; border: 1px solid #2f74e0; color: white; font-weight: 600;
                      padding: 9px 14px; font-size: 14px; }}
QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#Primary:disabled {{ background: #2b3a52; color: #8090a8; border-color: #2b3a52; }}
QPushButton#Danger {{ background: #4a2a2c; border-color: #6b3436; color: #ffb3b3; }}
QPushButton#Danger:hover {{ background: #5a3033; }}
QPushButton#Transport {{ background: #3a3a3f; border: 1px solid #4a4a50; border-radius: 6px; padding: 0; }}
QPushButton#Transport:hover {{ background: #46464c; }}
QPushButton#Transport:checked {{ background: #2d5d3a; border-color: #3c8a52; }}
QPushButton#Flat {{ background: transparent; border: none; color: {TEXT_DIM}; padding: 4px 6px; }}
QPushButton#Flat:hover {{ color: {TEXT}; }}

/* 세그먼트(모드 선택) */
QPushButton#Segment {{ background: #2e2e33; border: 1px solid #424248; border-radius: 0; padding: 8px 0;
                      font-weight: 600; color: {TEXT_DIM}; }}
QPushButton#Segment:checked {{ background: {ACCENT}; color: white; border-color: #2f74e0; }}
QPushButton#SegmentLeft {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QPushButton#Segment[pos="left"] {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QPushButton#Segment[pos="right"] {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; }}

/* 칩(MR 제외 악기) */
QPushButton#Chip {{ background: #2e2e33; border: 1px solid #45454b; border-radius: 13px; padding: 4px 11px;
                   color: {TEXT}; }}
QPushButton#Chip:checked {{ background: #4a2a2c; border-color: #a54848; color: #ffc4c4;
                           text-decoration: line-through; }}

/* 인스펙터(왼쪽 패널) */
QScrollArea#InspectorScroll, QWidget#Inspector {{ background: {PANEL}; border: none; }}
QFrame#Card {{ background: {PANEL_2}; border: 1px solid {BORDER_SOFT}; border-radius: 9px; }}
QLabel#SectionTitle {{ color: {TEXT_DIM}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#Hint {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel#Note {{ color: {WARN}; font-size: 12px; }}
QLabel#Notice {{ background: #3a3020; border: 1px solid #6b5426; border-radius: 6px; color: #ffd591;
                padding: 6px 8px; font-size: 12px; }}

QComboBox {{ background: #2e2e33; border: 1px solid #45454b; border-radius: 6px; padding: 5px 8px; color: {TEXT}; }}
QComboBox:hover {{ border-color: #55555c; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {BORDER}; selection-background-color: {ACCENT}; }}
QCheckBox {{ color: {TEXT}; spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid #5a5a62; background: #2a2a2e; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("@CHECK@"); }}
QCheckBox:disabled {{ color: {TEXT_FAINT}; }}

QProgressBar {{ background: #18181a; border: 1px solid #3a3a3f; border-radius: 5px; height: 10px; text-align: center;
               color: transparent; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2f74e0, stop:1 #5aa8ff);
                      border-radius: 4px; }}

QSlider::groove:horizontal {{ height: 4px; background: #151517; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #8a8a92; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
                             background: #e8e8ea; border: 1px solid #777; }}
QSlider#Seek::sub-page:horizontal {{ background: {ACCENT}; }}

/* 트랙 */
QWidget#TrackArea {{ background: {TRACK_BG}; }}
QFrame#ExportBar {{ background: {PANEL}; border-top: 1px solid #111; }}
QFrame#ExportBar QPushButton#Primary {{ padding: 6px 14px; font-size: 13px; }}
QFrame#TrackHeader {{ background: {PANEL_2}; border-right: 1px solid #111; border-bottom: 1px solid #18181a; }}
QLabel#TrackName {{ font-weight: 600; font-size: 13px; }}
QLabel#TrackSub {{ color: {TEXT_FAINT}; font-size: 11px; }}
QPushButton#MuteBtn, QPushButton#SoloBtn {{ background: #3b3b40; border: 1px solid #4b4b51; border-radius: 4px;
                                           padding: 0; font-weight: 700; font-size: 11px; color: #cfcfd4; }}
QPushButton#MuteBtn:checked {{ background: {MUTE_ON}; color: white; border-color: #1d7ad6; }}
QPushButton#SoloBtn:checked {{ background: {SOLO_ON}; color: #1a1a1a; border-color: #c9a10f; }}
QLabel#Gain {{ color: {TEXT_DIM}; font-size: 11px; }}

QStatusBar {{ background: #19191b; color: {TEXT_DIM}; border-top: 1px solid #111; }}
QStatusBar QLabel {{ color: {TEXT_DIM}; }}
QMenuBar {{ background: #232326; color: {TEXT}; }}
QMenuBar::item:selected {{ background: #3a3a3f; }}
QMenu {{ background: {PANEL_2}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #45454b; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QMessageBox {{ background: {PANEL}; }}
"""


def make_icon(kind: str, color: str = TEXT, size: int = 64) -> QIcon:
    """play/pause/stop/rewind/folder 등의 간단한 벡터 아이콘."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    s = size
    if kind == "play":
        path = QPainterPath()
        path.moveTo(QPointF(s * 0.30, s * 0.20))
        path.lineTo(QPointF(s * 0.80, s * 0.50))
        path.lineTo(QPointF(s * 0.30, s * 0.80))
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "pause":
        p.drawRoundedRect(QRectF(s * 0.27, s * 0.20, s * 0.16, s * 0.60), 2, 2)
        p.drawRoundedRect(QRectF(s * 0.57, s * 0.20, s * 0.16, s * 0.60), 2, 2)
    elif kind == "stop":
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.26, s * 0.48, s * 0.48), 3, 3)
    elif kind == "rewind":
        p.drawRect(QRectF(s * 0.20, s * 0.24, s * 0.09, s * 0.52))
        path = QPainterPath()
        path.moveTo(QPointF(s * 0.78, s * 0.24))
        path.lineTo(QPointF(s * 0.33, s * 0.50))
        path.lineTo(QPointF(s * 0.78, s * 0.76))
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "speaker":
        path = QPainterPath()
        path.moveTo(QPointF(s * 0.15, s * 0.38))
        path.lineTo(QPointF(s * 0.32, s * 0.38))
        path.lineTo(QPointF(s * 0.52, s * 0.20))
        path.lineTo(QPointF(s * 0.52, s * 0.80))
        path.lineTo(QPointF(s * 0.32, s * 0.62))
        path.lineTo(QPointF(s * 0.15, s * 0.62))
        path.closeSubpath()
        p.drawPath(path)
        pen = QPen(c, s * 0.06)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(s * 0.40, s * 0.30, s * 0.30, s * 0.40), -55 * 16, 110 * 16)
        p.drawArc(QRectF(s * 0.38, s * 0.18, s * 0.46, s * 0.64), -55 * 16, 110 * 16)
    elif kind == "folder":
        path = QPainterPath()
        path.moveTo(QPointF(s * 0.12, s * 0.28))
        path.lineTo(QPointF(s * 0.40, s * 0.28))
        path.lineTo(QPointF(s * 0.48, s * 0.36))
        path.lineTo(QPointF(s * 0.88, s * 0.36))
        path.lineTo(QPointF(s * 0.88, s * 0.78))
        path.lineTo(QPointF(s * 0.12, s * 0.78))
        path.closeSubpath()
        p.drawPath(path)
    p.end()
    return QIcon(pm)
