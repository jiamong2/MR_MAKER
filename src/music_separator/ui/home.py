"""홈 화면 — 음원 파일 또는 YouTube URL 하나로 보컬/MR 을 원클릭 추출.

구성: 오디오 입력(파일 / YouTube) → 추출 대상(보컬만 / 반주(MR)만) → 키 조절 → [음원 추출 시작]
→ 진행 상황(단계별) → 결과(파일 열기·폴더 열기) / 오류(다시 시도).
고급 옵션 화면과 같은 색·글꼴·카드 스타일을 씁니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .. import APP_BRAND, APP_SHORT, audio_io, pitch, youtube
from . import icons, theme
from .widgets import LogoTile, glow, icon_button, label, repolish, text_button


def _shadow(w: QWidget, radius: int = 40, alpha: int = 18, dy: float = 6):
    glow(w, theme.SHADOW, radius, alpha, dy)


# ============================================================ 입력: 파일 ===

class HomeDropZone(QFrame):
    browseRequested = Signal()
    cleared = Signal()
    fileDropped = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(196)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._hover = False
        self.active = False
        self.dimmed = False
        lay = QVBoxLayout(self)
        self._lay = lay
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(8)
        lay.addStretch(1)
        self.dense = False
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.icon)
        self.title = QLabel("음원 파일을 끌어놓거나 선택하세요.")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setFont(theme.font(16, QFont.Weight.DemiBold))
        self.title.setWordWrap(True)
        lay.addWidget(self.title)
        self.sub = QLabel("MP3, WAV, M4A, FLAC 등 다양한 형식을 지원합니다.")
        self.sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:13px;")
        self.sub.setWordWrap(True)
        lay.addWidget(self.sub)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        self.browse_btn = text_button("파일 선택", "folder", "soft", 42, "#2f6f62")
        self.browse_btn.setMinimumWidth(150)
        self.browse_btn.clicked.connect(self.browseRequested)
        self.clear_btn = text_button("선택 해제", "x", None, 42, theme.TEXT_DIM)
        self.clear_btn.clicked.connect(self.cleared)
        self.clear_btn.hide()
        row.addWidget(self.browse_btn)
        row.addWidget(self.clear_btn)
        row.addStretch(1)
        lay.addSpacing(6)
        lay.addLayout(row)
        lay.addStretch(1)
        self.set_empty()

    def set_dense(self, level: int, compact: bool = False):
        """창 높이가 낮으면(level 1, 2) 여백·높이를 줄여 홈 화면 핵심 기능이 한 화면에 보이게 함."""
        self.dense = level > 0
        self.setMinimumHeight({0: 170 if compact else 190, 1: 146, 2: 112}[level])
        m = {0: 22, 1: 14, 2: 10}[level]
        self._lay.setContentsMargins(24, m, 24, m)
        self._lay.setSpacing(5 if level else 8)
        self.sub.setVisible(level < 2)
        if not self.active:
            self.set_empty()

    def set_empty(self):
        self.active = False
        self.icon.setPixmap(icons.pixmap("cloud-upload", theme.ACCENT, 38 if self.dense else 46, 2.0, 1.8))
        self.title.setText("음원 파일을 끌어놓거나 선택하세요.")
        self.sub.setText("MP3, WAV, M4A, FLAC 등 다양한 형식을 지원합니다.")
        self.browse_btn.setText("파일 선택")
        self.clear_btn.hide()
        self.update()

    def set_file(self, path: Path, detail: str = ""):
        self.active = True
        self.icon.setPixmap(icons.pixmap("file-music", theme.ACCENT, 34 if self.dense else 40, 2.0, 1.8))
        fm = self.title.fontMetrics()
        self.title.setText(fm.elidedText(Path(path).name, Qt.TextElideMode.ElideMiddle, max(200, self.width() - 80)))
        self.title.setToolTip(str(path))
        self.sub.setText(detail or "파일 정보를 확인하는 중…")
        self.browse_btn.setText("다른 파일")
        self.clear_btn.show()
        self.update()

    def set_detail(self, text: str):
        self.sub.setText(text)

    def set_dimmed(self, v: bool):
        self.dimmed = v
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        if self.active:
            p.setBrush(QColor("#f6fbf9"))
            p.setPen(QPen(QColor(theme.ACCENT), 2))
        else:
            p.setBrush(QColor(theme.ACCENT_SOFT if self._hover else ("#fbfcfa" if not self.dimmed else "#f7f8f5")))
            pen = QPen(QColor(theme.ACCENT if self._hover else "#b9d6cd"), 1.6, Qt.PenStyle.DashLine)
            pen.setDashPattern([5, 4])
            p.setPen(pen)
        p.drawRoundedRect(r, 16, 16)
        p.end()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover = True
            self.update()

    def dragLeaveEvent(self, e):
        self._hover = False
        self.update()

    def dropEvent(self, e):
        self._hover = False
        self.update()
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.fileDropped.emit(Path(urls[0].toLocalFile()))
        elif e.mimeData().hasText():
            self.fileDropped.emit(e.mimeData().text())

    def mouseReleaseEvent(self, e):
        if not self.active and e.button() == Qt.MouseButton.LeftButton:
            self.browseRequested.emit()


# ======================================================== 입력: YouTube ===

class YouTubeBadge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(56, 50)
        self.active = False

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(QColor("#ffffff"))
        p.setPen(QPen(QColor(theme.ACCENT if self.active else theme.BORDER), 1.4 if self.active else 1))
        p.drawRoundedRect(r, 12, 12)
        w, h = 30, 21
        logo = QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ff0033"))
        p.drawRoundedRect(logo, 6, 6)
        tri = QPainterPath()
        cx, cy = logo.center().x() + 1, logo.center().y()
        tri.moveTo(cx - 4.5, cy - 6)
        tri.lineTo(cx + 6.5, cy)
        tri.lineTo(cx - 4.5, cy + 6)
        tri.closeSubpath()
        p.setBrush(QColor("#ffffff"))
        p.drawPath(tri)
        p.end()


# ============================================================ 추출 대상 ===

class TargetCard(QAbstractButton):
    def __init__(self, icon_name: str, title: str, sub: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.icon_name, self.title, self.sub = icon_name, title, sub
        self._h = 92
        self.setMinimumHeight(self._h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._hover = False

    def set_dense(self, level: int):
        self._h = {0: 92, 1: 78, 2: 66}[level]
        self.setMinimumHeight(self._h)
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        return QSize(320, self._h)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        on = self.isChecked()
        if on:
            p.setBrush(QColor("#f1f8f6"))
            p.setPen(QPen(QColor(theme.ACCENT), 1.8))
        else:
            p.setBrush(QColor("#ffffff" if not self._hover else theme.SURFACE_2))
            p.setPen(QPen(QColor(theme.BORDER_2 if not self._hover else "#c3cbc0"), 1))
        p.drawRoundedRect(r, 14, 14)
        if not self.isEnabled():
            p.setOpacity(0.55)
        # 아이콘 원
        d = 54 if self._h >= 90 else (46 if self._h >= 75 else 40)
        c = QRectF(r.left() + 22, r.center().y() - d / 2, d, d)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.ACCENT_SOFT_2 if on else "#eff1ee"))
        p.drawEllipse(c)
        ic = 26
        p.drawPixmap(QPointF(c.center().x() - ic / 2, c.center().y() - ic / 2),
                     icons.pixmap(self.icon_name, theme.ACCENT_2 if on else "#56605b", ic, 2.0, 1.9))
        # 글자
        x = c.right() + 18
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.font(17, QFont.Weight.Bold))
        cy = r.center().y()
        p.drawText(QRectF(x, cy - 27, r.width() - x - 50, 26), Qt.AlignmentFlag.AlignVCenter, self.title)
        p.setPen(QColor(theme.TEXT_DIM))
        p.setFont(theme.font(13))
        p.drawText(QRectF(x, cy + 1, r.width() - x - 40, 22), Qt.AlignmentFlag.AlignVCenter, self.sub)
        # 라디오
        rr = QRectF(r.right() - 38, r.top() + (16 if self._h >= 90 else 12), 22, 22)
        if on:
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.setBrush(QColor("#ffffff"))
            p.drawEllipse(rr)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.ACCENT))
            p.drawEllipse(rr.adjusted(5, 5, -5, -5))
        else:
            p.setPen(QPen(QColor("#c3cbc0"), 1.4))
            p.setBrush(QColor("#ffffff"))
            p.drawEllipse(rr)
        if self.hasFocus():
            p.setPen(QPen(QColor(theme.ACCENT), 1, Qt.PenStyle.DotLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(3, 3, -3, -3), 12, 12)
        p.end()


# =============================================================== 키 슬라이더 ===

class KeySlider(QWidget):
    """-12~+12 반음 슬라이더: 눈금, 숫자, 현재 값 말풍선."""

    valueChanged = Signal(int)

    def __init__(self, lo: int = pitch.KEY_MIN, hi: int = pitch.KEY_MAX, parent=None):
        super().__init__(parent)
        self.lo, self.hi = lo, hi
        self._value = 0
        self._drag = False
        self._track_y = 58
        self.setMinimumHeight(104)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("반음 단위로 키를 조절합니다 (−12 ~ +12). 방향키·마우스 휠로도 조절할 수 있습니다.")

    def value(self) -> int:
        return self._value

    def setValue(self, v: int):
        v = max(self.lo, min(self.hi, int(v)))
        if v != self._value:
            self._value = v
            self.update()
            self.valueChanged.emit(v)

    def set_dense(self, level: int):
        self._track_y = {0: 58, 1: 54, 2: 50}[level]
        self.setMinimumHeight(self._track_y + 42)
        self.setMaximumHeight(self._track_y + 46)
        self.update()

    def _geo(self):
        m = 20
        return m, self.width() - m, self._track_y

    def _x_for(self, v: int) -> float:
        x0, x1, _ = self._geo()
        return x0 + (v - self.lo) / (self.hi - self.lo) * (x1 - x0)

    def _v_for(self, x: float) -> int:
        x0, x1, _ = self._geo()
        t = (x - x0) / max(1, (x1 - x0))
        return int(round(self.lo + t * (self.hi - self.lo)))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._drag = True
            self.setValue(self._v_for(e.position().x()))

    def mouseMoveEvent(self, e):
        if self._drag:
            self.setValue(self._v_for(e.position().x()))

    def mouseReleaseEvent(self, e):
        self._drag = False

    def mouseDoubleClickEvent(self, e):
        self.setValue(0)

    def wheelEvent(self, e):
        if not self.isEnabled():
            return
        d = e.angleDelta().y() or e.angleDelta().x()
        if d:
            self.setValue(self._value + (1 if d > 0 else -1))
        e.accept()

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.setValue(self._value + 1)
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.setValue(self._value - 1)
        elif k in (Qt.Key.Key_Home, Qt.Key.Key_0):
            self.setValue(0)
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0, x1, ty = self._geo()
        en = self.isEnabled()
        acc = QColor(theme.ACCENT if en else "#b9cfc8")
        # 트랙
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#e8ebe6"))
        p.drawRoundedRect(QRectF(x0, ty - 3, x1 - x0, 6), 3, 3)
        xz, xv = self._x_for(0), self._x_for(self._value)
        if self._value:
            p.setBrush(acc)
            p.drawRoundedRect(QRectF(min(xz, xv), ty - 3, abs(xv - xz), 6), 3, 3)
        # 눈금
        for v in range(self.lo, self.hi + 1):
            x = self._x_for(v)
            major = v % 3 == 0
            p.setPen(QPen(QColor("#b8c0b9" if major else "#d4d9d2"), 1.2))
            p.drawLine(QPointF(x, ty + 10), QPointF(x, ty + (17 if major else 14)))
            if major:
                p.setPen(QColor(theme.TEXT if v == self._value else theme.TEXT_DIM))
                p.setFont(theme.font(12, QFont.Weight.DemiBold if v == self._value else QFont.Weight.Normal))
                txt = "0" if v == 0 else f"{v:+d}"
                p.drawText(QRectF(x - 20, ty + 19, 40, 18), Qt.AlignmentFlag.AlignCenter, txt)
        # 핸들
        p.setPen(QPen(QColor("#ffffff"), 3))
        p.setBrush(acc)
        p.drawEllipse(QPointF(xv, ty), 9.5, 9.5)
        # 말풍선
        main = "0" if self._value == 0 else f"{self._value:+d}"
        sub = "원곡 키" if self._value == 0 else ("반음 올림" if self._value > 0 else "반음 내림")
        bw, bh = 70, 40
        bx = max(2, min(self.width() - bw - 2, xv - bw / 2))
        bubble = QRectF(bx, 2, bw, bh)
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QColor("#ffffff"))
        p.drawRoundedRect(bubble, 9, 9)
        tri = QPainterPath()
        tri.moveTo(xv - 6, bubble.bottom() - 0.5)
        tri.lineTo(xv, bubble.bottom() + 6)
        tri.lineTo(xv + 6, bubble.bottom() - 0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(tri)
        p.setPen(QColor(theme.ACCENT_2 if self._value else theme.TEXT))
        p.setFont(theme.font(15, QFont.Weight.Bold))
        p.drawText(QRectF(bx, 3, bw, 21), Qt.AlignmentFlag.AlignCenter, main)
        p.setPen(QColor(theme.TEXT_DIM))
        p.setFont(theme.font(10))
        p.drawText(QRectF(bx, 22, bw, 16), Qt.AlignmentFlag.AlignCenter, sub)
        if self.hasFocus():
            p.setPen(QPen(QColor(theme.ACCENT), 1, Qt.PenStyle.DotLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(xv, ty), 13, 13)
        p.end()


# ============================================================ 진행 상황 ===

class StatusPanel(QFrame):
    cancelRequested = Signal()
    fastRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)
        self.steps_row = QHBoxLayout()
        self.steps_row.setSpacing(6)
        lay.addLayout(self.steps_row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedHeight(10)
        lay.addWidget(self.bar)
        row = QHBoxLayout()
        row.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.msg = QLabel("준비 중…")
        self.msg.setFont(theme.font(14, QFont.Weight.DemiBold))
        self.msg.setWordWrap(True)
        self.sub = label("", "dim", wrap=True)
        col.addWidget(self.msg)
        col.addWidget(self.sub)
        row.addLayout(col, 1)
        self.fast_btn = text_button("빠른 모드로 다시", "refresh", None, 40, theme.TEXT_DIM)
        self.fast_btn.setToolTip("CPU 에서 너무 오래 걸리면 속도 우선(Demucs)으로 다시 처리합니다.")
        self.fast_btn.clicked.connect(self.fastRequested)
        self.fast_btn.hide()
        self.cancel_btn = text_button("취소", "x", "danger", 40, theme.DANGER)
        self.cancel_btn.clicked.connect(self.cancelRequested)
        row.addWidget(self.fast_btn, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(row)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("background:#fff7ec; border:1px solid #f5d2a6; border-radius:10px;"
                                  " padding:8px 10px; color:#9a5a12; font-size:12px;")
        self.notice.hide()
        lay.addWidget(self.notice)
        self._labels: dict[str, QLabel] = {}
        self._order: list[str] = []

    def set_steps(self, steps: list[tuple[str, str]]):
        while self.steps_row.count():
            it = self.steps_row.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self._labels.clear()
        self._order = [k for k, _ in steps]
        for i, (k, name) in enumerate(steps):
            lb = QLabel(f"○ {name}")
            lb.setObjectName("StageTodo")
            self._labels[k] = lb
            self.steps_row.addWidget(lb)
            if i < len(steps) - 1:
                arrow = QLabel("›")
                arrow.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:13px;")
                self.steps_row.addWidget(arrow)
        self.steps_row.addStretch(1)
        self.notice.hide()
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("취소")

    def set_stage(self, key: str):
        if key not in self._labels:
            return
        idx = self._order.index(key)
        for i, k in enumerate(self._order):
            lb = self._labels[k]
            name = lb.text()[2:]
            if i < idx:
                lb.setText(f"✓ {name}")
                lb.setObjectName("StageDone")
            elif i == idx:
                lb.setText(f"● {name}")
                lb.setObjectName("StageActive")
            else:
                lb.setText(f"○ {name}")
                lb.setObjectName("StageTodo")
            repolish(lb)

    def finish_all(self):
        for k in self._order:
            lb = self._labels[k]
            lb.setText(f"✓ {lb.text()[2:]}")
            lb.setObjectName("StageDone")
            repolish(lb)

    def set_progress(self, overall: float, msg: str, sub: str = ""):
        self.bar.setValue(int(overall * 1000))
        self.msg.setText(msg)
        if sub:
            self.sub.setText(sub)

    def show_notice(self, text: str):
        self.notice.setText(text)
        self.notice.show()


class ResultPanel(QFrame):
    openFolder = Signal()
    openFile = Signal()
    showAdvanced = Signal()
    newJob = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ResultPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(10)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("check-circle", theme.ACCENT, 26))
        self.title = QLabel("완료되었습니다")
        self.title.setObjectName("ResultTitle")
        head.addWidget(ic)
        head.addWidget(self.title, 1)
        lay.addLayout(head)
        self.files = QLabel("")
        self.files.setWordWrap(True)
        self.files.setTextFormat(Qt.TextFormat.RichText)
        self.files.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.files.setStyleSheet(f"color:{theme.TEXT}; font-size:13px;")
        lay.addWidget(self.files)
        self.detail = label("", "dim", wrap=True)
        lay.addWidget(self.detail)
        row = QGridLayout()
        row.setHorizontalSpacing(8)
        row.setVerticalSpacing(8)
        self.folder_btn = text_button("폴더 열기", "folder", "primary", 42)
        self.folder_btn.clicked.connect(self.openFolder)
        self.file_btn = text_button("파일 열기", "play", None, 42, theme.TEXT_DIM)
        self.file_btn.setToolTip("기본 음악 앱으로 결과 파일을 엽니다.")
        self.file_btn.clicked.connect(self.openFile)
        self.adv_btn = text_button("고급 옵션에서 듣기", "wave", None, 42, theme.TEXT_DIM)
        self.adv_btn.setToolTip("고급 옵션 화면에서 보컬·반주 트랙을 재생하고, 다른 형식·결과물로 다시 저장할 수 있습니다.")
        self.adv_btn.clicked.connect(self.showAdvanced)
        self.new_btn = text_button("새 작업", "refresh", None, 42, theme.TEXT_DIM)
        self.new_btn.clicked.connect(self.newJob)
        self._btns = [self.folder_btn, self.file_btn, self.adv_btn, self.new_btn]
        self._grid = row
        lay.addLayout(row)
        self.set_columns(4)

    def set_columns(self, n: int):
        for b in self._btns:
            self._grid.removeWidget(b)
        for i, b in enumerate(self._btns):
            self._grid.addWidget(b, i // n, i % n)


class ErrorPanel(QFrame):
    retry = Signal()
    fast = Signal()
    update_youtube = Signal()
    dismiss = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ErrorPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(10)
        head = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(icons.pixmap("alert", theme.DANGER, 24))
        self.title = QLabel("처리하지 못했습니다")
        self.title.setObjectName("ErrorTitle")
        head.addWidget(ic)
        head.addWidget(self.title, 1)
        lay.addLayout(head)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.msg.setStyleSheet(f"color:{theme.TEXT}; font-size:13px;")
        lay.addWidget(self.msg)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.retry_btn = text_button("다시 시도", "refresh", "primary", 40)
        self.retry_btn.clicked.connect(self.retry)
        self.fast_btn = text_button("빠른 모델로 진행", "sparkles", None, 40, theme.TEXT_DIM)
        self.fast_btn.setToolTip("최고 품질 모델을 쓸 수 없을 때, 설치된 Demucs 보컬 모델(속도 우선)로 처리합니다.")
        self.fast_btn.clicked.connect(self.fast)
        self.update_btn = text_button("YouTube 기능 업데이트", "download", None, 40, theme.TEXT_DIM)
        self.update_btn.clicked.connect(self.update_youtube)
        self.close_btn = text_button("닫기", None, None, 40)
        self.close_btn.clicked.connect(self.dismiss)
        for b in (self.retry_btn, self.fast_btn, self.update_btn, self.close_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

    def show_error(self, title: str, msg: str, retry: bool, fast: bool, update: bool):
        self.title.setText(title)
        self.msg.setText(msg)
        self.retry_btn.setVisible(retry)
        self.fast_btn.setVisible(fast)
        self.update_btn.setVisible(update)


# ================================================================ 홈 화면 ===

class HomePage(QWidget):
    advancedRequested = Signal()
    settingsRequested = Signal()
    startRequested = Signal()
    browseRequested = Signal()
    fileChosen = Signal(object)        # Path
    sourceCleared = Signal()
    keyChanged = Signal(int)
    targetChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HomeRoot")
        self.file_path: Optional[Path] = None
        self._compact = None
        self._has_result = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_top())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        holder = QWidget()
        holder.setObjectName("HomeRoot")
        hl = QHBoxLayout(holder)
        self._outer = hl
        hl.setContentsMargins(24, 8, 24, 28)
        hl.addStretch(1)
        self.card = self._build_card()
        hl.addWidget(self.card, 100)
        hl.addStretch(1)
        scroll.setWidget(holder)
        self.scroll = scroll
        root.addWidget(scroll, 1)

    # --------------------------------------------------------------- 상단 ---
    def _build_top(self) -> QWidget:
        top = QWidget()
        top.setObjectName("HomeTop")
        lay = QHBoxLayout(top)
        self._top_lay = lay
        lay.setContentsMargins(26, 14, 22, 8)
        lay.setSpacing(12)
        self.logo = LogoTile()
        self.logo.setFixedSize(50, 50)
        lay.addWidget(self.logo)
        t = QLabel(APP_SHORT)
        t.setObjectName("AppTitle")
        b = QLabel(APP_BRAND)
        b.setObjectName("AppBrand")
        self.brand = b
        lay.addWidget(t, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        self.adv_btn = QPushButton("고급 옵션")
        self.adv_btn.setProperty("kind", "link")
        self.adv_btn.setIcon(icons.icon("chevron-right", theme.TEXT_DIM, 16))
        self.adv_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.adv_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.adv_btn.setToolTip("분리 모델·처리 방식·출력 형식, 트랙별 재생과 저장 (기존 화면)")
        self.adv_btn.clicked.connect(self.advancedRequested)
        lay.addWidget(self.adv_btn)
        self.gear = icon_button("settings", "설정 (저장 위치·출력 형식)", 46, 20, "square", theme.TEXT_DIM)
        self.gear.setStyleSheet("QPushButton { border-radius: 23px; }")
        self.gear.clicked.connect(self.settingsRequested)
        lay.addWidget(self.gear)
        return top

    # --------------------------------------------------------------- 카드 ---
    def _build_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("HomeCard")
        card.setMaximumWidth(1040)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        _shadow(card)
        lay = QVBoxLayout(card)
        self._card_lay = lay
        lay.setContentsMargins(40, 34, 40, 26)
        lay.setSpacing(14)

        self.title = QLabel("음원 분리, 한 번에 간편하게")
        self.title.setObjectName("HomeTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setWordWrap(True)
        self.subtitle = QLabel("음원 파일 또는 YouTube URL을 입력하세요.")
        self.subtitle.setObjectName("HomeSub")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.subtitle)
        self._gaps = [lay.count()]
        lay.addSpacing(8)

        # A. 입력
        self.drop = HomeDropZone()
        self.drop.browseRequested.connect(self.browseRequested)
        self.drop.cleared.connect(self._clear_file)
        self.drop.fileDropped.connect(self._dropped)
        lay.addWidget(self.drop)
        orow = QHBoxLayout()
        orow.setSpacing(14)
        for _ in range(2):
            pass
        l1, l2 = QFrame(), QFrame()
        for ln in (l1, l2):
            ln.setFrameShape(QFrame.Shape.HLine)
            ln.setStyleSheet(f"color:{theme.BORDER}; background:{theme.BORDER}; max-height:1px; border:none;")
        mid = QLabel("또는")
        mid.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:13px;")
        orow.addWidget(l1, 1)
        orow.addWidget(mid)
        orow.addWidget(l2, 1)
        lay.addLayout(orow)
        yrow = QHBoxLayout()
        yrow.setSpacing(10)
        self.yt_badge = YouTubeBadge()
        self.url = QLineEdit()
        self.url.setObjectName("UrlEdit")
        self.url.setPlaceholderText("YouTube URL을 붙여넣으세요.")
        self.url.setClearButtonEnabled(True)
        self.url.setMinimumHeight(50)
        self.url.textChanged.connect(self._url_changed)
        self.url.returnPressed.connect(self.startRequested)
        yrow.addWidget(self.yt_badge)
        yrow.addWidget(self.url, 1)
        lay.addLayout(yrow)
        srow = QHBoxLayout()
        srow.setSpacing(8)
        self.active_badge = QLabel("")
        self.active_badge.setObjectName("ActiveBadge")
        self.active_badge.hide()
        self.input_hint = QLabel("")
        self.input_hint.setObjectName("HomeHint")
        self.input_hint.setWordWrap(True)
        self.input_hint.hide()
        srow.addWidget(self.active_badge)
        srow.addWidget(self.input_hint, 1)
        lay.addLayout(srow)
        self._gaps.append(lay.count())
        lay.addSpacing(6)

        # B. 추출 대상
        lay.addWidget(self._section("추출할 음원"))
        self.target_box = QWidget()
        self.target_lay = QGridLayout(self.target_box)
        self.target_lay.setContentsMargins(0, 0, 0, 0)
        self.target_lay.setHorizontalSpacing(14)
        self.target_lay.setVerticalSpacing(10)
        self.vocal_card = TargetCard("mic", "보컬만", "보컬 소리만 추출합니다.")
        self.mr_card = TargetCard("music", "반주 (MR)만", "보컬을 제거한 반주만 추출합니다.")
        self.target_group = QButtonGroup(self)
        self.target_group.addButton(self.vocal_card)
        self.target_group.addButton(self.mr_card)
        self.mr_card.setChecked(True)
        self.vocal_card.toggled.connect(lambda on: on and self.targetChanged.emit("vocals"))
        self.mr_card.toggled.connect(lambda on: on and self.targetChanged.emit("mr"))
        lay.addWidget(self.target_box)
        # YouTube 입력일 때만: 분리하지 않고 받은 오디오를 그대로(키 조절은 적용) 저장
        self.orig_box = QCheckBox("분리하지 않고 원본 오디오 그대로 저장")
        self.orig_box.setToolTip("YouTube에서 받은 오디오를 보컬/반주 분리 없이 저장합니다. 키 조절은 그대로 적용됩니다.")
        self.orig_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.orig_box.toggled.connect(self._orig_toggled)
        self.orig_box.hide()
        lay.addWidget(self.orig_box)
        self._gaps.append(lay.count())
        lay.addSpacing(6)

        # C. 키 조절
        khead = QHBoxLayout()
        khead.setSpacing(12)
        khead.addWidget(self._section("키 조절"))
        kh = QLabel("−12 ~ +12 (반음 단위)로 조절할 수 있습니다. 템포와 곡 길이는 그대로입니다.")
        kh.setObjectName("HomeHint")
        kh.setWordWrap(True)
        khead.addWidget(kh, 1)
        self.key_reset = QPushButton("원래 키")
        self.key_reset.setProperty("kind", "link")
        self.key_reset.setIcon(icons.icon("refresh", theme.TEXT_DIM, 15))
        self.key_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.key_reset.setToolTip("키를 원곡(0)으로 되돌립니다")
        self.key_reset.clicked.connect(lambda: self.key.setValue(0))
        khead.addWidget(self.key_reset)
        lay.addLayout(khead)
        krow = QHBoxLayout()
        krow.setSpacing(12)
        self.key_down = icon_button("minus", "반음 내리기", 50, 20, "square", theme.TEXT)
        self.key = KeySlider()
        self.key_up = icon_button("plus", "반음 올리기", 50, 20, "square", theme.TEXT)
        self.key_down.clicked.connect(lambda: self.key.setValue(self.key.value() - 1))
        self.key_up.clicked.connect(lambda: self.key.setValue(self.key.value() + 1))
        self.key.valueChanged.connect(self._key_changed)
        krow.addWidget(self.key_down, 0, Qt.AlignmentFlag.AlignBottom)
        krow.addWidget(self.key, 1)
        krow.addWidget(self.key_up, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(krow)
        self.key_note = QLabel("")
        self.key_note.setObjectName("HomeHint")
        self.key_note.setWordWrap(True)
        lay.addWidget(self.key_note)

        # E. 실행 / 진행 / 결과
        self.start_btn = text_button("음원 추출 시작", None, "primary", 60)
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.setIcon(icons.icon("arrow-right", "#ffffff", 22))
        self.start_btn.setIconSize(QSize(22, 22))
        self.start_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.start_btn.clicked.connect(self.startRequested)
        self._gaps.append(lay.count())
        lay.addSpacing(4)
        lay.addWidget(self.start_btn)
        self.status = StatusPanel()
        self.status.hide()
        lay.addWidget(self.status)
        self.result = ResultPanel()
        self.result.hide()
        lay.addWidget(self.result)
        self.error = ErrorPanel()
        self.error.hide()
        lay.addWidget(self.error)
        self.foot = QLabel("")
        self.foot.setObjectName("HomeFoot")
        self.foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.foot.setWordWrap(True)
        self.foot.setTextFormat(Qt.TextFormat.RichText)
        self.foot.setOpenExternalLinks(False)
        self.foot.linkActivated.connect(lambda _l: self.settingsRequested.emit())
        lay.addWidget(self.foot)
        self._layout_targets(compact=False)
        self._update_key_note()
        return card

    def _section(self, text: str) -> QLabel:
        lb = QLabel(text)
        lb.setObjectName("HomeSection")
        return lb

    def _layout_targets(self, compact: bool):
        for w in (self.vocal_card, self.mr_card):
            self.target_lay.removeWidget(w)
        if compact:
            self.target_lay.addWidget(self.vocal_card, 0, 0)
            self.target_lay.addWidget(self.mr_card, 1, 0)
        else:
            self.target_lay.addWidget(self.vocal_card, 0, 0)
            self.target_lay.addWidget(self.mr_card, 0, 1)

    # 창 높이에 따른 배치 단계 (일반 배치 전체 높이 ≈ 952px, 촘촘한 배치 ≈ 790px 기준)
    DENSE_BELOW = 958
    TIGHT_BELOW = 795

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.brand.setVisible(self.width() >= 560)
        compact = self.width() < 760
        h = self.height()
        level = 2 if h < self.TIGHT_BELOW else (1 if h < self.DENSE_BELOW else 0)
        if (compact, level) != self._compact:
            self._compact = (compact, level)
            self._layout_targets(compact)
            m = 18 if compact else 40
            top, bottom = {0: (26, 22) if compact else (34, 24), 1: (18, 14), 2: (12, 10)}[level]
            self._card_lay.setContentsMargins(m, top, m, bottom)
            self._card_lay.setSpacing({0: 14, 1: 10, 2: 8}[level])
            gaps = {0: (8, 6, 6, 4), 1: (2, 0, 0, 2), 2: (0, 0, 0, 0)}[level]
            for i, g in zip(self._gaps, gaps):
                item = self._card_lay.itemAt(i)
                if item is not None and item.spacerItem() is not None:
                    item.spacerItem().changeSize(0, g)
            self._card_lay.invalidate()
            self._outer.setContentsMargins(10 if compact else 24, 2 if level else 4, 10 if compact else 24,
                                           {0: 18, 1: 12, 2: 8}[level])
            size = 23 if compact else {0: 30, 1: 26, 2: 23}[level]
            self.title.setStyleSheet(f"font-size:{size}px;")
            self.result.set_columns(2 if compact else 4)
            self.drop.set_dense(level, compact)
            for c in (self.vocal_card, self.mr_card):
                c.set_dense(level)
            self.key.set_dense(level)
            self.url.setMinimumHeight({0: 50, 1: 44, 2: 40}[level])
            self.yt_badge.setFixedSize(52 if level else 56, {0: 50, 1: 44, 2: 40}[level])
            self.start_btn.setFixedHeight({0: 60, 1: 52, 2: 48}[level])
            self._top_lay.setContentsMargins(26 if level < 2 else 18, 14 if level == 0 else 8, 22, 8 if level < 2 else 4)
            self.logo.setFixedSize(*((50, 50) if level < 2 else (42, 42)))

    # -------------------------------------------------------------- 입력 ---
    def _dropped(self, obj):
        if isinstance(obj, str):
            if youtube.video_id(obj):
                self.url.setText(obj.strip())
            return
        self.fileChosen.emit(Path(obj))

    def set_file(self, path: Path, detail: str = ""):
        self.file_path = Path(path)
        self.url.blockSignals(True)
        self.url.clear()
        self.url.blockSignals(False)
        self.drop.set_file(path, detail)
        self._refresh_source()

    def set_file_detail(self, text: str):
        self.drop.set_detail(text)

    def _clear_file(self):
        self.file_path = None
        self.drop.set_empty()
        self._refresh_source()
        self.sourceCleared.emit()

    def _url_changed(self, text: str):
        if text.strip() and self.file_path is not None:
            self.file_path = None
            self.drop.set_empty()
        self._refresh_source()

    def source_text(self) -> str:
        t = self.url.text().strip()
        if t:
            return t
        return str(self.file_path) if self.file_path else ""

    def source_kind(self) -> str:
        t = self.url.text().strip()
        if t:
            return "youtube" if youtube.video_id(t) else "invalid_url"
        return "file" if self.file_path else ""

    def _refresh_source(self):
        kind = self.source_kind()
        url_active = kind == "youtube"
        invalid = kind == "invalid_url"
        self.url.setProperty("active", "true" if url_active else "false")
        self.url.setProperty("invalid", "true" if invalid else "false")
        repolish(self.url)
        self.yt_badge.active = url_active
        self.yt_badge.update()
        self.drop.set_dimmed(url_active or invalid)
        if kind == "file":
            self.active_badge.setText("사용할 입력: 파일")
            self.active_badge.show()
            self.input_hint.setText("로컬 파일을 그대로 읽어 처리합니다 (원본 파일은 바뀌지 않습니다).")
        elif url_active:
            self.active_badge.setText("사용할 입력: YouTube")
            self.active_badge.show()
            self.input_hint.setText("영상에서 가장 좋은 오디오 전용 스트림을 재인코딩 없이 받아 처리합니다.")
        elif invalid:
            self.active_badge.hide()
            self.input_hint.setText("YouTube 영상 주소가 아닙니다. 예: https://www.youtube.com/watch?v=…")
        else:
            self.active_badge.hide()
            self.input_hint.setText("")
        self.input_hint.setVisible(bool(self.input_hint.text()))
        if not hasattr(self, "orig_box"):
            return
        self.orig_box.setVisible(url_active)
        status = getattr(self, "status", None)
        busy = status is not None and not status.isHidden()
        orig = self.original_mode()
        for c in (self.vocal_card, self.mr_card):
            c.setEnabled(not busy and not orig)
        self.orig_box.setEnabled(not busy)
        self._update_start_text()

    def original_mode(self) -> bool:
        return self.source_kind() == "youtube" and self.orig_box.isChecked()

    def target(self) -> str:
        if self.original_mode():
            return "original"
        return "vocals" if self.vocal_card.isChecked() else "mr"

    def set_target(self, t: str):
        self.orig_box.setChecked(t == "original")
        if t != "original":
            (self.vocal_card if t == "vocals" else self.mr_card).setChecked(True)
        self._refresh_source()

    def _orig_toggled(self, _on: bool):
        self._refresh_source()
        self.targetChanged.emit(self.target())

    def _update_start_text(self):
        if not hasattr(self, "start_btn"):
            return
        orig = self.original_mode()
        if self._has_result:
            self.start_btn.setText("다시 저장" if orig else "다시 추출")
        else:
            self.start_btn.setText("원본 저장 시작" if orig else "음원 추출 시작")

    # --------------------------------------------------------------- 키 ---
    def _key_changed(self, v: int):
        self._update_key_note()
        self.keyChanged.emit(v)

    def set_key(self, v: int):
        self.key.blockSignals(True)
        self.key.setValue(v)
        self.key.blockSignals(False)
        self.key.update()
        self._update_key_note()

    def _update_key_note(self):
        v = self.key.value()
        if v == 0:
            self.key_note.setText("")
            self.key_note.hide()
        else:
            extra = " 키를 크게 바꿀수록(±7 이상) 음질 변화가 커질 수 있습니다." if abs(v) >= 7 else ""
            self.key_note.setText(f"선택한 결과물에만 {abs(v)}반음 {'올려' if v > 0 else '내려'} 저장합니다."
                                  f"{extra}")
            self.key_note.show()
        self.key_reset.setEnabled(v != 0)

    # ------------------------------------------------------------ 상태 표시 ---
    def set_footer(self, fmt_label: str, out_dir: Path):
        d = str(out_dir)
        if len(d) > 46:
            d = "…" + d[-44:]
        self.foot.setText(f"2-stem 분리 · 자동 입력 감지 · 키 변경 &nbsp;|&nbsp; 출력: {fmt_label} · 저장 위치: {d} "
                          f"<a href='settings' style='color:{theme.ACCENT_2}; text-decoration:none;'>변경</a>")

    def set_busy(self, busy: bool):
        for w in (self.drop, self.url, self.vocal_card, self.mr_card, self.orig_box, self.key, self.key_down,
                  self.key_up,
                  self.key_reset, self.adv_btn, self.gear):
            w.setEnabled(not busy)
        self.start_btn.setVisible(not busy)
        if busy:
            self.result.hide()
            self.error.hide()
            self.status.show()
            self._scroll_to(self.status)
        else:
            self.status.hide()
            self._refresh_source()
            self.adv_btn.setEnabled(True)
            self.gear.setEnabled(True)

    def show_result(self, title: str, files_html: str, detail: str):
        self.result.title.setText(title)
        self.result.files.setText(files_html)
        self.result.detail.setText(detail)
        self.error.hide()
        self.result.show()
        self._has_result = True
        self._update_start_text()
        self._scroll_to(self.result)

    def show_error(self, title: str, msg: str, retry: bool, fast: bool, update: bool):
        self.error.show_error(title, msg, retry, fast, update)
        self.result.hide()
        self.error.show()
        self._scroll_to(self.error)

    def reset_panels(self):
        self.result.hide()
        self.error.hide()
        self._has_result = False
        self._update_start_text()

    def _scroll_to(self, w: QWidget):
        from PySide6.QtCore import QTimer

        QTimer.singleShot(50, lambda: self.scroll.ensureWidgetVisible(w, 0, 40))


def fmt_seconds(sec: float) -> str:
    return audio_io.format_time(sec)
