"""화면 구성 요소 (시안 디자인 기준).

ToggleSwitch, ModeTile, ChipButton, PlayButton, LogoTile, Lcd,
Viewport(줌/스크롤 상태), WaveformView(파형 리전), TimeRuler, TrackCard, TrackRow, TracksPanel, DropZone
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import (Property, QEasingCurve, QObject, QPointF, QPropertyAnimation, QRectF, QSize, Qt,
                            Signal)
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractButton, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from . import icons, theme

HEADER_WIDTH = 318      # 트랙 카드 열 너비
COLUMN_GAP = 14
ROW_HEIGHT = 96
ROW_SPACING = 8


class ElideLabel(QLabel):
    """폭이 부족하면 글자 끝을 '…'로 줄여 표시 (전체 내용은 툴팁). 너무 좁으면 비움."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def setText(self, text: str):
        self._full = text or ""
        self.setToolTip(self._full)
        self._apply()

    def text(self) -> str:
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply()

    def _apply(self):
        w = self.width()
        shown = "" if w < 60 else self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, w)
        QLabel.setText(self, shown)


def glow(widget: QWidget, color: str = theme.ACCENT, radius: int = 22, alpha: int = 70,
         dy: float = 3.0) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    c = QColor(color)
    c.setAlpha(alpha)
    eff.setColor(c)
    eff.setBlurRadius(radius)
    eff.setOffset(0, dy)
    widget.setGraphicsEffect(eff)


def icon_button(name: str, tip: str = "", size: int = 40, icon_size: int = 18, kind: str = "square",
                color: str = theme.TEXT) -> QPushButton:
    b = QPushButton()
    b.setProperty("kind", kind)
    b.setIcon(icons.icon(name, color, icon_size))
    b.setIconSize(QSize(icon_size, icon_size))
    b.setFixedSize(size, size)
    b.setToolTip(tip)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def text_button(text: str, icon_name: Optional[str] = None, kind: Optional[str] = None,
                height: int = 40, icon_color: str = theme.TEXT) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if icon_name:
        b.setIcon(icons.icon(icon_name, "#ffffff" if kind == "primary" else icon_color, 18))
        b.setIconSize(QSize(17, 17))
    b.setFixedHeight(height)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def label(text: str, role: Optional[str] = None, name: Optional[str] = None, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if role:
        lb.setProperty("role", role)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def repolish(w: QWidget) -> None:
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


# =========================================================== 토글 스위치 ===

class ToggleSwitch(QAbstractButton):
    def __init__(self, parent=None, on_color: str = theme.ACCENT):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(42, 24)
        self._on_color = QColor(on_color)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, v: bool):  # 애니메이션 없이 즉시 반영 (초기화 시)
        super().setChecked(v)
        self._anim.stop()
        self._pos = 1.0 if v else 0.0
        self.update()

    def get_knob(self):
        return self._pos

    def set_knob(self, v):
        self._pos = v
        self.update()

    knob = Property(float, get_knob, set_knob)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        off = QColor(theme.TOGGLE_OFF)
        on = QColor(self._on_color)
        t = self._pos
        col = QColor(int(off.red() + (on.red() - off.red()) * t), int(off.green() + (on.green() - off.green()) * t),
                     int(off.blue() + (on.blue() - off.blue()) * t))
        if not self.isEnabled():
            col = QColor("#e5e9e3")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 6
        x = r.left() + 3 + (r.width() - d - 6) * t
        p.setPen(QPen(QColor(0, 0, 0, 22), 1))
        p.setBrush(QColor("#ffffff") if self.isEnabled() else QColor("#f6f7f4"))
        p.drawEllipse(QRectF(x, r.top() + 3, d, d))
        p.end()


# ============================================================ 모드 타일 ===

class ModeTile(QAbstractButton):
    def __init__(self, title: str, desc: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.title = title
        self.desc = desc
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._hover = False
        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, on: bool):
        if on:
            glow(self, theme.ACCENT, 24, 55, 2)
        else:
            self.setGraphicsEffect(None)
        self.update()

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def sizeHint(self):
        return QSize(100, 118)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self.isChecked():
            p.setBrush(QColor(theme.ACCENT_SOFT))
            p.setPen(QPen(QColor(theme.ACCENT), 1.6))
        else:
            p.setBrush(QColor("#ffffff") if not self._hover else QColor(theme.SURFACE_2))
            p.setPen(QPen(QColor(theme.BORDER if not self._hover else "#c8cfc5"), 1))
        p.drawRoundedRect(r, 12, 12)
        enabled = self.isEnabled()
        p.setPen(QColor(theme.ACCENT if self.isChecked() else (theme.TEXT if enabled else theme.TEXT_FAINT)))
        p.setFont(theme.font(15, QFont.Weight.Bold))
        p.drawText(QRectF(r.left(), r.top() + 18, r.width(), 24), Qt.AlignmentFlag.AlignCenter, self.title)
        p.setPen(QColor("#2f6f62" if self.isChecked() else theme.TEXT_DIM))
        p.setFont(theme.font(12))
        p.drawText(QRectF(r.left() + 8, r.top() + 48, r.width() - 16, r.height() - 56),
                   int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
                   self.desc)
        p.end()


# ================================================================= 칩 ===

class ChipButton(QAbstractButton):
    """MR 에서 제외할 악기. 체크 = 제외."""

    def __init__(self, text: str, icon_name: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text)
        self.icon_name = icon_name
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedHeight(40)
        self._hover = False
        self.toggled.connect(lambda on: (glow(self, theme.DANGER, 18, 45, 1) if on
                                         else self.setGraphicsEffect(None)))

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self.text()) + 50, 40)

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
            p.setBrush(QColor(theme.DANGER_SOFT))
            p.setPen(QPen(QColor("#f0a2a8"), 1.4))
        else:
            p.setBrush(QColor("#ffffff" if not self._hover else theme.SURFACE_2))
            p.setPen(QPen(QColor(theme.BORDER_2 if not self._hover else "#c3cbc0"), 1))
        p.drawRoundedRect(r, 10, 10)
        name = "mic-off" if on and self.icon_name == "mic" else self.icon_name
        col = theme.DANGER if on else theme.TEXT_DIM
        fm = self.fontMetrics()
        tw = fm.horizontalAdvance(self.text())
        total = 18 + 8 + tw
        x = r.center().x() - total / 2
        p.drawPixmap(QPointF(x, r.center().y() - 9), icons.pixmap(name, col, 18))
        p.setPen(QColor(theme.DANGER if on else theme.TEXT))
        p.setFont(self.font())
        p.drawText(QRectF(x + 26, r.top(), tw + 4, r.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        p.end()


# ========================================================== 재생 버튼 ===

class PlayButton(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 54)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.playing = False
        glow(self, theme.ACCENT, 22, 110, 3)

    def set_playing(self, v: bool):
        if v != self.playing:
            self.playing = v
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        g = QLinearGradient(r.topLeft(), r.bottomLeft())
        if self.isEnabled():
            g.setColorAt(0, QColor(theme.ACCENT))
            g.setColorAt(1, QColor(theme.ACCENT))
        else:
            g.setColorAt(0, QColor("#bcd8d1"))
            g.setColorAt(1, QColor("#bcd8d1"))
        p.setBrush(g)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawEllipse(r)
        ic = "pause" if self.playing else "play"
        size = 22
        off = 1.5 if ic == "play" else 0
        p.drawPixmap(QPointF(r.center().x() - size / 2 + off, r.center().y() - size / 2),
                     icons.pixmap(ic, "#ffffff", size))
        p.end()


# ============================================================== 로고 ===

class LogoTile(QWidget):
    """헤더 로고: 흰 타일 위 네잎클로버."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(58, 58)

    def paintEvent(self, e):
        from .clover import paint_clover

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        g = QLinearGradient(r.topLeft(), r.bottomLeft())
        g.setColorAt(0, QColor("#ffffff"))
        g.setColorAt(1, QColor("#f2f6f3"))
        p.setBrush(g)
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRoundedRect(r, 15, 15)
        paint_clover(p, r.adjusted(10, 9, -10, -11))
        p.end()


# =============================================================== LCD ===

class Lcd(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Lcd")
        self.setFixedSize(196, 58)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 6, 18, 6)
        lay.setSpacing(10)

        def block(big: int, text: str, cap: str, color: str):
            col = QVBoxLayout()
            col.setSpacing(1)
            v = QLabel(text)
            v.setFont(theme.font(big, QFont.Weight.Bold, tabular=True))
            v.setStyleSheet(f"color:{color};")
            c = label(cap, "caps")
            c.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:9px; letter-spacing:1px;")
            col.addWidget(v)
            col.addWidget(c)
            return col, v

        c1, self.pos_lbl = block(20, "00:00.0", "POSITION", theme.TEXT)
        slash = QLabel("/")
        slash.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:16px;")
        c2, self.len_lbl = block(15, "00:00", "LENGTH", theme.TEXT_DIM)
        lay.addLayout(c1)
        lay.addWidget(slash, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(c2)


# ======================================================= 뷰포트(줌) ===

class Viewport(QObject):
    """전체 곡 중 화면에 보이는 구간 [start, start+span] (0~1 비율). 모든 파형/룰러가 공유."""

    changed = Signal()
    MAX_ZOOM = 16.0

    def __init__(self):
        super().__init__()
        self.start = 0.0
        self.span = 1.0

    @property
    def zoom(self) -> float:
        return 1.0 / self.span

    def set_zoom(self, zoom: float, anchor: Optional[float] = None):
        zoom = max(1.0, min(self.MAX_ZOOM, zoom))
        new_span = 1.0 / zoom
        if anchor is None:
            anchor = self.start + self.span / 2
        rel = (anchor - self.start) / self.span if self.span else 0.5
        self.span = new_span
        self.start = anchor - rel * new_span
        self._clamp()
        self.changed.emit()

    def scroll_to(self, start: float):
        self.start = start
        self._clamp()
        self.changed.emit()

    def ensure_visible(self, frac: float):
        if self.span >= 1.0:
            return
        if frac < self.start or frac > self.start + self.span:
            self.scroll_to(frac - self.span * 0.1)

    def reset(self):
        self.start, self.span = 0.0, 1.0
        self.changed.emit()

    def _clamp(self):
        self.span = max(1.0 / self.MAX_ZOOM, min(1.0, self.span))
        self.start = max(0.0, min(1.0 - self.span, self.start))

    def to_x(self, frac: float, width: float) -> float:
        return (frac - self.start) / self.span * width

    def to_frac(self, x: float, width: float) -> float:
        return self.start + x / max(1.0, width) * self.span


def _slice_peaks(peaks: np.ndarray, start: float, span: float, cols: int) -> np.ndarray:
    n = len(peaks)
    a = int(start * n)
    b = max(a + 1, int(np.ceil((start + span) * n)))
    seg = peaks[a:min(n, b)]
    m = len(seg)
    if m == 0:
        return np.zeros((cols, 2), np.float32)
    if m <= cols:
        idx = np.linspace(0, m - 1, cols).round().astype(int)
        return seg[idx]
    edges = np.minimum(np.linspace(0, m, cols + 1).astype(int), m - 1)
    return np.stack([np.minimum.reduceat(seg[:, 0], edges[:-1]), np.maximum.reduceat(seg[:, 1], edges[:-1])], 1)


# ============================================================ 파형 리전 ===

class WaveformView(QWidget):
    seekRequested = Signal(float)

    def __init__(self, viewport: Viewport, color: str, name: str, sub: str, parent=None):
        super().__init__(parent)
        self.vp = viewport
        self.vp.changed.connect(self._invalidate)
        self.color = QColor(color)
        self.name = name
        self.sub = sub
        self._peaks: Optional[np.ndarray] = None
        self._gain = 1.0
        self._dim = False
        self._pos = 0.0
        self._cache: Optional[QPixmap] = None
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_peaks(self, peaks: Optional[np.ndarray]):
        self._peaks = peaks
        if peaks is not None and len(peaks):
            peak = float(np.max(np.abs(peaks)))
            self._gain = 0.92 / max(peak, 0.92 / 3.0) if peak > 1e-6 else 1.0
        self._invalidate()

    def set_dimmed(self, v: bool):
        if v != self._dim:
            self._dim = v
            self._invalidate()

    def set_position(self, frac: float):
        x_old = self.vp.to_x(self._pos, self.width())
        x_new = self.vp.to_x(frac, self.width())
        self._pos = frac
        if abs(x_new - x_old) >= 0.5:
            self.update()

    def _invalidate(self):
        self._cache = None
        self.update()

    def resizeEvent(self, e):
        self._invalidate()

    def _render(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        w, h = self.width(), self.height()
        pm = QPixmap(max(1, int(w * dpr)), max(1, int(h * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        base = QColor(self.color)
        col = base.name()
        x0 = self.vp.to_x(0.0, w)
        x1 = self.vp.to_x(1.0, w)
        region = QRectF(max(x0, -20) + 0.5, 2.5, min(x1, w + 20) - max(x0, -20) - 1, h - 5)
        clip = QPainterPath()
        clip.addRoundedRect(region, 10, 10)
        # 파스텔 배경 + 상단 라벨 띠
        p.fillPath(clip, theme.tint(col, 0.05 if self._dim else 0.10))
        p.save()
        p.setClipPath(clip)
        strip = QRectF(region.left(), region.top(), region.width(), 22)
        p.fillRect(strip, theme.tint(col, 0.08 if self._dim else 0.17))
        p.restore()
        p.setPen(QPen(theme.tint(col, 0.22 if self._dim else 0.42), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(clip)

        # 라벨 (보이는 영역 왼쪽에 고정)
        lx = max(region.left(), 0) + 10
        p.setFont(theme.font(13, QFont.Weight.Bold))
        p.setPen(QColor(theme.TEXT_FAINT) if self._dim else theme.shade(col, 0.28))
        nw = p.fontMetrics().horizontalAdvance(self.name)
        p.drawText(QRectF(lx, strip.top(), nw + 4, strip.height()), Qt.AlignmentFlag.AlignVCenter, self.name)
        p.setFont(theme.font(12))
        p.setPen(QColor(theme.TEXT_FAINT) if self._dim else theme.shade(col, 0.15))
        p.drawText(QRectF(lx + nw + 4, strip.top(), 300, strip.height()), Qt.AlignmentFlag.AlignVCenter,
                   f"·  {self.sub}")
        if self._dim:
            base = theme.tint(col, 0.35)

        if self._peaks is not None and len(self._peaks):
            area = QRectF(0, strip.bottom() + 3, w, region.bottom() - strip.bottom() - 6)
            cols = max(2, int(w))
            pk = _slice_peaks(self._peaks, self.vp.start, self.vp.span, cols)
            mid = area.center().y()
            half = area.height() / 2
            g = self._gain
            top = [QPointF(i, mid - float(pk[i, 1]) * g * half) for i in range(cols)]
            bot = [QPointF(i, mid - float(pk[i, 0]) * g * half) for i in reversed(range(cols))]
            poly = QPolygonF(top + bot)
            p.save()
            p.setClipPath(clip)
            hi = QColor(base)
            lo = QColor(base).lighter(118)
            wg = QLinearGradient(0, area.top(), 0, area.bottom())
            wg.setColorAt(0.0, lo)
            wg.setColorAt(0.5, hi)
            wg.setColorAt(1.0, lo)
            p.setPen(QPen(QColor(base), 0.8))
            p.setBrush(wg)
            p.drawPolygon(poly)
            p.restore()
        p.end()
        return pm

    def paintEvent(self, e):
        if self._cache is None:
            self._cache = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        if self._peaks is not None:
            x = self.vp.to_x(self._pos, self.width())
            if -1 <= x <= self.width() + 1:
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setPen(QPen(QColor(theme.ACCENT), 1.6))
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.end()

    def _seek(self, x: float):
        if self._peaks is not None:
            self.seekRequested.emit(max(0.0, min(1.0, self.vp.to_frac(x, self.width()))))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._seek(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._seek(e.position().x())

    def wheelEvent(self, e):
        if wheel_zoom(self.vp, e, self.width()):
            e.accept()
        else:
            super().wheelEvent(e)


def wheel_zoom(vp: Viewport, e, width: float) -> bool:
    """Ctrl/Cmd+휠 = 확대/축소, Shift+휠·가로 스크롤 = 좌우 이동. 처리했으면 True."""
    mods = e.modifiers()
    dy = e.angleDelta().y()
    dx = e.angleDelta().x()
    if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
        anchor = vp.to_frac(e.position().x(), width)
        vp.set_zoom(vp.zoom * (1.25 if dy > 0 else 0.8), anchor)
        return True
    if (mods & Qt.KeyboardModifier.ShiftModifier) or abs(dx) > abs(dy):
        delta = dx if abs(dx) > abs(dy) else dy
        vp.scroll_to(vp.start - delta / 1200 * vp.span)
        return True
    return False


# ============================================================ 타임 룰러 ===

class TimeRuler(QWidget):
    seekRequested = Signal(float)

    def __init__(self, viewport: Viewport, parent=None):
        super().__init__(parent)
        self.vp = viewport
        self.vp.changed.connect(self.update)
        self.duration = 0.0
        self._pos = 0.0
        self.setFixedHeight(30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_duration(self, d: float):
        self.duration = d
        self.update()

    def set_position(self, frac: float):
        if abs(self.vp.to_x(frac, self.width()) - self.vp.to_x(self._pos, self.width())) >= 0.5:
            self._pos = frac
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if self.duration <= 0:
            p.end()
            return
        t0 = self.vp.start * self.duration
        t1 = (self.vp.start + self.vp.span) * self.duration
        pps = w / max(1e-6, t1 - t0)
        step = next((c for c in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if c * pps >= 64), 600)
        p.setFont(theme.font(11, tabular=True))
        t = np.floor(t0 / step) * step
        while t <= t1 + 1e-6:
            x = (t - t0) * pps
            if t >= 0 and -1 <= x <= w + 1:
                p.setPen(QPen(QColor("#c3cbc0"), 1))
                p.drawLine(QPointF(x, h - 9), QPointF(x, h - 1))
                p.setPen(QColor(theme.TEXT_DIM))
                m, s = divmod(t, 60)
                txt = f"{int(m)}:{s:04.1f}" if step < 1 else f"{int(m)}:{int(s):02d}"
                p.drawText(QPointF(x + (3 if x < 8 else -p.fontMetrics().horizontalAdvance(txt) / 2),
                                   h - 13), txt)
                for k in range(1, 4):
                    xs = x + step * k / 4 * pps
                    if xs < w:
                        p.setPen(QPen(QColor("#e1e6df"), 1))
                        p.drawLine(QPointF(xs, h - 5), QPointF(xs, h - 1))
            t += step
        x = self.vp.to_x(self._pos, w)
        if -6 <= x <= w + 6:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.ACCENT))
            p.drawPolygon(QPolygonF([QPointF(x - 6, h - 12), QPointF(x + 6, h - 12), QPointF(x, h - 3)]))
            p.setPen(QPen(QColor(theme.ACCENT), 1.5))
            p.drawLine(QPointF(x, h - 4), QPointF(x, h))
        p.end()

    def _seek(self, x):
        if self.duration > 0:
            self.seekRequested.emit(max(0.0, min(1.0, self.vp.to_frac(x, self.width()))))

    def mousePressEvent(self, e):
        self._seek(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._seek(e.position().x())

    def wheelEvent(self, e):
        if not wheel_zoom(self.vp, e, self.width()):
            self.vp.set_zoom(self.vp.zoom * (1.25 if e.angleDelta().y() > 0 else 0.8),
                             self.vp.to_frac(e.position().x(), self.width()))
        e.accept()


# ============================================================ 트랙 카드 ===

class IconTile(QWidget):
    def __init__(self, icon_name: str, color: str, parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.color = QColor(color)
        self.setFixedSize(40, 40)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(self.color)
        c.setAlpha(40)
        p.setBrush(c)
        b = QColor(self.color)
        b.setAlpha(70)
        p.setPen(QPen(b, 1))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)
        p.drawPixmap(QPointF(9, 9), icons.pixmap(self.icon_name, self.color.name(), 22))
        p.end()


class TrackCard(QFrame):
    muteChanged = Signal(str, bool)
    soloChanged = Signal(str, bool)
    gainChanged = Signal(str, float)
    selectedChanged = Signal(str, bool)

    def __init__(self, key: str, number: int, name: str, sub: str, color: str, icon_name: str,
                 reference: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("TrackCard")
        self.key = key
        self.color = QColor(color)
        self.reference = reference
        self.setFixedWidth(HEADER_WIDTH)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 14, 10)
        lay.setSpacing(10)

        num = QLabel(str(number))
        num.setObjectName("TrackNum")
        num.setFixedWidth(14)
        num.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        num.setContentsMargins(0, 10, 0, 0)
        lay.addWidget(num)
        lay.addWidget(IconTile(icon_name, color), 0, Qt.AlignmentFlag.AlignTop)

        col = QVBoxLayout()
        col.setSpacing(6)
        top = QHBoxLayout()
        names = QVBoxLayout()
        names.setSpacing(0)
        self.name_lbl = QLabel(name)
        self.name_lbl.setObjectName("TrackName")
        sub_lbl = QLabel(sub)
        sub_lbl.setObjectName("TrackSub")
        names.addWidget(self.name_lbl)
        names.addWidget(sub_lbl)
        top.addLayout(names, 1)
        self.toggle = ToggleSwitch()
        self.toggle.setChecked(not reference)
        if reference:
            self.toggle.setEnabled(False)
            self.toggle.setToolTip("원본은 저장 대상이 아닙니다")
        else:
            self.toggle.setToolTip("켜면 '선택한 트랙 저장'에 포함됩니다")
        self.toggle.toggled.connect(lambda v: self.selectedChanged.emit(self.key, v))
        top.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(top)

        ctl = QHBoxLayout()
        ctl.setSpacing(6)
        self.mute_btn = QPushButton("M")
        self.mute_btn.setObjectName("MuteBtn")
        self.solo_btn = QPushButton("S")
        self.solo_btn.setObjectName("SoloBtn")
        for b, tip in ((self.mute_btn, "음소거 (Mute)"), (self.solo_btn, "솔로 (Solo) — 이 트랙만 듣기")):
            b.setCheckable(True)
            b.setFixedSize(28, 26)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mute_btn.toggled.connect(lambda v: self.muteChanged.emit(self.key, v))
        self.solo_btn.toggled.connect(lambda v: self.soloChanged.emit(self.key, v))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 150)
        self.slider.setValue(100)
        self.slider.setToolTip("볼륨 (더블클릭: 100%)")
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.mouseDoubleClickEvent = lambda e: self.slider.setValue(100)  # type: ignore
        c = self.color.name()
        self.slider.setStyleSheet(
            f"QSlider::sub-page:horizontal {{ background: {c}; border-radius: 2px; }}"
            f"QSlider::handle:horizontal {{ background: #f6f7f4; border: 2px solid {c}; width: 10px;"
            f" margin: -5px 0; border-radius: 7px; }}")
        self.slider.valueChanged.connect(self._on_gain)
        self.gain_lbl = QLabel("100%")
        self.gain_lbl.setObjectName("Gain")
        self.gain_lbl.setFixedWidth(38)
        self.gain_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctl.addWidget(self.mute_btn)
        ctl.addWidget(self.solo_btn)
        ctl.addSpacing(4)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.gain_lbl)
        col.addLayout(ctl)
        lay.addLayout(col, 1)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color)
        p.drawRoundedRect(QRectF(1, 10, 3, self.height() - 20), 1.5, 1.5)
        p.end()

    def _on_gain(self, v: int):
        self.gain_lbl.setText(f"{v}%")
        self.gainChanged.emit(self.key, v / 100.0)

    @property
    def gain(self) -> float:
        return self.slider.value() / 100.0

    @property
    def selected(self) -> bool:
        return self.toggle.isChecked() and not self.reference


class TrackRow(QWidget):
    """트랙 카드 + 파형 리전 한 줄."""

    muteChanged = Signal(str, bool)
    soloChanged = Signal(str, bool)
    gainChanged = Signal(str, float)
    selectedChanged = Signal(str, bool)
    seekRequested = Signal(float)

    def __init__(self, viewport: Viewport, key: str, number: int, name: str, sub: str, color: str,
                 icon_name: str, peaks: Optional[np.ndarray], reference: bool = False, parent=None):
        super().__init__(parent)
        self.key = key
        self.reference = reference
        self.setFixedHeight(ROW_HEIGHT)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(COLUMN_GAP + 20)
        self.card = TrackCard(key, number, name, sub, color, icon_name, reference)
        self.wave = WaveformView(viewport, color, name, sub)
        self.wave.set_peaks(peaks)
        lay.addWidget(self.card)
        lay.addWidget(self.wave, 1)
        self.card.muteChanged.connect(self.muteChanged)
        self.card.soloChanged.connect(self.soloChanged)
        self.card.gainChanged.connect(self.gainChanged)
        self.card.selectedChanged.connect(self.selectedChanged)
        self.wave.seekRequested.connect(self.seekRequested)

    # 편의 접근자
    def set_muted(self, v: bool):
        self.card.mute_btn.setChecked(v)

    def set_solo(self, v: bool):
        self.card.solo_btn.setChecked(v)

    def set_audible(self, v: bool):
        self.wave.set_dimmed(not v)

    @property
    def gain(self) -> float:
        return self.card.gain

    @property
    def selected(self) -> bool:
        return self.card.selected

    @property
    def slider(self) -> QSlider:
        return self.card.slider


class TracksPanel(QWidget):
    """트랙 카드 열과 타임라인 열의 둥근 배경 패널을 그림 (자식 위젯은 투명)."""

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        left = QRectF(0.5, 0.5, HEADER_WIDTH + 20, h - 1)
        right = QRectF(left.right() + COLUMN_GAP, 0.5, self.width() - left.right() - COLUMN_GAP - 1, h - 1)
        for r in (left, right):
            p.setBrush(QColor(theme.TRACK_PANEL))
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawRoundedRect(r, 14, 14)
        p.end()


# ============================================================ 드롭 존 ===

class DropZone(QFrame):
    browseRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("upload", theme.ACCENT, 44))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setFixedSize(96, 96)
        ic.setStyleSheet(f"background:{theme.ACCENT_SOFT}; border:1px solid "
                         f"{theme.rgba(theme.ACCENT, .25)}; border-radius:48px;")
        t = QLabel("음악 파일을 여기로 끌어다 놓으세요")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet("font-size:22px; font-weight:700;")
        s = QLabel("MP3 · WAV · FLAC · M4A · AAC · OGG · AIFF · 동영상(MP4 · MOV)")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setProperty("role", "dim")
        btn = text_button("파일 선택", "folder", "primary", 46)
        btn.setFixedWidth(170)
        btn.clicked.connect(self.browseRequested)
        glow(btn, theme.ACCENT, 20, 70, 3)
        lay.addWidget(ic, 0, Qt.AlignmentFlag.AlignCenter)
        lay.addSpacing(4)
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addSpacing(10)
        lay.addWidget(btn, 0, Qt.AlignmentFlag.AlignCenter)

    def set_hover(self, v: bool):
        self._hover = v
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(QColor(theme.TRACK_PANEL))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRoundedRect(r, 16, 16)
        inner = r.adjusted(22, 22, -22, -22)
        pen = QPen(QColor(theme.ACCENT if self._hover else "#c9d0dc"), 1.6, Qt.PenStyle.DashLine)
        pen.setDashPattern([5, 5])
        p.setPen(pen)
        p.setBrush(QColor(52, 120, 246, 18) if self._hover else QColor(theme.SURFACE_2))
        p.drawRoundedRect(inner, 14, 14)
        p.end()
