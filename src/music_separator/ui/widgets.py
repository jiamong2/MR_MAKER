"""재사용 위젯: 파형 리전, 타임 룰러, 트랙 행, 드롭존, 세그먼트 컨트롤."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)

from . import theme

HEADER_WIDTH = 252
ROW_HEIGHT = 86


def _resample_peaks(peaks: np.ndarray, width: int) -> np.ndarray:
    n = len(peaks)
    width = max(1, width)
    if n == 0:
        return np.zeros((width, 2), np.float32)
    if n <= width:
        idx = np.linspace(0, n - 1, width).round().astype(int)
        return peaks[idx]
    edges = np.linspace(0, n, width + 1).astype(int)
    edges = np.minimum(edges, n - 1)
    mins = np.minimum.reduceat(peaks[:, 0], edges[:-1])
    maxs = np.maximum.reduceat(peaks[:, 1], edges[:-1])
    return np.stack([mins, maxs], 1)


class WaveformView(QWidget):
    """Logic Pro 의 리전처럼 색이 칠해진 파형. 클릭/드래그로 재생 위치 이동."""

    seekRequested = Signal(float)  # 0..1

    def __init__(self, color: str = "#8fa3b0", label: str = "", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._peaks: Optional[np.ndarray] = None
        self._color = QColor(color)
        self._label = label
        self._dim = False
        self._pos = 0.0
        self._cache: Optional[QPixmap] = None
        self._placeholder = ""

    def set_peaks(self, peaks: Optional[np.ndarray]) -> None:
        self._peaks = peaks
        self._cache = None
        self.update()

    def set_dimmed(self, dim: bool) -> None:
        if dim != self._dim:
            self._dim = dim
            self._cache = None
            self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self._cache = None
        self.update()

    def set_position(self, frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        if abs(frac - self._pos) * self.width() >= 0.5:
            self._pos = frac
            self.update()

    def resizeEvent(self, e):
        self._cache = None
        super().resizeEvent(e)

    def _render_cache(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        w, h = self.width(), self.height()
        pm = QPixmap(int(w * dpr), int(h * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(1.5, 3.5, w - 3, h - 7)
        base = QColor(self._color)
        if self._dim:
            base = QColor(base.red() // 3 + 40, base.green() // 3 + 40, base.blue() // 3 + 40)
        if self._peaks is None:
            p.setPen(QPen(QColor(theme.BORDER), 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(255, 255, 255, 6))
            p.drawRoundedRect(rect, 6, 6)
            if self._placeholder:
                p.setPen(QColor(theme.TEXT_FAINT))
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder)
            p.end()
            return pm
        fill = QColor(base)
        fill.setAlpha(60 if not self._dim else 35)
        border = QColor(base)
        border.setAlpha(170)
        p.setPen(QPen(border, 1))
        p.setBrush(fill)
        p.drawRoundedRect(rect, 6, 6)
        # 상단 라벨 스트립
        strip = QRectF(rect.left(), rect.top(), rect.width(), 17)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 6, 6)
        p.save()
        p.setClipPath(clip)
        sc = QColor(base)
        sc.setAlpha(150 if not self._dim else 70)
        p.fillRect(strip, sc)
        p.restore()
        p.setPen(QColor(20, 20, 22) if not self._dim else QColor(theme.TEXT_FAINT))
        f = QFont(self.font())
        f.setPointSizeF(max(8.0, f.pointSizeF() - 1.5))
        f.setBold(True)
        p.setFont(f)
        p.drawText(strip.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self._label)
        # 파형
        area = rect.adjusted(2, 19, -2, -3)
        cols = max(2, int(area.width()))
        pk = _resample_peaks(self._peaks, cols)
        peak = float(np.max(np.abs(self._peaks))) if len(self._peaks) else 0.0
        gain = 0.95 / max(peak, 0.95 / 3.0) if peak > 1e-6 else 1.0  # 조용한 트랙은 최대 3배 확대
        mid = area.center().y()
        half = area.height() / 2
        top = [QPointF(area.left() + i, mid - float(pk[i, 1]) * gain * half) for i in range(cols)]
        bot = [QPointF(area.left() + i, mid - float(pk[i, 0]) * gain * half) for i in reversed(range(cols))]
        wave = QColor(base).lighter(135 if not self._dim else 100)
        wave.setAlpha(235 if not self._dim else 120)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(wave)
        p.drawPolygon(QPolygonF(top + bot))
        p.setPen(QPen(QColor(255, 255, 255, 25), 1))
        p.drawLine(QPointF(area.left(), mid), QPointF(area.right(), mid))
        p.end()
        return pm

    def paintEvent(self, e):
        if self._cache is None or self._cache.size() != QSize(int(self.width() * self.devicePixelRatioF()),
                                                              int(self.height() * self.devicePixelRatioF())):
            self._cache = self._render_cache()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        if self._peaks is not None:
            x = 1.5 + self._pos * (self.width() - 3)
            p.setPen(QPen(QColor(255, 255, 255, 230), 1.4))
            p.drawLine(QPointF(x, 2), QPointF(x, self.height() - 2))
        p.end()

    def _seek_from(self, x: float) -> None:
        if self._peaks is None:
            return
        frac = (x - 1.5) / max(1.0, self.width() - 3)
        self.seekRequested.emit(max(0.0, min(1.0, frac)))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._seek_from(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._seek_from(e.position().x())


class TimeRuler(QWidget):
    seekRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.duration = 0.0
        self._pos = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_duration(self, d: float) -> None:
        self.duration = d
        self.update()

    def set_position(self, frac: float) -> None:
        if abs(frac - self._pos) * self.width() >= 0.5:
            self._pos = frac
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#1e1e21"))
        p.setPen(QColor("#111"))
        p.drawLine(0, h - 1, w, h - 1)
        if self.duration <= 0:
            p.end()
            return
        usable = w - 3
        px_per_sec = usable / self.duration
        step = 1
        for cand in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
            if cand * px_per_sec >= 70:
                step = cand
                break
        else:
            step = 1200
        f = QFont(self.font())
        f.setPointSizeF(max(7.5, f.pointSizeF() - 2.5))
        p.setFont(f)
        t = 0
        while t <= self.duration + 1e-6:
            x = 1.5 + t * px_per_sec
            p.setPen(QColor("#5a5a62"))
            p.drawLine(QPointF(x, h - 9), QPointF(x, h - 1))
            p.setPen(QColor(theme.TEXT_DIM))
            m, s = divmod(int(t), 60)
            p.drawText(QPointF(x + 3, h - 11), f"{m}:{s:02d}")
            sub = step / 4 if step >= 4 else 0
            if sub:
                for k in range(1, 4):
                    xs = x + sub * k * px_per_sec
                    if xs < w:
                        p.setPen(QColor("#3e3e44"))
                        p.drawLine(QPointF(xs, h - 5), QPointF(xs, h - 1))
            t += step
        x = 1.5 + self._pos * usable
        tri = QPolygonF([QPointF(x - 6, 2), QPointF(x + 6, 2), QPointF(x, 12)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#f2f2f2"))
        p.drawPolygon(tri)
        p.setPen(QPen(QColor("#f2f2f2"), 1.2))
        p.drawLine(QPointF(x, 10), QPointF(x, h))
        p.end()

    def _seek(self, x):
        if self.duration > 0:
            self.seekRequested.emit(max(0.0, min(1.0, (x - 1.5) / max(1, self.width() - 3))))

    def mousePressEvent(self, e):
        self._seek(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._seek(e.position().x())


class TrackRow(QWidget):
    muteChanged = Signal(str, bool)
    soloChanged = Signal(str, bool)
    gainChanged = Signal(str, float)
    selectedChanged = Signal(str, bool)
    seekRequested = Signal(float)

    def __init__(self, key: str, name: str, sub: str, color: str, peaks: Optional[np.ndarray],
                 reference: bool = False, parent=None):
        super().__init__(parent)
        self.key = key
        self.reference = reference
        self.setFixedHeight(ROW_HEIGHT)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QFrame()
        head.setObjectName("TrackHeader")
        head.setFixedWidth(HEADER_WIDTH)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 10, 0)
        hl.setSpacing(8)
        bar = QFrame()
        bar.setFixedWidth(5)
        bar.setStyleSheet(f"background:{color}; border:none;")
        hl.addWidget(bar)

        col = QVBoxLayout()
        col.setContentsMargins(4, 8, 0, 8)
        col.setSpacing(5)
        top = QHBoxLayout()
        top.setSpacing(6)
        names = QVBoxLayout()
        names.setSpacing(0)
        self.name_lbl = QLabel(name)
        self.name_lbl.setObjectName("TrackName")
        sub_lbl = QLabel(sub)
        sub_lbl.setObjectName("TrackSub")
        names.addWidget(self.name_lbl)
        names.addWidget(sub_lbl)
        top.addLayout(names, 1)
        self.select_box = QCheckBox("저장")
        self.select_box.setToolTip("'선택한 트랙 저장' 버튼으로 저장할 트랙")
        self.select_box.setChecked(True)
        self.select_box.toggled.connect(lambda v: self.selectedChanged.emit(self.key, v))
        if reference:
            self.select_box.setChecked(False)
            self.select_box.hide()
        top.addWidget(self.select_box, 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(top)

        ctl = QHBoxLayout()
        ctl.setSpacing(5)
        self.mute_btn = QPushButton("M")
        self.mute_btn.setObjectName("MuteBtn")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedSize(26, 22)
        self.mute_btn.setToolTip("음소거 (Mute)")
        self.mute_btn.toggled.connect(self._on_mute)
        self.solo_btn = QPushButton("S")
        self.solo_btn.setObjectName("SoloBtn")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedSize(26, 22)
        self.solo_btn.setToolTip("솔로 (Solo) — 이 트랙만 듣기")
        self.solo_btn.toggled.connect(lambda v: self.soloChanged.emit(self.key, v))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 150)
        self.slider.setValue(100)
        self.slider.setToolTip("볼륨 (더블클릭: 100%)")
        self.slider.valueChanged.connect(self._on_gain)
        self.slider.mouseDoubleClickEvent = lambda e: self.slider.setValue(100)  # type: ignore
        self.gain_lbl = QLabel("100%")
        self.gain_lbl.setObjectName("Gain")
        self.gain_lbl.setFixedWidth(34)
        self.gain_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctl.addWidget(self.mute_btn)
        ctl.addWidget(self.solo_btn)
        ctl.addSpacing(2)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.gain_lbl)
        col.addLayout(ctl)
        hl.addLayout(col, 1)
        lay.addWidget(head)

        self.wave = WaveformView(color, f"{name}  ·  {sub}")
        self.wave.set_peaks(peaks)
        self.wave.seekRequested.connect(self.seekRequested)
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(4, 0, 6, 0)
        wl.addWidget(self.wave)
        lay.addWidget(wrap, 1)

    def _on_mute(self, v: bool):
        self.wave.set_dimmed(v)
        self.muteChanged.emit(self.key, v)

    def _on_gain(self, v: int):
        self.gain_lbl.setText(f"{v}%")
        self.gainChanged.emit(self.key, v / 100.0)

    def set_muted(self, v: bool):
        self.mute_btn.setChecked(v)

    def set_solo(self, v: bool):
        self.solo_btn.setChecked(v)

    def set_audible(self, audible: bool):
        """솔로 상태 등으로 실제로 들리지 않는 트랙을 흐리게 표시."""
        self.wave.set_dimmed(not audible)

    @property
    def gain(self) -> float:
        return self.slider.value() / 100.0

    @property
    def selected(self) -> bool:
        return self.select_box.isChecked() and not self.reference


class DropZone(QFrame):
    browseRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(10)
        self.icon = QLabel("♪")
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.setStyleSheet(f"font-size:54px; color:{theme.TEXT_FAINT};")
        t = QLabel("음악 파일을 여기로 끌어다 놓으세요")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet("font-size:20px; font-weight:600;")
        s = QLabel("MP3 · WAV · FLAC · M4A · AAC · OGG · AIFF · 동영상(MP4·MOV) 지원")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setObjectName("Hint")
        btn = QPushButton("파일 선택…")
        btn.setObjectName("Primary")
        btn.setFixedWidth(160)
        btn.clicked.connect(self.browseRequested)
        lay.addWidget(self.icon)
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addSpacing(6)
        lay.addWidget(btn, 0, Qt.AlignmentFlag.AlignCenter)

    def set_hover(self, v: bool):
        self._hover = v
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(24, 24, -24, -24)
        pen = QPen(QColor(theme.ACCENT if self._hover else "#4a4a52"), 2, Qt.PenStyle.DashLine)
        pen.setDashPattern([6, 5])
        p.setPen(pen)
        p.setBrush(QColor(61, 139, 253, 28) if self._hover else QColor(255, 255, 255, 5))
        p.drawRoundedRect(r, 16, 16)
        p.end()


class SegmentedControl(QWidget):
    changed = Signal(str)

    def __init__(self, items: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for i, (key, text) in enumerate(items):
            b = QPushButton(text)
            b.setObjectName("Segment")
            b.setCheckable(True)
            if i == 0:
                b.setProperty("pos", "left")
            elif i == len(items) - 1:
                b.setProperty("pos", "right")
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.group.addButton(b)
            self.buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            lay.addWidget(b)

    def set_current(self, key: str):
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def current(self) -> str:
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return next(iter(self.buttons))
