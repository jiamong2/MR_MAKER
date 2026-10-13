"""네잎클로버 로고 (벡터로 그려 어떤 크기에서도 선명)."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient, QTransform

LEAF_LIGHT = QColor("#4b7a59")
LEAF_MID = QColor("#30503f")
LEAF_DARK = QColor("#1c2a25")
STEM = QColor("#1f2d28")


def _heart(L: float, width: float = 0.90) -> QPainterPath:
    """끝이 원점을 향한 하트 잎 (+y 방향, 길이 L). 고전 하트 곡선을 사용해 부드러운 볼과 얕은 홈."""
    k = L / 29.0
    p = QPainterPath()
    n = 240
    for i in range(n + 1):
        t = math.pi + 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pt = QPointF(x * k * width, (y + 17) * k)
        if i == 0:
            p.moveTo(pt)
        else:
            p.lineTo(pt)
    p.closeSubpath()
    return p


def paint_clover(p: QPainter, rect: QRectF, shadow: bool = True) -> None:
    """rect 안에 클로버를 그림 (원본 이미지와 같은 배치: 대각선 4잎 + 아래로 휜 줄기)."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    side = min(rect.width(), rect.height())
    cx = rect.center().x() - side * 0.01
    cy = rect.top() + side * 0.47
    L = side * 0.52
    gap = side * 0.010
    # 전체에 왼쪽 위(밝음) → 오른쪽 아래(어두움) 그라데이션
    g = QLinearGradient(QPointF(cx - side * 0.45, cy - side * 0.45), QPointF(cx + side * 0.45, cy + side * 0.45))
    g.setColorAt(0.0, LEAF_LIGHT)
    g.setColorAt(0.5, LEAF_MID)
    g.setColorAt(1.0, LEAF_DARK)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    heart = _heart(L)
    for ang in (225, 315, 135, 45):
        rad = math.radians(ang)
        t = QTransform()
        t.translate(cx + math.cos(rad) * gap, cy + math.sin(rad) * gap)
        t.rotate(ang - 90)
        p.drawPath(t.map(heart))
    if shadow:   # 잎 위에 은은한 빛
        hl = QRadialGradient(QPointF(cx - side * 0.2, cy - side * 0.22), side * 0.42)
        hl.setColorAt(0.0, QColor(255, 255, 255, 28))
        hl.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(hl)
        for ang in (225, 315, 135, 45):
            rad = math.radians(ang)
            t = QTransform()
            t.translate(cx + math.cos(rad) * gap, cy + math.sin(rad) * gap)
            t.rotate(ang - 90)
            p.drawPath(t.map(heart))
    # 줄기: 가운데에서 아래로, 끝이 살짝 오른쪽으로 휨
    stem = QPainterPath(QPointF(cx, cy + side * 0.01))
    stem.cubicTo(QPointF(cx - side * 0.005, cy + side * 0.20), QPointF(cx + side * 0.01, cy + side * 0.36),
                 QPointF(cx + side * 0.085, cy + side * 0.50))
    pen = QPen(STEM, max(1.0, side * 0.026))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(stem)
    p.restore()


def paint_app_icon(p: QPainter, size: int) -> None:
    """앱 아이콘: 흰색 둥근 사각형 위 클로버."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    m = size * 0.09
    r = QRectF(m, m, size - 2 * m, size - 2 * m)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0, QColor("#ffffff"))
    g.setColorAt(1, QColor("#eef3f0"))
    path = QPainterPath()
    path.addRoundedRect(r, size * 0.2, size * 0.2)
    p.fillPath(path, g)
    p.setPen(QPen(QColor(20, 50, 35, 40), max(1.0, size * 0.006)))
    p.drawPath(path)
    paint_clover(p, r.adjusted(r.width() * 0.14, r.height() * 0.12, -r.width() * 0.14, -r.height() * 0.16))
    p.restore()
