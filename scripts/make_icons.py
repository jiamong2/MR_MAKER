"""앱 아이콘(icon.png / icon.ico / icon.icns)과 체크 표시 이미지를 생성합니다.

사용: python scripts/make_icons.py      (PySide6, Pillow 필요)
아이콘을 바꾸고 싶으면 assets/icon.png(1024x1024)를 교체한 뒤 --from-png 옵션으로 실행하세요.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"
COLORS = ["#F5A623", "#3FB8F0", "#B57BFF", "#5BD08A", "#FF7EB6"]


def draw_icon(size: int = 1024) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size
    # macOS 스타일 여백 + 둥근 사각형
    m = s * 0.09
    rect = QRectF(m, m, s - 2 * m, s - 2 * m)
    g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    g.setColorAt(0, QColor("#34343c"))
    g.setColorAt(1, QColor("#16161a"))
    path = QPainterPath()
    path.addRoundedRect(rect, s * 0.19, s * 0.19)
    p.fillPath(path, g)
    p.setPen(QPen(QColor(255, 255, 255, 30), s * 0.006))
    p.drawPath(path)

    # 왼쪽: 하나의 흰 파형 → 오른쪽: 색색의 트랙으로 갈라짐
    cy = s * 0.5
    left_x = s * 0.20
    split_x = s * 0.44
    right_x = s * 0.80
    import math

    wave = QPainterPath()
    n = 60
    for i in range(n + 1):
        x = left_x + (split_x - left_x) * i / n
        amp = s * 0.13 * math.sin(math.pi * i / n) * (0.6 + 0.4 * math.sin(i * 0.9))
        y = cy - amp
        (wave.moveTo if i == 0 else wave.lineTo)(QPointF(x, y))
    for i in range(n, -1, -1):
        x = left_x + (split_x - left_x) * i / n
        amp = s * 0.13 * math.sin(math.pi * i / n) * (0.6 + 0.4 * math.sin(i * 0.9))
        wave.lineTo(QPointF(x, cy + amp))
    p.fillPath(wave, QColor("#e9e9ee"))

    lanes = len(COLORS)
    gap = s * 0.115
    for k, col in enumerate(COLORS):
        y = cy + (k - (lanes - 1) / 2) * gap
        pen = QPen(QColor(col), s * 0.05)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        curve = QPainterPath(QPointF(split_x - s * 0.02, cy))
        curve.cubicTo(QPointF(split_x + s * 0.10, cy), QPointF(split_x + s * 0.06, y),
                      QPointF(split_x + s * 0.17, y))
        p.drawPath(curve)
        # 각 트랙의 미니 파형 막대
        x0 = split_x + s * 0.235
        bars = 6
        bpen = QPen(QColor(col), s * 0.022)
        bpen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(bpen)
        for b in range(bars):
            x = x0 + b * (right_x - x0) / (bars - 1)
            h = s * (0.008 + 0.030 * ((b * 7 + k * 3) % 5) / 4)
            p.drawLine(QPointF(x, y - h), QPointF(x, y + h))
    p.end()
    return img


def draw_check(size: int = 30) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("white"), size * 0.13)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    path = QPainterPath(QPointF(size * 0.24, size * 0.52))
    path.lineTo(QPointF(size * 0.43, size * 0.70))
    path.lineTo(QPointF(size * 0.77, size * 0.32))
    p.drawPath(path)
    p.end()
    return img


def main():
    _app = QGuiApplication(sys.argv[:1])  # noqa: F841
    ASSETS.mkdir(exist_ok=True)
    png = ASSETS / "icon.png"
    if "--from-png" not in sys.argv:
        draw_icon(1024).save(str(png))
    draw_check(30).save(str(ASSETS / "check.png"))
    sys.path.insert(0, str(ASSETS.parent / "src"))
    from music_separator.ui.icons import pixmap  # noqa: E402

    pixmap("chevron-down", "#8f99b2", 28, 1.0, 2.2).save(str(ASSETS / "chevron.png"))
    from PIL import Image

    im = Image.open(png).convert("RGBA")
    im.save(ASSETS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    im.save(ASSETS / "icon.icns", sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512),
                                         (1024, 1024)])
    print("written:", *sorted(p.name for p in ASSETS.iterdir()))


if __name__ == "__main__":
    main()
