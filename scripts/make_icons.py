"""앱 아이콘(icon.png / icon.ico / icon.icns)과 체크 표시 이미지를 생성합니다.

사용: python scripts/make_icons.py      (PySide6, Pillow 필요)
아이콘을 바꾸고 싶으면 assets/icon.png(1024x1024)를 교체한 뒤 --from-png 옵션으로 실행하세요.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def draw_icon(size: int = 1024) -> QImage:
    """앱 아이콘: 흰색 둥근 사각형 + 네잎클로버 (ui/clover.py 의 벡터 그림)."""
    sys.path.insert(0, str(ASSETS.parent / "src"))
    from music_separator.ui.clover import paint_app_icon

    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    paint_app_icon(p, size)
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

    pixmap("chevron-down", "#8a93a5", 28, 1.0, 2.2).save(str(ASSETS / "chevron.png"))
    from PIL import Image

    im = Image.open(png).convert("RGBA")
    im.save(ASSETS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    im.save(ASSETS / "icon.icns", sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512),
                                         (1024, 1024)])
    print("written:", *sorted(p.name for p in ASSETS.iterdir()))


if __name__ == "__main__":
    main()
