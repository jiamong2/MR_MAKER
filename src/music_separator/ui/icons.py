"""선형(stroke) SVG 아이콘 세트. 색상은 렌더링할 때 지정합니다.

외부 아이콘 파일 없이 코드에 포함 (24×24 viewBox, stroke-width 2 기준).
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_STROKE = ('fill="none" stroke="{c}" stroke-width="{w}" stroke-linecap="round" '
           'stroke-linejoin="round"')

# 각 값은 <svg> 내부 요소. {S} = stroke 속성, {c} = 색상
_ICONS: dict[str, str] = {
    "mic": '<rect x="9" y="2.5" width="6" height="12" rx="3" {S}/>'
           '<path d="M5.5 10.5v1a6.5 6.5 0 0 0 13 0v-1" {S}/><path d="M12 18v3.5M8.5 21.5h7" {S}/>',
    "mic-off": '<path d="M3 3l18 18" {S}/><path d="M9 9v2.5a3 3 0 0 0 5.1 2.1M15 9.3V5.5a3 3 0 0 0-5.9-.8" {S}/>'
               '<path d="M18.5 10.5v1a6.4 6.4 0 0 1-.8 3.1M5.5 10.5v1a6.5 6.5 0 0 0 10.4 5.2" {S}/>'
               '<path d="M12 18v3.5M8.5 21.5h7" {S}/>',
    "drum": '<ellipse cx="12" cy="10" rx="8.5" ry="3.2" {S}/>'
            '<path d="M3.5 10v6.2c0 1.8 3.8 3.3 8.5 3.3s8.5-1.5 8.5-3.3V10" {S}/>'
            '<path d="M7 12.8v6M12 13.4v6.1M17 12.8v6" {S}/><path d="M8 2.5l4 6.2M16.5 3l-3.3 5.5" {S}/>',
    "guitar": '<path d="M14.5 9.5l5.8-5.8M18.6 2.6l2.8 2.8" {S}/>'
              '<path d="M11.3 8.8c1.6-.9 3.5-.6 4.4.6.9 1.2.5 2.9-.7 3.9.5 1.9-.2 4.3-2.1 6.1-2.6 2.5-6.3 2.9-8.2 1-1.9-1.9-1.5-5.6 1-8.2 1.8-1.9 4.2-2.6 6.1-2.1" {S}/>'
              '<circle cx="9.3" cy="14.7" r="1.6" {S}/><path d="M6.5 15.5l2 2" {S}/>',
    "piano": '<rect x="3" y="4" width="18" height="16" rx="2.5" {S}/>'
             '<path d="M8 20v-5.5M12 20v-5.5M16 20v-5.5" {S}/>'
             '<path d="M6.5 4v8.5h3V4M10.5 4v8.5h3V4" {S}/><path d="M14.5 4v8.5h3V4" {S}/>',
    "wave": '<path d="M3 10v4M7 6v12M11 3v18M15 7v10M19 10v4" {S}/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z" {S}/><path d="M3 13l9 5 9-5" {S}/><path d="M3 17.5l9 5 9-5" {S}/>',
    "folder": '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.8l2 2.2h7.2a2 2 0 0 1 2 2v7.8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" {S}/>',
    "folder-fill": '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.8l2 2.2h7.2a2 2 0 0 1 2 2v7.8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" fill="{c}" stroke="none"/>',
    "settings": '<circle cx="12" cy="12" r="3.2" {S}/>'
                '<path d="M12 2.8v2.4M12 18.8v2.4M2.8 12h2.4M18.8 12h2.4M5.5 5.5l1.7 1.7M16.8 16.8l1.7 1.7M5.5 18.5l1.7-1.7M16.8 7.2l1.7-1.7" {S}/>'
                '<circle cx="12" cy="12" r="6.6" {S}/>',
    "download": '<path d="M12 3.5v11.5M7.5 10.5l4.5 4.5 4.5-4.5" {S}/><path d="M4 16.5v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" {S}/>',
    "save-multi": '<rect x="7.5" y="7.5" width="13" height="13" rx="2.5" {S}/><path d="M16.5 4.5v-.2a1.8 1.8 0 0 0-1.8-1.8H5.3a1.8 1.8 0 0 0-1.8 1.8v9.4a1.8 1.8 0 0 0 1.8 1.8h.2" {S}/>'
                  '<path d="M14 11v6M11.5 14.5l2.5 2.5 2.5-2.5" {S}/>',
    "play": '<path d="M7.5 4.8v14.4a1 1 0 0 0 1.5.9l11.2-7.2a1 1 0 0 0 0-1.7L9 4a1 1 0 0 0-1.5.8z" fill="{c}" stroke="none"/>',
    "pause": '<rect x="6" y="4.5" width="4.2" height="15" rx="1.3" fill="{c}"/><rect x="13.8" y="4.5" width="4.2" height="15" rx="1.3" fill="{c}"/>',
    "stop": '<rect x="5.5" y="5.5" width="13" height="13" rx="2.2" fill="{c}"/>',
    "rewind": '<rect x="5" y="5" width="2.6" height="14" rx="1" fill="{c}"/><path d="M19 5.9v12.2a.9.9 0 0 1-1.4.8L9.6 12.8a.9.9 0 0 1 0-1.6l8-6.1a.9.9 0 0 1 1.4.8z" fill="{c}"/>',
    "volume": '<path d="M4 9.5h3.2L12 5.5v13l-4.8-4H4z" {S}/><path d="M15.5 9a4.2 4.2 0 0 1 0 6M18.3 6.3a8 8 0 0 1 0 11.4" {S}/>',
    "dots": '<circle cx="5.5" cy="12" r="1.6" fill="{c}"/><circle cx="12" cy="12" r="1.6" fill="{c}"/><circle cx="18.5" cy="12" r="1.6" fill="{c}"/>',
    "zoom": '<circle cx="10.5" cy="10.5" r="6.5" {S}/><path d="M15.5 15.5l5 5" {S}/>',
    "minus": '<path d="M5 12h14" {S}/>',
    "plus": '<path d="M12 5v14M5 12h14" {S}/>',
    "expand": '<path d="M14.5 3.5h6v6M9.5 20.5h-6v-6M20.5 3.5l-7 7M3.5 20.5l7-7" {S}/>',
    "sparkles": '<path d="M10 3l1.6 4.6L16 9.2l-4.4 1.6L10 15.5l-1.6-4.7L4 9.2l4.4-1.6z" fill="{c}" stroke="none"/>'
                '<path d="M18 13.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z" fill="{c}" stroke="none"/>'
                '<path d="M17.5 2.5l.6 1.5 1.5.6-1.5.6-.6 1.5-.6-1.5-1.5-.6 1.5-.6z" fill="{c}" stroke="none"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2" {S}/><rect x="9.5" y="9.5" width="5" height="5" rx="1" {S}/>'
           '<path d="M9.5 2.5V6M14.5 2.5V6M9.5 18v3.5M14.5 18v3.5M2.5 9.5H6M2.5 14.5H6M18 9.5h3.5M18 14.5h3.5" {S}/>',
    "music": '<path d="M9 18V5.5l11-2v12.5" {S}/><circle cx="6.5" cy="18" r="2.5" {S}/><circle cx="17.5" cy="16" r="2.5" {S}/>',
    "check-circle": '<circle cx="12" cy="12" r="9.5" fill="{c}" stroke="none"/><path d="M7.8 12.3l2.8 2.8 5.6-5.8" fill="none" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>',
    "alert": '<circle cx="12" cy="12" r="9.5" fill="{c}" stroke="none"/><path d="M12 7.2v5.6M12 16.4v.2" fill="none" stroke="#ffffff" stroke-width="2.6" stroke-linecap="round"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18" {S}/>',
    "chevron-down": '<path d="M6.5 9.5l5.5 5.5 5.5-5.5" {S}/>',
    "upload": '<path d="M12 15.5V4M7.5 8.5L12 4l4.5 4.5" {S}/><path d="M4 15.5v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" {S}/>',
}

STEM_ICON = {
    "original": "wave", "vocals": "mic", "drums": "drum", "bass": "guitar", "guitar": "guitar",
    "piano": "piano", "other": "wave", "instrumental": "music",
}


def svg(name: str, color: str = "#e8ecf5", width: float = 2.0) -> bytes:
    body = _ICONS[name].replace("{S}", _STROKE).replace("{c}", color).replace("{w}", str(width))
    body = body.replace("{c}", color).replace("{w}", str(width))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>').encode()


@lru_cache(maxsize=512)
def pixmap(name: str, color: str = "#e8ecf5", size: int = 20, dpr: float = 2.0, width: float = 2.0) -> QPixmap:
    px = int(size * dpr)
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    r = QSvgRenderer(QByteArray(svg(name, color, width)))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    r.render(p, QRectF(0, 0, px, px))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, color: str = "#e8ecf5", size: int = 20, width: float = 2.0) -> QIcon:
    ic = QIcon()
    ic.addPixmap(pixmap(name, color, size, 2.0, width))
    return ic


def qsize(n: int) -> QSize:
    return QSize(n, n)
