"""화면 위젯 동작 (오프스크린)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_waveform_playhead_moves_in_small_steps(monkeypatch):
    """긴 곡·좁은 창처럼 한 번에 0.5px 미만씩 움직여도 재생 위치 세로선이 계속 다시 그려짐 (v1.3.5 버그)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from music_separator.ui import widgets

    QApplication.instance() or QApplication([])
    vp = widgets.Viewport()
    wave = widgets.WaveformView(vp, "#e8a33d", "보컬", "Vocals")
    wave.set_peaks(np.zeros((1000, 2), np.float32))
    ruler = widgets.TimeRuler(vp)
    for w in (wave, ruler):
        w.resize(730, 90 if w is wave else 30)
    calls = {"wave": 0, "ruler": 0}
    monkeypatch.setattr(widgets.WaveformView, "update", lambda self, *a: calls.__setitem__("wave", calls["wave"] + 1))
    monkeypatch.setattr(widgets.TimeRuler, "update", lambda self, *a: calls.__setitem__("ruler", calls["ruler"] + 1))
    f = 0.1
    for _ in range(300):                       # 145초 곡을 33ms 마다 갱신하며 10초 재생
        f += 0.033 / 145
        wave.set_position(f)
        ruler.set_position(f)
    assert calls["wave"] >= 40 and abs(calls["wave"] - calls["ruler"]) <= 2
    assert abs(vp.to_x(wave._pos, 730) - vp.to_x(f, 730)) < 0.5
