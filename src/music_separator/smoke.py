"""GUI 스모크 테스트 (--gui-smoke out.png).

실제 메인 윈도우를 띄워 파일 열기 → 분리 → 뮤트/솔로 → 내보내기(MR 포함)를 자동으로 수행하고
화면을 PNG 로 저장한 뒤 종료합니다. CI 에서 빌드된 실행 파일의 GUI 동작 확인용입니다.
기본은 초소형 테스트 모델(demucs_unittest)을 사용하며 MUSIC_SEPARATOR_FORCE_MODEL 로 바꿀 수 있습니다.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from . import audio_io
from .selftest import make_test_signal

log = logging.getLogger(__name__)


class GuiSmoke:
    def __init__(self, win, png: Path, timeout: float = 1200):
        self.win = win
        self.png = Path(png)
        self.timeout = timeout
        self.t0 = time.monotonic()
        self.step = "start"
        self.report: dict = {"steps": []}
        os.environ["MUSIC_SEPARATOR_NONINTERACTIVE"] = "1"
        os.environ.setdefault("MUSIC_SEPARATOR_FORCE_MODEL", "demucs_unittest")
        self.work = Path(tempfile.mkdtemp(prefix="ms-smoke-"))
        win.output_dir = self.work / "out"
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._poll)

    def _log(self, msg):
        log.info("smoke: %s", msg)
        self.report["steps"].append(f"{time.monotonic() - self.t0:6.1f}s {msg}")

    def start(self):
        src = self.work / "Smoke Test 노래.wav"
        audio_io.write_wav(src, make_test_signal(15.0), 16)
        self._log(f"open {src}")
        self.win.open_file(src)
        self.step = "loading"
        self.timer.start()

    def _finish(self, ok: bool, why: str = ""):
        self.timer.stop()
        self._log(f"finish ok={ok} {why}")
        self.win.repaint()
        QApplication.processEvents()
        self.win.grab().save(str(self.png))
        self.report["ok"] = ok
        self.report["files"] = sorted(str(p.relative_to(self.work)) for p in self.work.rglob("*") if p.is_file())
        self.report["last_error"] = getattr(self.win, "last_error", None)
        self.png.with_suffix(".json").write_text(json.dumps(self.report, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        QApplication.instance().exit(0 if ok else 1)

    def _poll(self):
        w = self.win
        if time.monotonic() - self.t0 > self.timeout:
            return self._finish(False, f"timeout at step {self.step}")
        if getattr(w, "last_error", None) and self.step != "done":
            return self._finish(False, f"error: {w.last_error}")
        if w.busy():
            return
        if self.step == "loading":
            if w.loaded is None:
                return self._finish(False, "load failed")
            self._log(f"loaded {w.loaded.seconds:.1f}s")
            mode = os.environ.get("MUSIC_SEPARATOR_SMOKE_MODE", "4")
            w.mode_ctl.set_current(mode)
            w._on_mode(mode, save=False)
            w.start_separation()
            self.step = "separating"
        elif self.step == "separating":
            if w.result is None:
                return self._finish(False, "separation failed")
            self._log(f"separated: {[s.key for s in w.result.stems]} on {w.result.device}")
            w.rows["drums"].set_muted(True)
            w.rows["vocals"].set_solo(True)
            w.rows["bass"].slider.setValue(70)
            w.mixer.seek(int(w.mixer.length * 0.4))
            block, _ = w.mixer.render(4096)
            self._log(f"mixer render ok, peak={float(abs(block).max()):.3f}")
            w.export_all()
            self.step = "export_all"
        elif self.step == "export_all":
            w.rows["vocals"].set_solo(False)
            w.chips["drums"].setChecked(True)  # 보컬+드럼 제외 MR
            w.fmt_combo.setCurrentIndex(w.fmt_combo.findData("mp3"))
            w.export_mr()
            self.step = "export_mr"
        elif self.step == "export_mr":
            files = list((self.work / "out").rglob("*.*"))
            self._log(f"exported {len(files)} files")
            self.step = "done"
            ok = len(files) >= 5
            return self._finish(ok, "" if ok else "missing exports")
