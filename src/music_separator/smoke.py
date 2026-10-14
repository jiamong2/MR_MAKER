"""GUI 스모크 테스트 (--gui-smoke out.png).

실제 메인 윈도우를 띄워
  1) 홈 화면: 파일 입력 → 반주(MR) + 키 +2 → [음원 추출 시작] → 결과 확인
  2) 홈 화면: 같은 입력으로 보컬만 · MP3 (분리 결과 재사용)
  3) 고급 옵션: 트랙 재생 조작 → 전체 트랙 저장 → 가이드 보컬 MR 저장 → 키 변경 후 보컬 저장
을 자동으로 수행하고 두 화면을 PNG 로 저장한 뒤 종료합니다. (CI 에서 빌드된 실행 파일의 GUI 동작 확인용)
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
    def __init__(self, win, png: Path, timeout: float = 1500):
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
        self.wait_t = 0.0

    def _log(self, msg):
        log.info("smoke: %s", msg)
        self.report["steps"].append(f"{time.monotonic() - self.t0:6.1f}s {msg}")

    def start(self):
        src = self.work / "Smoke Test 노래.wav"
        audio_io.write_wav(src, make_test_signal(15.0, 48000) * 0.9, 24, 48000)
        self.src = src
        w = self.win
        w.show_home()
        w.home_set_file(src)
        w.home.set_target("mr")
        w.home.key.setValue(2)
        w.set_format("wav24")
        self._log(f"home: file={src.name} target=mr key=+2")
        w.start_home_job()
        self.step = "home1"
        self.timer.start()

    def _shot(self, name: str):
        self.win.repaint()
        QApplication.processEvents()
        p = self.png.with_name(self.png.stem + name + self.png.suffix)
        self.win.grab().save(str(p))
        return p

    def _finish(self, ok: bool, why: str = ""):
        self.timer.stop()
        self._log(f"finish ok={ok} {why}")
        self.win.show_advanced()
        QApplication.processEvents()
        self._shot("")
        self.report["ok"] = ok
        self.report["files"] = sorted(str(p.relative_to(self.work)) for p in self.work.rglob("*") if p.is_file())
        self.report["last_error"] = getattr(self.win, "last_error", None)
        job = getattr(self.win, "last_job", None)
        if job is not None:
            self.report["last_job_report"] = job.report
        self.png.with_suffix(".json").write_text(json.dumps(self.report, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
        QApplication.instance().exit(0 if ok else 1)

    def _outputs(self) -> list:
        return sorted((self.work / "out").rglob("*.*"))

    def _poll(self):
        w = self.win
        if time.monotonic() - self.t0 > self.timeout:
            return self._finish(False, f"timeout at step {self.step}")
        if w.busy():
            return
        if self.step == "home1":
            job = w.last_job
            if job is None:
                return self._finish(False, f"home job failed: {w.last_pipeline_error}")
            names = [Path(f).name for f in job.files]
            self._log(f"home1 done: {names}")
            rep = job.report
            out = rep["outputs"][0]
            if not (len(names) == 1 and "MR" in names[0] and "Key +2" in names[0]):
                return self._finish(False, f"unexpected home1 outputs {names}")
            if out["sample_rate"] != 48000 or out["format"] != "wav24":
                return self._finish(False, f"home1 rate/format wrong: {out}")
            self._shot("-home")
            w.home.set_target("vocals")
            w.home.key.setValue(0)
            w.set_format("mp3")
            self.prev_sep = w.last_sep
            w.start_home_job()
            self.step = "home2"
        elif self.step == "home2":
            job = w.last_job
            names = [Path(f).name for f in job.files]
            self._log(f"home2 done: {names} reused={job.reused_separation}")
            if not (len(names) == 1 and "보컬" in names[0] and names[0].endswith(".mp3")):
                return self._finish(False, f"unexpected home2 outputs {names}")
            if not job.reused_separation:
                return self._finish(False, "separation was not reused for the same input")
            out = job.report["outputs"][0]
            if out["sample_rate"] != 44100 or out["resampled_from"] is not None:
                return self._finish(False, f"vocals must stay at model rate: {out}")
            if "320kbps" not in out["verified"]:
                return self._finish(False, f"mp3 not 320kbps: {out['verified']}")
            # 고급 옵션 화면 (홈 결과가 불러와져 있어야 함)
            w.show_advanced()
            if w.result is None or [s.key for s in w.result.stems] != ["vocals", "instrumental"]:
                return self._finish(False, "advanced view did not receive the home result")
            w.rows["vocals"].set_solo(True)
            w.rows["instrumental"].slider.setValue(70)
            w.mixer.seek(int(w.mixer.length * 0.4))
            block, _ = w.mixer.render(4096)
            self._log(f"mixer render ok, peak={float(abs(block).max()):.3f}")
            w.rows["vocals"].set_solo(False)
            w.set_format("flac")
            w.export_all()
            self.step = "export_all"
        elif self.step == "export_all":
            w.guide_box.setChecked(True)
            w.guide_slider.setValue(25)
            w.set_format("wav24")
            w.export_mr()
            self.step = "export_guide"
        elif self.step == "export_guide":
            w.key_slider.setValue(-3)
            w._apply_key()
            self.wait_t = time.monotonic()
            self.step = "key"
        elif self.step == "key":
            if w.key_semitones != -3:
                if time.monotonic() - self.wait_t > 300:
                    return self._finish(False, "key shift timeout")
                return
            self._log(f"key -3 applied to {len(w.mixer.tracks)} tracks")
            w.export_vocals()
            self.step = "done_wait"
        elif self.step == "done_wait":
            files = self._outputs()
            self._log(f"exported {len(files)} files: {[f.name for f in files]}")
            self.step = "done"
            ok = (len(files) >= 6 and any("가이드 보컬 25%" in f.name for f in files)
                  and any("Key -3" in f.name and "보컬" in f.name for f in files)
                  and sum(1 for f in files if f.suffix == ".flac") == 2)
            return self._finish(ok, "" if ok else "missing exports")
