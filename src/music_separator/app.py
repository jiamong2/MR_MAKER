"""애플리케이션 진입점."""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import APP_ID, APP_NAME, ORG_NAME, __version__, paths  # noqa: E402

log = logging.getLogger("music_separator")


def setup_logging(verbose: bool = False) -> Path:
    log_file = paths.logs_dir() / "music_separator.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    if sys.stderr is not None:  # Windows 창 모드 실행 파일은 stderr 가 없음
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)

    def excepthook(t, v, tb):
        log.critical("unhandled exception", exc_info=(t, v, tb))
    sys.excepthook = excepthook
    return log_file


def parse_args(argv):
    p = argparse.ArgumentParser(prog="MRMaker", description=f"{APP_NAME} {__version__}")
    p.add_argument("file", nargs="?", help="시작할 때 열 오디오 파일")
    p.add_argument("--selftest", action="store_true", help="헤드리스 자가 진단 실행 후 종료")
    p.add_argument("--report", help="자가 진단 결과 JSON 경로")
    p.add_argument("--model", help="자가 진단에 사용할 모델 (htdemucs, htdemucs_6s, demucs_unittest …)")
    p.add_argument("--mode", default="2", choices=["2", "4", "6"])
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--engine", default="auto", choices=["auto", "demucs", "roformer"])
    p.add_argument("--preset", default="quality", choices=["speed", "quality", "best"])
    p.add_argument("--youtube-test", metavar="URL", help="자가 진단에서 실제 YouTube 오디오 받기 시험 (선택)")
    p.add_argument("--seconds", type=float, default=12.0, help="자가 진단 테스트 신호 길이")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--gui-smoke", metavar="PNG", help="GUI 를 띄워 분리까지 실행하고 스크린샷 저장 후 종료")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    # macOS 가 붙이는 -psn_ 인자 무시
    argv = [a for a in argv if not a.startswith("-psn_")]
    return p.parse_args(argv)


def main(argv=None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    setup_logging(args.verbose)
    log.info("%s %s starting (%s)", APP_NAME, __version__, sys.platform)

    if args.selftest:
        from .selftest import run

        return run(model=args.model, allow_download=not args.no_download, report=args.report,
                   device_pref=args.device, mode=args.mode, engine_key=args.engine, seconds=args.seconds,
                   preset=args.preset, youtube_url=args.youtube_test)

    from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_ID)

    class App(QApplication):
        def __init__(self, a):
            super().__init__(a)
            self.window = None
            self.pending: list[str] = []

        def event(self, e):  # macOS Finder 의 "다음으로 열기" / Dock 드롭
            if e.type() == QEvent.Type.FileOpen:
                f = e.file()
                if self.window is not None:
                    self.window.open_file(Path(f))
                else:
                    self.pending.append(f)
                return True
            return super().event(e)

    app = App(sys.argv[:1])
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    icon = paths.assets_dir() / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    from .ui import theme
    from .ui.main_window import MainWindow

    theme.apply(app)
    theme.finalize_stylesheet(app)

    paths.cleanup_stale_sessions()
    win = MainWindow(QSettings())
    app.window = win
    win.show()

    if args.gui_smoke:
        from .smoke import GuiSmoke

        smoke = GuiSmoke(win, Path(args.gui_smoke))
        QTimer.singleShot(300, smoke.start)
    else:
        start = args.file or (app.pending[0] if app.pending else None)
        if start:
            QTimer.singleShot(200, lambda: win.open_file(Path(start)))
    return app.exec()
