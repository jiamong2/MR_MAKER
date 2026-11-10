"""MR SEPARATOR for ui.clover (6-Stem 에디션) 소스 실행: python run_separator.py"""
import os

os.environ.setdefault("MUSIC_SEPARATOR_EDITION", "separator")

import runpy  # noqa: E402
from pathlib import Path  # noqa: E402

runpy.run_path(str(Path(__file__).with_name("run.py")), run_name="__main__")
