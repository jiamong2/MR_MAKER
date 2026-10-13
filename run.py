"""Music Separator 실행 진입점 (소스 실행 및 PyInstaller 빌드 공용)."""
import os
import sys

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from music_separator.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
