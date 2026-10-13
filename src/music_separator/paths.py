"""운영체제별 경로 관리.

- 리소스(번들 모델, 아이콘): PyInstaller 실행 파일이면 sys._MEIPASS, 소스 실행이면 프로젝트 폴더
- 사용자 데이터(다운로드한 모델, 로그): OS 표준 위치
- 캐시(분리된 트랙 임시 파일): OS 표준 캐시 위치
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from . import APP_ID, APP_NAME


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """번들에 포함된 읽기 전용 리소스 폴더."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # src/music_separator/paths.py -> 프로젝트 루트
    return Path(__file__).resolve().parents[2]


def assets_dir() -> Path:
    return resource_dir() / "assets"


def bundled_models_dir() -> Path:
    return resource_dir() / "models"


def user_data_dir() -> Path:
    override = os.environ.get("MUSIC_SEPARATOR_DATA_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_ID
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_ID
    base.mkdir(parents=True, exist_ok=True)
    return base


def user_models_dir() -> Path:
    p = user_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = user_data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_root() -> Path:
    override = os.environ.get("MUSIC_SEPARATOR_CACHE_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_ID / "cache"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / APP_NAME
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / APP_ID
    base.mkdir(parents=True, exist_ok=True)
    return base


def new_session_dir() -> Path:
    p = cache_root() / f"session-{os.getpid()}-{int(time.time() * 1000)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cleanup_stale_sessions(max_age_hours: float = 12.0) -> None:
    """이전 실행에서 남은 임시 폴더 정리 (Windows에서 열린 파일은 삭제 실패 가능 → 무시)."""
    now = time.time()
    try:
        for child in cache_root().glob("session-*"):
            try:
                pid = int(child.name.split("-")[1])
            except (IndexError, ValueError):
                pid = -1
            if pid == os.getpid():
                continue
            if now - child.stat().st_mtime > max_age_hours * 3600 or not _pid_alive(pid):
                shutil.rmtree(child, ignore_errors=True)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return True


def default_output_dir() -> Path:
    music = Path.home() / "Music"
    if not music.exists():
        music = Path.home()
    return music / APP_NAME


def open_in_file_manager(path: Path) -> None:
    import subprocess

    path = Path(path)
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
