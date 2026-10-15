"""새 버전 확인과 업데이트 (GitHub Releases).

- 프로그램 시작 시 최신 릴리스를 조용히 확인하고, 새 버전이 있으면 알림만 띄운다(설치는 사용자가 선택).
- Windows 설치판: 같은 종류(cpu/cuda)의 설치 파일을 내려받아 조용히 설치한 뒤 다시 실행한다.
- macOS·포터블판: 릴리스 페이지를 연다.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import __version__

log = logging.getLogger(__name__)

REPO = "jiamong2/MR_MAKER"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"


@dataclass
class Asset:
    name: str
    url: str
    size: int


@dataclass
class UpdateInfo:
    version: str
    page_url: str
    assets: list = field(default_factory=list)   # 설치에 필요한 파일들 (setup.exe + 분할 .bin)

    @property
    def can_auto_install(self) -> bool:
        return bool(self.assets)


def version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3])


def is_newer(remote: str, local: str = __version__) -> bool:
    return version_tuple(remote) > version_tuple(local)


def installed_windows() -> bool:
    """Inno Setup 으로 설치된 실행 파일인지 (포터블 zip 은 제외)."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    return any(Path(sys.executable).parent.glob("unins*.exe"))


MSI_DIR_NAME = "MR MAKER (MSI)"     # packaging/windows/msi/MRMaker.wxs 의 INSTALLFOLDER 이름


def installed_msi() -> bool:
    """MSI 로 설치된 실행 파일인지."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    return Path(sys.executable).parent.name == MSI_DIR_NAME


def pick_assets(release: dict, version: str, variant: str, msi: bool = False) -> list:
    """현재 설치 종류에 맞는 Windows 설치 파일 선택 (setup.exe + 분할 .bin, 또는 .msi)."""
    if msi:
        # .msi (+ 2GB 를 넘는 CUDA 버전은 같은 폴더에 둘 외부 .cab 들)
        base = f"MRMaker-{version}-windows-x64-{variant}"
        out = [Asset(a["name"], a["browser_download_url"], int(a.get("size", 0)))
               for a in release.get("assets", [])
               if a.get("name") == f"{base}.msi" or (a.get("name", "").startswith(f"{base}-")
                                                     and a.get("name", "").endswith(".cab"))]
        if not any(x.name.endswith(".msi") for x in out):
            return []
        return sorted(out, key=lambda x: (not x.name.endswith(".msi"), x.name))
    prefix = f"MRMaker-{version}-windows-x64-{variant}-setup"
    out = []
    for a in release.get("assets", []):
        name = a.get("name", "")
        if name.startswith(prefix) and (name.endswith(".exe") or name.endswith(".bin")):
            out.append(Asset(name, a["browser_download_url"], int(a.get("size", 0))))
    if not any(a.name.endswith(".exe") for a in out):
        return []
    return sorted(out, key=lambda a: (not a.name.endswith(".exe"), a.name))


def parse_release(release: dict, variant: str, auto: bool, msi: bool = False) -> Optional[UpdateInfo]:
    if release.get("draft") or release.get("prerelease"):
        return None
    ver = str(release.get("tag_name", "")).lstrip("vV")
    if not ver or not is_newer(ver):
        return None
    assets = pick_assets(release, ver, variant, msi) if auto else []
    return UpdateInfo(ver, release.get("html_url") or RELEASES_PAGE, assets)


def _urlopen(url: str, timeout: float):
    from .models import _ssl_context

    req = urllib.request.Request(url, headers={"User-Agent": f"MRMaker/{__version__}",
                                               "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def check(timeout: float = 8.0) -> Optional[UpdateInfo]:
    """새 버전이 있으면 UpdateInfo, 없으면 None. 네트워크 오류는 예외."""
    from .engine import build_variant

    with _urlopen(API_LATEST, timeout) as r:
        release = json.loads(r.read().decode("utf-8"))
    msi = installed_msi()
    return parse_release(release, build_variant(), installed_windows() or msi, msi)


ProgressCb = Callable[[int, int], None]


def download(info: UpdateInfo, progress: Optional[ProgressCb] = None,
             cancel: Optional[threading.Event] = None) -> Path:
    """설치 파일들을 임시 폴더에 내려받고 setup.exe 경로를 돌려준다."""
    dest = Path(tempfile.mkdtemp(prefix="MRMaker-update-"))
    total = sum(a.size for a in info.assets) or 1
    done = 0
    for a in info.assets:
        path = dest / a.name
        got = 0
        with _urlopen(a.url, 30) as r, open(path, "wb") as f:
            while True:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("cancelled")
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if progress:
                    progress(done + got, total)
        if a.size and got != a.size:
            raise IOError(f"{a.name} 을(를) 끝까지 받지 못했습니다 ({got}/{a.size} bytes).")
        done += got
    return dest / info.assets[0].name


def launch_installer(setup: Path) -> None:
    """설치 프로그램을 조용히 실행(설치가 끝나면 프로그램을 다시 켬). 호출 후 앱은 바로 종료해야 한다."""
    if setup.suffix.lower() == ".msi":
        # 기본 UI(/qb, 진행 표시만) → 설치가 끝나면 MSI 가 프로그램을 다시 실행
        args = ["msiexec", "/i", str(setup), "/qb", "/norestart"]
    else:
        args = [str(setup), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/relaunch=1"]
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(args, close_fds=True, creationflags=flags, cwd=str(setup.parent))
