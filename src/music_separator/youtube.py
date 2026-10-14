"""YouTube 오디오 획득 (yt-dlp).

음질 원칙
- 영상에 제공되는 '오디오 전용' 스트림 중에서 고르고, 영상 데이터는 받지 않음
  (오디오 전용 스트림이 없을 때만 해상도가 가장 낮은 영상+오디오 스트림 사용)
- 비트레이트 숫자로 서로 다른 코덱을 비교하지 않음: 코덱 등급(무손실 > Opus > Vorbis > AAC-LC > MP3 >
  HE-AAC)을 먼저 비교하고, 같은 코덱 안에서만 비트레이트·샘플레이트로 비교
- 음량을 압축한 DRC 스트림(예: 251-drc), DRM 스트림, 손상 표시 스트림, 해설(descriptive) 음성은 제외
- 원래 언어(original) 오디오 트랙 우선, 스테레오 이하 채널 우선 (AI 모델이 스테레오 입력)
- 받은 스트림은 재인코딩하지 않고 원래 컨테이너 그대로 저장 → 이후 FFmpeg 로 PCM 디코딩 1회
  (MP3 등 중간 손실 변환 없음. 원본이 이미 손실 압축이라 사라진 정보는 복원되지 않음)

권리 안내: 본인이 권리를 가졌거나 권리자가 다운로드를 허락한 콘텐츠(본인 업로드, 크리에이티브 커먼즈 등)만
처리하세요. DRM 보호·비공개·회원 전용·연령 제한 영상은 처리하지 않습니다.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import logging
import os
import re
import shutil
import sys
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import paths

log = logging.getLogger(__name__)

MAX_DURATION_SEC = 40 * 60
YTDLP_LATEST = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/"

RIGHTS_NOTICE = (
    "YouTube 입력은 본인이 권리를 가지고 있거나 권리자가 다운로드를 허락한 콘텐츠만 사용해 주세요.\n"
    "(예: 본인·모임이 올린 영상, 크리에이티브 커먼즈 라이선스 영상)\n\n"
    "저작권이 있는 음원을 허락 없이 내려받는 것은 YouTube 서비스 약관과 저작권법에 위반될 수 있습니다. "
    "DRM으로 보호된 영상, 비공개·회원 전용·연령 제한 영상은 처리하지 않습니다."
)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "youtube-nocookie.com"}


class YouTubeError(Exception):
    """사용자에게 보여줄 수 있는 오류. kind 로 재시도/업데이트 안내 여부를 판단."""

    def __init__(self, message: str, kind: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class YouTubeCancelled(Exception):
    pass


# ------------------------------------------------------------------ URL ---

def video_id(text: str) -> Optional[str]:
    """YouTube 영상 주소에서 영상 ID(11자). 영상 주소가 아니면 None."""
    text = (text or "").strip()
    if not text:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        text = "https://" + text
    try:
        u = urllib.parse.urlsplit(text)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in _HOSTS:
        return None
    parts = [p for p in u.path.split("/") if p]
    vid = None
    if host == "youtu.be":
        vid = parts[0] if parts else None
    elif parts and parts[0] == "watch":
        vid = (urllib.parse.parse_qs(u.query).get("v") or [None])[0]
    elif len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v", "e"):
        vid = parts[1]
    if vid and _ID_RE.match(vid):
        return vid
    return None


def canonical_url(text: str) -> Optional[str]:
    vid = video_id(text)
    return f"https://www.youtube.com/watch?v={vid}" if vid else None


def looks_like_url(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith(("http://", "https://", "www.", "youtu", "m.youtube", "music.youtube"))


# ------------------------------------------------------ format selection ---

def codec_tier(acodec: str) -> tuple[int, str]:
    """코덱 등급 (작을수록 좋음)과 표시 이름. 서로 다른 코덱은 비트레이트가 아니라 이 등급으로 비교."""
    a = (acodec or "").lower()
    if a.startswith(("flac", "alac", "pcm", "wav")):
        return 0, "무손실"
    if a.startswith("opus"):
        return 1, "Opus"
    if a.startswith("vorbis"):
        return 2, "Vorbis"
    if a.startswith(("mp4a.40.5", "mp4a.40.29")):     # HE-AAC (저비트레이트용)
        return 5, "HE-AAC"
    if a.startswith(("mp4a", "aac")):
        return 3, "AAC"
    if a.startswith(("mp3", "mp4a.40.34", "mp4a.6b")):
        return 4, "MP3"
    if a.startswith(("ac-3", "ec-3", "eac3", "ac3")):
        return 6, "AC-3"
    return 7, acodec or "?"


def _is_excluded(f: dict) -> Optional[str]:
    fid = str(f.get("format_id") or "")
    note = str(f.get("format_note") or "").lower()
    if f.get("has_drm"):
        return "DRM"
    if fid.endswith("-drc") or "drc" in note.split(", ") or " drc" in note:
        return "DRC(음량 압축)"
    if "damaged" in note or (f.get("preference") or 0) <= -10:
        return "손상"
    if (f.get("language_preference") or 0) <= -10 or "descriptive" in note:
        return "해설 음성"
    if str(f.get("protocol") or "") in ("mhtml",) or "storyboard" in note:
        return "미리보기"
    return None


@dataclass
class FormatChoice:
    format: dict
    reason: str
    audio_only: bool
    table: list = field(default_factory=list)   # (format_id, codec, abr, asr, ch, 상태)


def _abr(f: dict) -> float:
    return float(f.get("abr") or f.get("tbr") or 0)


def select_format(formats: list) -> FormatChoice:
    """가장 적합한 오디오 스트림 선택. 후보가 없으면 YouTubeError."""
    table = []
    audio_only, combined = [], []
    for f in formats or []:
        ac, vc = f.get("acodec"), f.get("vcodec")
        if not ac or ac == "none":
            continue
        why = _is_excluded(f)
        tier, cname = codec_tier(ac)
        table.append((str(f.get("format_id")), cname, round(_abr(f)), f.get("asr"), f.get("audio_channels"),
                      why or ("오디오 전용" if (not vc or vc == "none") else "영상 포함")))
        if why:
            continue
        (audio_only if (not vc or vc == "none") else combined).append(f)

    def key(f: dict):
        ch = f.get("audio_channels") or 2
        return (-(f.get("language_preference") or 0),     # 원래 언어 트랙 우선
                0 if ch <= 2 else 1,                       # 스테레오 이하 우선
                codec_tier(f.get("acodec"))[0],            # 코덱 등급
                -(f.get("source_preference") or 0),        # (Premium 등 공식 고품질 표시)
                -_abr(f), -(f.get("asr") or 0))

    if audio_only:
        best = sorted(audio_only, key=key)[0]
        tier, cname = codec_tier(best.get("acodec"))
        reason = (f"오디오 전용 스트림 {len(audio_only)}개 중 {cname} {round(_abr(best))}kbps "
                  f"{(best.get('asr') or 0) / 1000:g}kHz 선택 (코덱 등급 우선, 같은 코덱 안에서만 비트레이트 비교)")
        return FormatChoice(best, reason, True, table)
    if combined:
        best = sorted(combined, key=lambda f: (key(f)[:3], f.get("height") or 0, key(f)[3:]))[0]
        reason = "오디오 전용 스트림이 없어 해상도가 가장 낮은 영상 포함 스트림에서 오디오만 사용"
        return FormatChoice(best, reason, False, table)
    raise YouTubeError("이 영상에서 내려받을 수 있는 오디오 스트림을 찾지 못했습니다.", "no_audio")


# ------------------------------------------------------- yt-dlp loading ---

def tools_dir() -> Path:
    p = paths.user_data_dir() / "tools"
    p.mkdir(parents=True, exist_ok=True)
    return p


def updated_zip() -> Path:
    return tools_dir() / "yt-dlp.zip"


class _OverrideFinder(importlib.abc.MetaPathFinder):
    """업데이트로 받은 yt-dlp(zip)를 번들된 버전보다 먼저 불러오기."""

    def __init__(self, root: str):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in ("yt_dlp", "yt_dlp_ejs"):
            return None
        search = [self.root] if "." not in fullname else path
        return importlib.machinery.PathFinder.find_spec(fullname, search)


_override_installed = False


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:4])


def _zip_version(zp: Path) -> str:
    import zipfile

    try:
        with zipfile.ZipFile(zp) as z:
            m = re.search(r"__version__\s*=\s*'([^']+)'", z.read("yt_dlp/version.py").decode("utf-8", "replace"))
            return m.group(1) if m else ""
    except Exception:
        return ""


def _bundled_version() -> str:
    try:
        from importlib.metadata import version

        return version("yt-dlp")
    except Exception:
        return ""


def load_ytdlp():
    """yt_dlp 모듈. 사용자 폴더에 더 새 버전(업데이트)이 있으면 그것을 사용."""
    global _override_installed
    zp = updated_zip()
    if not _override_installed and zp.is_file():
        zv = _zip_version(zp)
        bv = _bundled_version()
        if zv and (not bv or _version_tuple(zv) > _version_tuple(bv)):
            sys.meta_path.insert(0, _OverrideFinder(str(zp)))
            _override_installed = True
            log.info("using updated yt-dlp %s (bundled %s)", zv, bv)
    import yt_dlp

    return yt_dlp


def ytdlp_version() -> str:
    try:
        return load_ytdlp().version.__version__
    except Exception as exc:
        return f"사용 불가 ({exc})"


def find_deno() -> Optional[str]:
    exe = "deno.exe" if sys.platform == "win32" else "deno"
    env = os.environ.get("MUSIC_SEPARATOR_DENO")
    cands = [Path(env)] if env else []
    cands.append(paths.resource_dir() / "bin" / exe)
    try:
        import deno  # PyPI 'deno' 패키지 (개발 환경)

        cands.append(Path(deno.find_deno_bin()))
    except Exception:
        pass
    w = shutil.which(exe)
    if w:
        cands.append(Path(w))
    for c in cands:
        if c and c.is_file():
            return str(c)
    return None


def update_downloader(progress: Optional[Callable[[int, int], None]] = None,
                      cancel: Optional[threading.Event] = None) -> str:
    """최신 yt-dlp(zipimport 배포본, yt-dlp-ejs 포함)를 GitHub 에서 받아 SHA-256 확인 후 저장.
    프로그램을 다시 시작하면 적용됩니다. 반환: 받은 버전."""
    from .models import _download, _sha256

    dest = updated_zip()
    tmp = dest.with_name("yt-dlp.download")
    sums = dest.with_name("SHA2-256SUMS")
    try:
        _download(YTDLP_LATEST + "SHA2-256SUMS", sums, "", None, cancel, 0, 0)
        expected = None
        for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == "yt-dlp":
                expected = parts[0].lower()
        if not expected:
            raise YouTubeError("업데이트 정보를 확인하지 못했습니다.", "update")
        _download(YTDLP_LATEST + "yt-dlp", tmp, "",
                  (lambda d, t, _l: progress(d, t)) if progress else None, cancel, 0, 0)
        if _sha256(tmp) != expected:
            raise YouTubeError("받은 파일이 손상되었습니다 (체크섬 불일치). 다시 시도해 주세요.", "update")
        ver = _zip_version(tmp)
        if not ver:
            raise YouTubeError("받은 파일을 확인할 수 없습니다.", "update")
        os.replace(tmp, dest)
        return ver
    finally:
        tmp.unlink(missing_ok=True)
        sums.unlink(missing_ok=True)


# ------------------------------------------------------------- download ---

@dataclass
class AcquiredAudio:
    path: Path
    video_id: str
    title: str
    uploader: str
    duration: float
    license: str
    format_id: str
    codec: str               # 예: opus
    codec_name: str          # 예: Opus
    abr_kbps: float
    sample_rate: int
    channels: int
    container: str           # webm / m4a …
    filesize: int
    audio_only: bool
    reason: str
    table: list
    reencoded: bool = False  # 항상 False (재인코딩하지 않음)


def friendly_error(exc: BaseException) -> YouTubeError:
    if isinstance(exc, YouTubeError):
        return exc
    msg = str(exc)
    low = msg.lower()
    checks = [
        (("drm",), "DRM으로 보호된 영상이라 처리할 수 없습니다.", "drm", False),
        (("private video",), "비공개 영상이라 내려받을 수 없습니다.", "private", False),
        (("members-only", "join this channel", "members only"), "채널 회원 전용 영상이라 내려받을 수 없습니다.",
         "members", False),
        (("confirm your age", "age-restricted", "inappropriate for some users"),
         "연령 제한 영상이라 내려받을 수 없습니다.", "age", False),
        (("not a bot", "sign in to confirm"),
         "YouTube가 자동 접근 확인을 요구해 내려받지 못했습니다.\n잠시 후 다시 시도하거나, 다른 네트워크에서 시도해 주세요.",
         "bot", True),
        (("not available in your country", "geo restrict", "blocked it in your country"),
         "이 지역에서는 볼 수 없는 영상입니다.", "geo", False),
        (("live event will begin", "premieres in", "is_upcoming"), "아직 시작하지 않은 영상(예정된 방송)입니다.",
         "upcoming", False),
        (("copyright",), "저작권 문제로 삭제되었거나 볼 수 없는 영상입니다.", "unavailable", False),
        (("video unavailable", "this video is unavailable", "has been removed", "does not exist",
          "account associated with this video has been terminated"),
         "삭제되었거나 볼 수 없는 영상입니다. 주소를 확인해 주세요.", "unavailable", False),
        (("unsupported url", "is not a valid url", "incomplete youtube id"),
         "올바른 YouTube 영상 주소가 아닙니다.", "invalid_url", False),
        (("timed out", "timeout", "temporary failure in name resolution", "getaddrinfo", "network is unreachable",
          "connection reset", "connection refused", "connection aborted", "urlopen error", "no route to host",
          "ssl", "remote end closed", "unable to download webpage"),
         "네트워크 오류로 내려받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.", "network", True),
        (("http error 403", "signature", "nsig", "unable to extract", "challenge", "player response",
          "requested format is not available", "only images are available", "javascript runtime", "ejs"),
         "YouTube가 바뀌어 다운로드 도구 업데이트가 필요할 수 있습니다.\n[YouTube 기능 업데이트] 후 다시 시도해 주세요.",
         "outdated", True),
    ]
    for keys, text, kind, retry in checks:
        if any(k in low for k in keys):
            return YouTubeError(text, kind, retry)
    short = re.sub(r"\x1b\[[0-9;]*m", "", msg).replace("ERROR: ", "").strip()
    return YouTubeError(f"YouTube 오디오를 가져오지 못했습니다.\n({short[:300]})", "unknown", True)


class _Logger:
    def __init__(self):
        self.lines: list[str] = []

    def debug(self, msg):
        if msg.startswith("[debug] "):
            return
        self.lines.append(msg)

    def info(self, msg):
        self.lines.append(msg)

    def warning(self, msg):
        self.lines.append("WARNING: " + msg)
        log.info("yt-dlp: %s", msg)

    def error(self, msg):
        self.lines.append("ERROR: " + msg)
        log.warning("yt-dlp: %s", msg)


def _base_params(dest_dir: Path, logger: _Logger) -> dict:
    from .audio_io import find_ffmpeg

    params = {
        "quiet": True, "no_warnings": False, "noprogress": True, "logger": logger,
        "noplaylist": True, "playlist_items": "1",
        "socket_timeout": 20, "retries": 5, "fragment_retries": 5, "extractor_retries": 2,
        "overwrites": True, "continuedl": False,
        "outtmpl": {"default": str(Path(dest_dir) / "%(id)s.f%(format_id)s.%(ext)s")},
        "paths": {"home": str(dest_dir), "temp": str(dest_dir)},
        "cachedir": str(paths.cache_root() / "yt-dlp"),
        "postprocessors": [],          # 오디오 변환(MP3 등) 하지 않음
        "check_formats": False,
        "windowsfilenames": True,
    }
    try:
        params["ffmpeg_location"] = find_ffmpeg()   # DASH 컨테이너 보정(무손실 리먹스) 용
    except Exception:
        pass
    deno = find_deno()
    if deno:
        params["js_runtimes"] = {"deno": {"path": deno}}
    return params


def fetch_info(url: str, cancel: Optional[threading.Event] = None) -> dict:
    """영상 정보만 확인 (다운로드 없음)."""
    curl = canonical_url(url)
    if not curl:
        raise YouTubeError("올바른 YouTube 영상 주소가 아닙니다.\n예: https://www.youtube.com/watch?v=…", "invalid_url")
    yt = load_ytdlp()
    logger = _Logger()
    tmp = paths.cache_root() / "yt-info"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with yt.YoutubeDL(_base_params(tmp, logger)) as ydl:
            info = ydl.extract_info(curl, download=False, process=False)
            info = ydl.process_ie_result(info, download=False) if info.get("_type") != "playlist" else info
    except Exception as exc:  # noqa: BLE001
        raise friendly_error(exc) from exc
    return info


def acquire(url: str, dest_dir: Path, progress: Optional[Callable[[float, str], None]] = None,
            cancel: Optional[threading.Event] = None) -> AcquiredAudio:
    """URL 확인 → 정보 조회 → 최고 품질 오디오 스트림 선택 → 원본 그대로 다운로드."""
    cancel = cancel or threading.Event()
    report = progress or (lambda f, m: None)
    curl = canonical_url(url)
    if not curl:
        raise YouTubeError("올바른 YouTube 영상 주소가 아닙니다.\n예: https://www.youtube.com/watch?v=…", "invalid_url")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    report(0.02, "YouTube 영상 정보 확인 중…")
    yt = load_ytdlp()
    logger = _Logger()
    params = _base_params(dest_dir, logger)
    try:
        with yt.YoutubeDL(params) as ydl:
            info = ydl.extract_info(curl, download=False, process=False)
    except Exception as exc:  # noqa: BLE001
        raise friendly_error(exc) from exc
    if cancel.is_set():
        raise YouTubeCancelled()
    if info.get("_type") == "playlist":
        raise YouTubeError("재생목록이 아니라 영상 하나의 주소를 넣어 주세요.", "invalid_url")
    live = info.get("live_status") or ("is_live" if info.get("is_live") else "")
    if live in ("is_live", "is_upcoming"):
        raise YouTubeError("실시간 방송이나 예정된 방송은 처리할 수 없습니다. 방송이 끝난 뒤 다시 시도해 주세요.",
                           "live")
    dur = float(info.get("duration") or 0)
    if dur > MAX_DURATION_SEC:
        raise YouTubeError(f"영상이 너무 깁니다 ({int(dur // 60)}분). {MAX_DURATION_SEC // 60}분 이하 영상만 처리합니다.",
                           "too_long")
    formats = info.get("formats") or []
    if any(f.get("has_drm") for f in formats) and not any(not f.get("has_drm") for f in formats):
        raise YouTubeError("DRM으로 보호된 영상이라 처리할 수 없습니다.", "drm")
    choice = select_format(formats)
    fmt = choice.format
    log.info("youtube %s: %s", info.get("id"), choice.reason)
    report(0.06, "오디오 내려받는 중…")

    def hook(d):
        if cancel.is_set():
            raise yt.utils.DownloadCancelled("cancelled")
        if d.get("status") == "downloading":
            done = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or fmt.get("filesize") or 0
            frac = done / total if total else 0.0
            report(0.06 + 0.9 * min(1.0, frac),
                   f"오디오 내려받는 중… {done / 1048576:.1f}" + (f" / {total / 1048576:.1f} MB" if total else " MB"))

    params = dict(params, format=str(fmt["format_id"]), progress_hooks=[hook])
    try:
        with yt.YoutubeDL(params) as ydl:
            result = ydl.process_ie_result(info, download=True)
    except yt.utils.DownloadCancelled as exc:
        _cleanup_partial(dest_dir)
        raise YouTubeCancelled() from exc
    except Exception as exc:  # noqa: BLE001
        _cleanup_partial(dest_dir)
        if cancel.is_set():
            raise YouTubeCancelled() from exc
        raise friendly_error(exc) from exc
    files = [Path(d["filepath"]) for d in (result.get("requested_downloads") or []) if d.get("filepath")]
    if not files or not files[0].is_file():
        cands = sorted(dest_dir.glob(f"{info.get('id')}.f*"), key=lambda p: p.stat().st_mtime, reverse=True)
        files = [p for p in cands if not p.name.endswith((".part", ".ytdl"))]
    if not files:
        raise YouTubeError("오디오 파일을 저장하지 못했습니다.", "unknown", True)
    path = files[0]
    report(1.0, "오디오 내려받기 완료")
    tier, cname = codec_tier(fmt.get("acodec"))
    return AcquiredAudio(
        path=path, video_id=str(info.get("id")), title=str(info.get("title") or info.get("id")),
        uploader=str(info.get("uploader") or info.get("channel") or ""), duration=dur,
        license=str(info.get("license") or ""), format_id=str(fmt.get("format_id")),
        codec=str(fmt.get("acodec")), codec_name=cname, abr_kbps=float(_abr(fmt)),
        sample_rate=int(fmt.get("asr") or 0), channels=int(fmt.get("audio_channels") or 0),
        container=str(fmt.get("ext") or path.suffix.lstrip(".")), filesize=path.stat().st_size,
        audio_only=choice.audio_only, reason=choice.reason, table=choice.table)


def _cleanup_partial(folder: Path) -> None:
    for p in Path(folder).glob("*"):
        if p.name.endswith((".part", ".ytdl", ".temp")) or ".part-Frag" in p.name:
            try:
                p.unlink()
            except OSError:
                pass


def sha1_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
