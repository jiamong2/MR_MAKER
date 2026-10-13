"""Demucs 사전학습 모델 관리 (탐색 / 다운로드 / 로드).

모델 저장 구조 (번들 폴더 또는 사용자 모델 폴더):

    models/
      htdemucs/
        htdemucs.yaml
        955717e8.safetensors        (HuggingFace 형식)  또는
        955717e8-8726e21a.th        (기존 Meta 서버 형식)

다운로드 순서: HuggingFace(adefossez/*) → 실패 시 dl.fbaipublicfiles.com(기존 서버).
인터넷이 없는 PC 에서는 다른 PC 의 models 폴더를 그대로 복사해 넣으면 됩니다.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

from . import __version__
from .paths import bundled_models_dir, user_models_dir

log = logging.getLogger(__name__)

LEGACY_ROOT = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
HF_NAMESPACE = "adefossez"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    hf_repo: str
    sigs: tuple            # 모델 시그니처 (bag 구성)
    legacy_files: tuple    # 기존 서버 파일명 (sig-checksum.th)
    approx_mb: int
    sources: tuple
    yaml_text: str


MODELS: dict[str, ModelSpec] = {
    "htdemucs": ModelSpec(
        "htdemucs", "Hybrid Transformer Demucs (표준)", "HTDemucs",
        ("955717e8",), ("955717e8-8726e21a.th",), 80,
        ("drums", "bass", "other", "vocals"),
        "models: ['955717e8']\n",
    ),
    "htdemucs_ft": ModelSpec(
        "htdemucs_ft", "Hybrid Transformer Demucs Fine-tuned (고품질, 4배 느림)", "HTDemucs-ft",
        ("f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d"),
        ("f7e0c4bc-ba3fe64a.th", "d12395a8-e57c48e6.th", "92cfc3b6-ef3bcb9c.th", "04573f0d-f3cf25b2.th"),
        320,
        ("drums", "bass", "other", "vocals"),
        "models: ['f7e0c4bc', 'd12395a8', '92cfc3b6', '04573f0d']\n"
        "weights: [\n  [1., 0., 0., 0.],\n  [0., 1., 0., 0.],\n  [0., 0., 1., 0.],\n  [0., 0., 0., 1.],\n]\n",
    ),
    "htdemucs_6s": ModelSpec(
        "htdemucs_6s", "Hybrid Transformer Demucs 6-Source (실험적)", "HTDemucs-6s",
        ("5c90dfd2",), ("5c90dfd2-34c22ccb.th",), 55,
        ("drums", "bass", "other", "vocals", "guitar", "piano"),
        "models: ['5c90dfd2']\n",
    ),
}


class ModelError(Exception):
    pass


class DownloadCancelled(Exception):
    pass


# ------------------------------------------------------------- discovery ---

def _sig_file(folder: Path, sig: str) -> Optional[Path]:
    st = folder / f"{sig}.safetensors"
    if st.exists() and st.stat().st_size > 1000:
        return st
    for th in folder.glob(f"{sig}-*.th"):
        if th.stat().st_size > 1000:
            return th
    th = folder / f"{sig}.th"
    if th.exists():
        return th
    return None


def search_dirs() -> list[Path]:
    dirs = []
    extra = os.environ.get("MUSIC_SEPARATOR_MODELS_DIR")
    if extra:
        dirs.append(Path(extra))
    dirs += [bundled_models_dir(), user_models_dir()]
    return dirs


def find_model_dir(name: str) -> Optional[Path]:
    spec = MODELS[name]
    for root in search_dirs():
        folder = root / name
        if not folder.is_dir():
            continue
        if all(_sig_file(folder, s) for s in spec.sigs):
            return folder
    return None


def is_installed(name: str) -> bool:
    return find_model_dir(name) is not None


def missing_download_mb(name: str) -> int:
    return 0 if is_installed(name) else MODELS[name].approx_mb


# -------------------------------------------------------------- download ---

def _ssl_context() -> ssl.SSLContext:
    # 회사 프록시 등 OS 인증서 저장소를 쓰는 환경을 위해 truststore 우선
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


ProgressCb = Callable[[int, int, str], None]


def _download(url: str, dest: Path, label: str, progress: Optional[ProgressCb],
              cancel: Optional[threading.Event], base_done: int, base_total: int) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": f"MusicSeparator/{__version__}"})
    ctx = _ssl_context()
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as f:
            last = 0.0
            while True:
                if cancel is not None and cancel.is_set():
                    f.close()
                    tmp.unlink(missing_ok=True)
                    raise DownloadCancelled()
                buf = resp.read(1 << 16)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                now = time.monotonic()
                if progress and now - last > 0.1:
                    last = now
                    progress(base_done + done, max(base_total, base_done + (total or done)), label)
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"다운로드가 중간에 끊겼습니다 ({done}/{total} bytes)")
    os.replace(tmp, dest)
    return done


def _sha256_prefix_ok(path: Path, prefix: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().startswith(prefix)


def download_model(name: str, progress: Optional[ProgressCb] = None,
                   cancel: Optional[threading.Event] = None) -> Path:
    """모델을 사용자 모델 폴더에 다운로드. 이미 있으면 바로 반환."""
    existing = find_model_dir(name)
    if existing:
        return existing
    spec = MODELS[name]
    folder = user_models_dir() / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.yaml").write_text(spec.yaml_text, encoding="utf-8")

    hf = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    est_total = spec.approx_mb * 1024 * 1024
    done_total = 0
    n = len(spec.sigs)
    for i, (sig, legacy) in enumerate(zip(spec.sigs, spec.legacy_files), 1):
        if _sig_file(folder, sig):
            continue
        label = f"AI 모델 다운로드 ({i}/{n})"
        errors = []
        # 1) HuggingFace
        try:
            url = f"{hf}/{HF_NAMESPACE}/{spec.hf_repo}/resolve/main/{sig}.safetensors"
            done_total += _download(url, folder / f"{sig}.safetensors", label, progress, cancel,
                                    done_total, est_total)
            continue
        except DownloadCancelled:
            raise
        except Exception as exc:
            log.warning("HF download failed for %s: %s", sig, exc)
            errors.append(f"HuggingFace: {exc}")
        # 2) 기존 Meta 서버
        try:
            dest = folder / legacy
            legacy_root = os.environ.get("MUSIC_SEPARATOR_LEGACY_ROOT", LEGACY_ROOT)
            done_total += _download(legacy_root + legacy, dest, label, progress, cancel,
                                    done_total, est_total)
            checksum = legacy.split("-", 1)[1].split(".")[0]
            if not _sha256_prefix_ok(dest, checksum):
                dest.unlink(missing_ok=True)
                raise ModelError("체크섬 불일치 (파일 손상)")
            continue
        except DownloadCancelled:
            raise
        except Exception as exc:
            log.warning("legacy download failed for %s: %s", sig, exc)
            errors.append(f"Meta 서버: {exc}")
        raise ModelError(
            "AI 모델을 다운로드하지 못했습니다. 인터넷 연결을 확인해 주세요.\n\n"
            "인터넷이 없는 PC 라면, 다른 PC 에서 받은 models 폴더를\n"
            f"{user_models_dir()} 에 복사하면 됩니다.\n\n" + "\n".join(errors)
        )
    if progress:
        progress(est_total, est_total, "다운로드 완료")
    return folder


def remove_partial_downloads() -> None:
    for p in user_models_dir().rglob("*.part"):
        p.unlink(missing_ok=True)


# ------------------------------------------------------------------ load ---

_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def load_model(name: str):
    """Demucs 모델(BagOfModels) 로드. 한 번 로드하면 메모리에 캐시."""
    with _cache_lock:
        if name in _model_cache:
            return _model_cache[name]
        if name == "demucs_unittest":  # 테스트용 초소형 무작위 모델
            from demucs.apply import BagOfModels
            from demucs.pretrained import demucs_unittest

            model = BagOfModels([demucs_unittest()])
            model.eval()
            _model_cache[name] = model
            return model
        folder = find_model_dir(name)
        if folder is None:
            raise ModelError(f"모델 '{name}' 이(가) 설치되어 있지 않습니다.")
        spec = MODELS[name]
        yaml_path = folder / f"{name}.yaml"
        bag = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.exists() \
            else yaml.safe_load(spec.yaml_text)

        from demucs.apply import BagOfModels
        from demucs.states import load_model as load_th

        models = []
        for sig in bag["models"]:
            f = _sig_file(folder, sig)
            if f is None:
                raise ModelError(f"모델 파일 누락: {sig}")
            try:
                if f.suffix == ".safetensors":
                    from demucs.hf import load_safetensors_model

                    models.append(load_safetensors_model(f))
                else:
                    models.append(load_th(f))
            except Exception as exc:
                raise ModelError(
                    f"모델 파일을 읽을 수 없습니다 (손상되었을 수 있음): {f.name}\n"
                    f"해당 파일을 삭제하고 다시 시도하면 새로 다운로드합니다.\n({exc})"
                ) from exc
        model = BagOfModels(models, bag.get("weights"), bag.get("segment"))
        model.eval()
        _model_cache[name] = model
        return model


def forget(name: str) -> None:
    with _cache_lock:
        _model_cache.pop(name, None)


def clear_cache() -> None:
    with _cache_lock:
        _model_cache.clear()


def copy_into_user_dir(src_root: Path) -> list[str]:
    """외부 폴더(다른 PC 에서 복사한 models)에서 모델을 가져오기."""
    imported = []
    for name in MODELS:
        src = Path(src_root) / name
        if src.is_dir():
            dst = user_models_dir() / name
            shutil.copytree(src, dst, dirs_exist_ok=True)
            imported.append(name)
    return imported
