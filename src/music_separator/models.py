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
    kind: str = "demucs"
    file_name: str = ""    # kind == "roformer" / "rmvpe": 가중치 파일 이름 (파일 하나짜리 모델)
    sha256: str = ""
    urls: tuple = ()
    config_name: str = ""  # kind == "roformer": ZFTurbo 형식 YAML 설정 (없으면 BS-RoFormer SW 고정 설정)
    config_urls: tuple = ()
    experimental: bool = False   # 실험 기능용 (일반 분리 엔진 목록에는 넣지 않음)


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
    # htdemucs_ft 의 보컬 전용 하위 모델 1개 (2-Stem 속도 우선 모드). htdemucs_ft 의 다른 3개 모델은
    # 각각 드럼/베이스/기타 전용이라 보컬·반주 분리에는 필요 없음 → 계산량 1/4, 보컬 품질은 동일.
    "htdemucs_ft_vocals": ModelSpec(
        "htdemucs_ft_vocals", "Hybrid Transformer Demucs Fine-tuned (보컬 전용)", "HTDemucs-ft",
        ("04573f0d",), ("04573f0d-f3cf25b2.th",), 80,
        ("drums", "bass", "other", "vocals"),
        "models: ['04573f0d']\n",
    ),
    "htdemucs_6s": ModelSpec(
        "htdemucs_6s", "Hybrid Transformer Demucs 6-Source (실험적)", "HTDemucs-6s",
        ("5c90dfd2",), ("5c90dfd2-34c22ccb.th",), 55,
        ("drums", "bass", "other", "vocals", "guitar", "piano"),
        "models: ['5c90dfd2']\n",
    ),
    # BS-RoFormer SW (6-stem, jarredou). 커뮤니티 학습 모델 — 가중치 사용 조건 미표기 (개인용 권장)
    "bs_roformer_sw": ModelSpec(
        "bs_roformer_sw", "BS-RoFormer SW 6-Stem (고급)", "", (), (), 667,
        ("bass", "drums", "other", "vocals", "guitar", "piano"), "",
        kind="roformer", file_name="BS-Roformer-SW.ckpt",
        sha256="24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
        urls=("https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/"
              "BS-Roformer-SW.ckpt",),
    ),
    # [실험] 화음 따기 ① 메인/화음(백보컬) 분리 — frazer & becruily 의 BS-RoFormer 카라오케 모델
    # (가중치 사용 조건 미표기 → 앱에 넣지 않고 처음 쓸 때 작성자 저장소에서 내려받음). 메인 보컬(Vocals) 하나만
    # 추정하고, 화음 = 입력(보컬 트랙) − 메인 보컬.
    "karaoke_lead": ModelSpec(
        "karaoke_lead", "BS-RoFormer 메인/화음 분리 (frazer & becruily, 실험)", "", (), (), 195,
        ("vocals",), "",
        kind="roformer", file_name="bs_roformer_karaoke_frazer_becruily.ckpt",
        sha256="eb90ee24c1154d83fbcfd27e96182f19e061557cc6e4746953125e08c29389f9",
        urls=("https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/"
              "bs_roformer_karaoke_frazer_becruily.ckpt",),
        config_name="config_karaoke_frazer_becruily.yaml",
        config_urls=("https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/"
                     "config_karaoke_frazer_becruily.yaml",),
        experimental=True,
    ),
    # [실험] 화음 따기 ② 보컬 음정 추적 RMVPE (RVC-Project, MIT)
    "rmvpe": ModelSpec(
        "rmvpe", "RMVPE 보컬 음정 추적", "", (), (), 173, (), "",
        kind="rmvpe", file_name="rmvpe.pt",
        sha256="6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
        urls=("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",),
        experimental=True,
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


def _installed_in(folder: Path, spec: ModelSpec) -> bool:
    if not folder.is_dir():
        return False
    if spec.file_name:                       # 파일 하나짜리 모델 (BS-RoFormer, RMVPE)
        f = folder / spec.file_name
        if spec.config_name and not (folder / spec.config_name).is_file():
            return False
        return f.exists() and f.stat().st_size > spec.approx_mb * 1024 * 1024 * 0.9
    return all(_sig_file(folder, s) for s in spec.sigs)


# 같은 가중치 파일을 다른 모델 폴더에서도 찾음 (예: htdemucs_ft 를 이미 받았으면 보컬 전용 모델도 있는 것)
ALT_DIRS = {"htdemucs_ft_vocals": ("htdemucs_ft",)}


def find_model_dir(name: str) -> Optional[Path]:
    if name not in MODELS:   # 테스트용 내장 모델
        return Path(".") if name.endswith("_unittest") or name.endswith("_unittest6") else None
    spec = MODELS[name]
    for root in search_dirs():
        for sub in (name, *ALT_DIRS.get(name, ())):
            if _installed_in(root / sub, spec):
                return root / sub
    return None


def is_installed(name: str) -> bool:
    return find_model_dir(name) is not None


def missing_download_mb(name: str) -> int:
    if name not in MODELS or is_installed(name):
        return 0
    return MODELS[name].approx_mb


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
    if spec.file_name:
        return _download_single(spec, folder, progress, cancel)
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_single(spec: ModelSpec, folder: Path, progress: Optional[ProgressCb],
                     cancel: Optional[threading.Event]) -> Path:
    if spec.config_name and not (folder / spec.config_name).is_file():
        errs = []
        for url in spec.config_urls:
            try:
                _download(url, folder / spec.config_name, "AI 모델 설정 다운로드", progress, cancel, 0, 1)
                break
            except DownloadCancelled:
                raise
            except Exception as exc:
                errs.append(str(exc))
        else:
            raise ModelError("AI 모델 설정 파일을 다운로드하지 못했습니다. 인터넷 연결을 확인해 주세요.\n"
                             + "\n".join(errs))
    dest = folder / spec.file_name
    if dest.exists() and dest.stat().st_size > spec.approx_mb * 1024 * 1024 * 0.9:
        return folder
    total = spec.approx_mb * 1024 * 1024
    errors = []
    urls = list(spec.urls)
    extra = os.environ.get("MUSIC_SEPARATOR_ROFORMER_URL")
    if extra:
        urls.insert(0, extra)
    for url in urls:
        try:
            _download(url, dest, "고급 AI 모델 다운로드", progress, cancel, 0, total)
            if spec.sha256 and _sha256(dest) != spec.sha256:
                dest.unlink(missing_ok=True)
                raise ModelError("체크섬 불일치 (파일 손상 또는 다른 파일)")
            if progress:
                progress(total, total, "다운로드 완료")
            return folder
        except DownloadCancelled:
            raise
        except Exception as exc:
            log.warning("download failed %s: %s", url, exc)
            errors.append(str(exc))
    raise ModelError("고급 AI 모델을 다운로드하지 못했습니다. 인터넷 연결을 확인해 주세요.\n\n"
                     "인터넷이 없는 PC 라면, 다른 PC 에서 받은 models 폴더를\n"
                     f"{user_models_dir()} 에 복사하면 됩니다.\n\n" + "\n".join(errors))


# 예전 버전에서 내려받았지만 더 이상 쓰지 않는 모델 폴더 (사용자 모델 폴더에서만 지움)
OBSOLETE_DIRS = ("duet_male_female",)      # v1.3.0~1.3.4 실험실 남/여 분리 모델 (약 503MB)


def remove_obsolete() -> list:
    removed = []
    for name in OBSOLETE_DIRS:
        d = user_models_dir() / name
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(name)
            log.info("removed obsolete model folder %s", d)
    return removed


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
        if name in ("demucs_unittest", "demucs_unittest6"):  # 테스트용 초소형 무작위 모델
            import torch
            from demucs.apply import BagOfModels
            from demucs.hdemucs import HDemucs

            torch.manual_seed(0)
            srcs = ["drums", "bass", "other", "vocals"] + (["guitar", "piano"] if name.endswith("6") else [])
            model = BagOfModels([HDemucs(channels=4, sources=srcs)])
            model.eval()
            _model_cache[name] = model
            return model
        if name == "karaoke_unittest":  # 테스트용 초소형 메인/화음 분리 RoFormer (1-stem, 구조만 동일)
            import torch

            from .roformer import SW_CONFIG, BSRoformer

            torch.manual_seed(0)
            model = BSRoformer(**dict(SW_CONFIG, dim=16, depth=1, dim_head=8, heads=2, num_stems=1))
            model.instruments, model.chunk, model.num_overlap = ("vocals",), 44100 * 2, 2
            model.eval()
            _model_cache[name] = model
            return model
        if name == "rmvpe_unittest":  # 테스트용 초소형 무작위 RMVPE (입출력 형태만 동일)
            import torch

            from .rmvpe import E2E

            torch.manual_seed(0)
            model = E2E(1, 1, (2, 2), inter_layers=1, en_out_channels=4)
            model.eval()
            _model_cache[name] = model
            return model
        if name == "roformer_unittest":  # 테스트용 초소형 무작위 RoFormer (구조만 동일)
            import torch

            from .roformer import SW_CONFIG, SW_INSTRUMENTS, BSRoformer

            torch.manual_seed(0)
            cfg = dict(SW_CONFIG, dim=16, depth=1, dim_head=8, heads=2)
            model = BSRoformer(**cfg)
            model.instruments = SW_INSTRUMENTS
            model.eval()
            _model_cache[name] = model
            return model
        folder = find_model_dir(name)
        if folder is None:
            raise ModelError(f"모델 '{name}' 이(가) 설치되어 있지 않습니다.")
        spec = MODELS[name]
        if spec.kind == "rmvpe":
            from . import rmvpe

            try:
                model = rmvpe.load(folder / spec.file_name)
            except Exception as exc:
                raise ModelError(
                    f"음정 추적 AI 모델 파일을 읽을 수 없습니다 (손상되었을 수 있음): {spec.file_name}\n"
                    f"해당 파일을 삭제하고 다시 시도하면 새로 다운로드합니다.\n({exc})") from exc
            _model_cache[name] = model
            return model
        if spec.kind == "roformer":
            from .roformer import SW_INSTRUMENTS, load_checkpoint, load_sw, read_config

            try:
                if spec.config_name:
                    model = load_checkpoint(folder / spec.file_name, read_config(folder / spec.config_name))
                    _model_cache[name] = model
                    return model
                model = load_sw(folder / spec.file_name)
            except Exception as exc:
                raise ModelError(
                    f"고급 AI 모델 파일을 읽을 수 없습니다 (손상되었을 수 있음): {spec.file_name}\n"
                    f"해당 파일을 삭제하고 다시 시도하면 새로 다운로드합니다.\n({exc})") from exc
            model.instruments = SW_INSTRUMENTS
            _model_cache[name] = model
            return model
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
    src_root = Path(src_root)
    for spec in MODELS.values():   # 체크포인트 파일만 있는 경우 (예: BS-Roformer-SW.ckpt)
        if spec.file_name and (src_root / spec.file_name).is_file():
            dst = user_models_dir() / spec.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_root / spec.file_name, dst / spec.file_name)
            if spec.config_name and (src_root / spec.config_name).is_file():
                shutil.copy2(src_root / spec.config_name, dst / spec.config_name)
            imported.append(spec.name)
    for name in MODELS:
        src = Path(src_root) / name
        if src.is_dir():
            dst = user_models_dir() / name
            shutil.copytree(src, dst, dirs_exist_ok=True)
            imported.append(name)
    return imported
