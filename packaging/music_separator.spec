# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 빌드 설정 (Windows / macOS / Linux 공용).

사용:  pyinstaller --noconfirm --clean packaging/music_separator.spec

환경 변수
  MS_BUNDLE_MODELS   : 모델 폴더 경로. 지정하면 앱 안에 AI 모델을 포함(오프라인 즉시 사용).
                       (scripts/fetch_models.py 로 미리 받아 둔 폴더)
  MS_CODESIGN_ID     : macOS 서명 인증서 이름 ("Developer ID Application: ..."), 없으면 ad-hoc
  MS_ENTITLEMENTS    : macOS entitlements plist 경로
  MS_TARGET_ARCH     : macOS 대상 아키텍처 (arm64 / x86_64), 기본은 현재 머신
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 (SPECPATH 는 PyInstaller 제공)
sys.path.insert(0, str(ROOT / "src"))
from music_separator import APP_NAME, APP_SHORT, BUNDLE_ID, EXE_NAME, __version__  # noqa: E402

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

datas = [(str(ROOT / "assets"), "assets")]
if (ROOT / "build" / "build_info.json").is_file():     # scripts/build.py 가 기록한 빌드 종류 (cpu / cuda …)
    datas.append((str(ROOT / "build" / "build_info.json"), "."))
datas += collect_data_files("demucs", includes=["remote/*.yaml", "remote/*.txt"])
datas += collect_data_files("imageio_ffmpeg", subdir="binaries")
datas += collect_data_files("certifi")

# 별도 실행 파일: 키 조절(rbshift, Rubber Band) / YouTube JavaScript 엔진(deno)
binaries = []
_exe = ".exe" if IS_WIN else ""
for cand in (ROOT / "build" / "rbshift" / f"rbshift{_exe}", ROOT / "build" / "rbshift" / "Release" / f"rbshift{_exe}"):
    if cand.is_file():
        binaries.append((str(cand), "bin"))
        print(f"[spec] bundling {cand}")
        break
else:
    print("[spec] WARNING: rbshift 가 없어 키 조절은 Signalsmith 로 대체됩니다 (scripts/build.py 가 먼저 빌드)")
try:
    import deno as _deno

    _d = Path(_deno.find_deno_bin())
    binaries.append((str(_d), "bin"))
    print(f"[spec] bundling deno {_d}")
except Exception as _exc:  # noqa: BLE001
    print(f"[spec] WARNING: deno 를 찾지 못함 ({_exc}) — YouTube 는 JavaScript 엔진 없이 동작 (일부 실패 가능)")

models_dir = os.environ.get("MS_BUNDLE_MODELS")
if models_dir and Path(models_dir).is_dir():
    for sub in Path(models_dir).iterdir():
        if sub.is_dir():
            datas.append((str(sub), f"models/{sub.name}"))
    print(f"[spec] bundling models from {models_dir}: {[p.name for p in Path(models_dir).iterdir()]}")

# demucs 모델 클래스는 문자열("demucs.htdemucs.HTDemucs")로 동적 import 되므로 명시적으로 포함
_skip = ("train", "solver", "evaluate", "grids", "distrib", "augment", "ema", "svd", "repitch",
         "separate", "api", "audio", "wav", "__main__")
hiddenimports = collect_submodules("demucs", filter=lambda n: n.split(".")[-1] not in _skip)
hiddenimports += ["safetensors", "safetensors.torch", "yaml", "lameenc", "imageio_ffmpeg",
                  "sounddevice", "psutil", "certifi", "truststore", "music_separator.smoke",
                  "music_separator.selftest", "music_separator.roformer", "music_separator.pitch",
                  "music_separator.export", "music_separator.pipeline", "music_separator.youtube",
                  "music_separator.ui.home", "music_separator.harmony", "music_separator.rmvpe",
                  "music_separator.transcribe", "music_separator.tempo", "python_stretch", "python_stretch.Signalsmith", "julius",
                  "soxr", "yt_dlp"]
hiddenimports += collect_submodules("yt_dlp_ejs")

excludes = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "torchaudio", "torchvision",
    "tensorboard", "torch.utils.tensorboard", "sphn", "huggingface_hub", "dora", "hydra", "pandas",
    "scipy", "sklearn", "PIL", "cv2", "sympy", "onnx", "triton",
    # PySide6 에서 사용하지 않는 큰 모듈
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtMultimedia",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "run.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe_name = EXE_NAME
icon = str(ROOT / "assets" / ("icon.ico" if IS_WIN else "icon.icns" if IS_MAC else "icon.png"))

version_file = None
if IS_WIN:
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct,
                                                     StringTable, VarFileInfo, VarStruct,
                                                     VSVersionInfo)
    nums = tuple(int(x) for x in (__version__.split(".") + ["0", "0", "0"])[:4])
    version_file = VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[
            StringFileInfo([StringTable("041204B0", [
                StringStruct("CompanyName", "ui.clover"),
                StringStruct("FileDescription", APP_NAME),
                StringStruct("FileVersion", __version__),
                StringStruct("InternalName", exe_name),
                StringStruct("OriginalFilename", f"{exe_name}.exe"),
                StringStruct("ProductName", APP_NAME),
                StringStruct("ProductVersion", __version__),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x0412, 1200])]),
        ],
    )

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=os.environ.get("MS_TARGET_ARCH") or None,
    codesign_identity=os.environ.get("MS_CODESIGN_ID") or None,
    entitlements_file=os.environ.get("MS_ENTITLEMENTS") or None,
    icon=icon,
    version=version_file,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=exe_name,
)

if IS_MAC:
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier=BUNDLE_ID,
        version=__version__,
        info_plist={
            "CFBundleName": APP_SHORT,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSApplicationCategoryType": "public.app-category.music",
            "NSDesktopFolderUsageDescription": "바탕화면의 음악 파일을 열거나 분리 결과를 저장합니다.",
            "NSDocumentsFolderUsageDescription": "문서 폴더의 음악 파일을 열거나 분리 결과를 저장합니다.",
            "NSDownloadsFolderUsageDescription": "다운로드 폴더의 음악 파일을 열거나 분리 결과를 저장합니다.",
            "NSHumanReadableCopyright": "ui.clover · Demucs © Meta Platforms (MIT)",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Audio",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": ["public.audio", "public.mp3", "com.microsoft.waveform-audio",
                                       "public.aiff-audio", "com.apple.m4a-audio", "public.movie"],
            }],
        },
    )
