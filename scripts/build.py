"""배포 파일 빌드 스크립트 (로컬 / GitHub Actions 공용).

사용 (프로젝트 루트에서, 빌드용 가상환경 활성화 후):
  python scripts/build.py                          # 현재 OS 용 빌드
  python scripts/build.py --bundle-models build/models   # AI 모델 포함 빌드 (오프라인 즉시 사용)
  python scripts/build.py --variant cuda           # (Windows) 이름표만 cuda 로 — torch CUDA 빌드는 미리 설치
  python scripts/build.py --edition separator      # MR SEPARATOR (6-Stem 에디션). 기본은 maker (MR MAKER)

결과물: dist/release/
  Windows : <MRMaker|MRSeparator>-<ver>-windows-x64-<variant>-setup.exe, ...-portable.zip, ....msi
  macOS   : <MRMaker|MRSeparator>-<ver>-macos-<arch>.dmg
  Linux   : <MRMaker|MRSeparator>-<ver>-linux-x86_64.tar.gz  (개발 확인용)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _edition_from_args() -> str:
    """--edition 은 프로그램 이름 등을 정하는 값이라 music_separator 를 import 하기 전에 읽는다."""
    e = os.environ.get("MS_EDITION") or os.environ.get("MUSIC_SEPARATOR_EDITION") or "maker"
    for i, a in enumerate(sys.argv):
        if a == "--edition" and i + 1 < len(sys.argv):
            e = sys.argv[i + 1]
        elif a.startswith("--edition="):
            e = a.split("=", 1)[1]
    if e not in ("maker", "separator"):
        raise SystemExit(f"알 수 없는 에디션: {e} (maker / separator)")
    return e


os.environ["MUSIC_SEPARATOR_EDITION"] = _edition_from_args()
from music_separator import (APP_NAME, APP_SHORT, EDITION, EXE_NAME, ICON_STEM, SETTINGS_ID,  # noqa: E402
                             __version__)

# 설치 프로그램 식별자 (에디션마다 달라야 두 프로그램을 함께 설치할 수 있음). maker 값은 기존 설치와 같아야 업데이트됨.
INSTALLER_IDS = {
    "maker": {"inno_app_id": "6B0E4C1A-3B7D-4F8E-9C2A-7D51E0A9F3B4",
              "msi_upgrade": "6FA73F04-BFA9-4E99-8B94-39B539B8F37D",
              "msi_startmenu": "3E7B1C52-6A0D-4C7E-9E1B-2F4A8D6C0B11",
              "msi_desktop": "8C2D4F61-1B3E-4A9C-B7D5-6E0F2A1C9D22",
              "reg_installer": r"Software\MusicSeparator\Installer",
              "reg_settings": r"Software\MusicSeparator",
              "msi_assets": ""},
    "separator": {"inno_app_id": "B4D9A2E7-5C1F-4E36-A8B0-3F6D7C2E91A5",
                  "msi_upgrade": "D2C8F5A1-7E43-4B9A-9F16-5A0B3E8D4C72",
                  "msi_startmenu": "A91E6D34-2F7B-4C58-8D03-7B5E1F9A6C40",
                  "msi_desktop": "5F3A8C19-D6E2-47B1-B4A7-0C9D2E6F8A53",
                  "reg_installer": r"Software\MusicSeparator\MRSeparatorInstaller",
                  "reg_settings": r"Software\MusicSeparator\MRSeparator",
                  "msi_assets": "separator"},
}[EDITION]

DIST = ROOT / "dist"
RELEASE = DIST / "release"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build_rbshift() -> Path:
    """키 조절 도구(Rubber Band, native/rbshift) 컴파일 → build/rbshift."""
    out = ROOT / "build" / "rbshift"
    arch = os.environ.get("MS_TARGET_ARCH")
    extra = [f"-DCMAKE_OSX_ARCHITECTURES={arch}"] if sys.platform == "darwin" and arch else []
    run(["cmake", "-S", ROOT / "native" / "rbshift", "-B", out, "-DCMAKE_BUILD_TYPE=Release", *extra])
    run(["cmake", "--build", out, "--config", "Release", "--parallel"])
    exe = "rbshift.exe" if sys.platform == "win32" else "rbshift"
    for cand in (out / exe, out / "Release" / exe):
        if cand.is_file():
            run([cand, "--version"])
            return cand
    raise SystemExit("rbshift 빌드 결과를 찾지 못했습니다")


def pyinstaller(bundle_models: str | None, variant: str = "cpu"):
    # 빌드 종류(cpu / cuda / arm64 / x86_64)를 앱 안에 기록 → 기본 처리 방식 결정에 사용
    info = ROOT / "build" / "build_info.json"
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(json.dumps({"variant": variant, "edition": EDITION}), encoding="utf-8")
    env = dict(os.environ)
    if bundle_models:
        env["MS_BUNDLE_MODELS"] = str(Path(bundle_models).resolve())
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", DIST, "--workpath", ROOT / "build" / "pyinstaller",
         ROOT / "packaging" / "music_separator.spec"], env=env, cwd=ROOT)


def find_iscc() -> str | None:
    cands = [shutil.which("iscc"), shutil.which("ISCC")]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.environ.get("LOCALAPPDATA") and str(Path(os.environ["LOCALAPPDATA"]) / "Programs")):
        if base:
            cands.append(str(Path(base) / "Inno Setup 6" / "ISCC.exe"))
    return next((c for c in cands if c and Path(c).exists()), None)


def find_signtool() -> str | None:
    if shutil.which("signtool"):
        return shutil.which("signtool")
    kits = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin"
    found = sorted(kits.glob("*/x64/signtool.exe"))
    return str(found[-1]) if found else None


def win_sign(files: list[Path]) -> None:
    """MS_WIN_SIGN_PFX(+MS_WIN_SIGN_PASSWORD) 가 있으면 Authenticode 서명."""
    pfx = os.environ.get("MS_WIN_SIGN_PFX")
    if not pfx or not Path(pfx).exists():
        return
    tool = find_signtool()
    if not tool:
        print("!! signtool.exe 를 찾지 못해 서명을 건너뜁니다")
        return
    for f in files:
        run([tool, "sign", "/f", pfx, "/p", os.environ.get("MS_WIN_SIGN_PASSWORD", ""), "/fd", "sha256",
             "/tr", os.environ.get("MS_WIN_TIMESTAMP_URL", "http://timestamp.digicert.com"), "/td", "sha256",
             f])


def package_windows(variant: str, skip_installer: bool):
    src = DIST / EXE_NAME
    win_sign([src / f"{EXE_NAME}.exe"])
    # CUDA 버전(약 5GB)은 MSI(+분할 .cab)만 만든다.
    #  - 포터블 zip: 2GB 를 넘어 GitHub 릴리스에 올릴 수 없음 (예전엔 만들었다가 버려졌음)
    #  - setup.exe: MSI 로 대체. 가장 느린 압축 단계라 빌드 시간을 크게 줄임
    light = variant == "cuda"
    if not light:
        zip_path = RELEASE / f"{EXE_NAME}-{__version__}-windows-x64-{variant}-portable.zip"
        print(f"portable zip -> {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for f in src.rglob("*"):
                if f.is_file():
                    z.write(f, Path(EXE_NAME) / f.relative_to(src))
    if skip_installer:
        return
    if not light:
        iscc = find_iscc()
        if not iscc:
            print("!! Inno Setup(ISCC.exe)이 없어 설치 프로그램은 건너뜁니다. https://jrsoftware.org/isdl.php")
        else:
            run([iscc, f"/DAppVersion={__version__}", f"/DSourceDir={src}", f"/DVariant={variant}",
                 f"/DOutputDir={RELEASE}", f"/DAppName={APP_NAME}", f"/DAppShort={APP_SHORT}",
                 f"/DExeName={EXE_NAME}", f"/DAppGuid={INSTALLER_IDS['inno_app_id']}",
                 f"/DIconFile={ROOT / 'assets' / (ICON_STEM + '.ico')}",
                 f"/DSettingsKey={INSTALLER_IDS['reg_settings']}",
                 ROOT / "packaging" / "windows" / "installer.iss"])
            win_sign(sorted(RELEASE.glob(f"{EXE_NAME}-{__version__}-windows-x64-{variant}-setup*.exe")))
    build_msi(src, variant)


def build_msi(src: Path, variant: str):
    """WiX v5 로 MSI 설치 파일 생성. MSI 한 파일은 2GB 를 넘을 수 없어 CUDA 버전은 외부 .cab 여러 개로 나눔."""
    wix = shutil.which("wix")
    if not wix:
        print("!! WiX Toolset(wix)이 없어 MSI 는 건너뜁니다. dotnet tool install --global wix --version 5.0.2")
        return
    base = f"{EXE_NAME}-{__version__}-windows-x64-{variant}"
    out = RELEASE / f"{base}.msi"
    msi_dir = (ROOT / "packaging" / "windows" / "msi").resolve()
    args = [wix, "build", msi_dir / "MRMaker.wxs", "-arch", "x64", "-ext", "WixToolset.UI.wixext", "-culture", "ko-KR",
            "-d", f"Version={__version__}", "-d", f"SourceDir={src.resolve()}",
            "-d", f"IconPath={(ROOT / 'assets' / (ICON_STEM + '.ico')).resolve()}",
            "-d", f"AssetsDir={msi_dir / INSTALLER_IDS['msi_assets'] if INSTALLER_IDS['msi_assets'] else msi_dir}",
            "-d", f"AppName={APP_NAME}", "-d", f"AppShort={APP_SHORT}", "-d", f"ExeName={EXE_NAME}",
            "-d", f"UpgradeCode={INSTALLER_IDS['msi_upgrade']}",
            "-d", f"StartMenuGuid={INSTALLER_IDS['msi_startmenu']}",
            "-d", f"DesktopGuid={INSTALLER_IDS['msi_desktop']}",
            "-d", f"InstallerKey={INSTALLER_IDS['reg_installer']}"]
    size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    if size > 1800 * 1048576:     # 압축해도 2GB 를 넘을 수 있는 크기 → 외부 cab 으로 분할
        args += ["-d", "EmbedCab=no", "-d", "MediaSize=1800", "-d", f"CabPrefix={base}-", "-d", "Compression=medium"]
    run(args + ["-o", out])
    out.with_suffix(".wixpdb").unlink(missing_ok=True)      # 디버그용 파일은 릴리스에 올리지 않음
    print(f"msi -> {out} ({out.stat().st_size / 1048576:.0f} MB)")
    win_sign([out])


def package_macos():
    app = DIST / f"{APP_NAME}.app"
    arch = os.environ.get("MS_TARGET_ARCH") or platform.machine()
    dmg = RELEASE / f"{EXE_NAME}-{__version__}-macos-{arch}.dmg"
    # 데이터로 포함된 ffmpeg 에 실행 권한 보장
    for f in list(app.rglob("ffmpeg-*")) + list(app.rglob("rbshift")) + list(app.rglob("deno")):
        if f.is_file():
            f.chmod(0o755)
    if os.environ.get("MS_CODESIGN_ID"):
        run(["bash", ROOT / "packaging" / "macos" / "sign_and_notarize.sh", app, dmg])
    else:
        print("서명 인증서 없음 → ad-hoc 서명 (다른 Mac 에서 처음 실행 시 '확인되지 않은 개발자' 경고)")
        run(["codesign", "--force", "--deep", "--sign", "-", app])
        run(["bash", ROOT / "packaging" / "macos" / "make_dmg.sh", app, dmg, APP_NAME])


def package_linux():
    src = DIST / EXE_NAME
    out = RELEASE / f"{EXE_NAME}-{__version__}-linux-{platform.machine()}.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        t.add(src, arcname=EXE_NAME)
    print(f"tarball -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-models", help="포함할 모델 폴더 (scripts/fetch_models.py 결과)")
    ap.add_argument("--variant", default=os.environ.get("MS_VARIANT", "cpu"), help="Windows: cpu / cuda")
    ap.add_argument("--edition", default=EDITION, choices=["maker", "separator"],
                    help="maker = MR MAKER (2-Stem, 기본) / separator = MR SEPARATOR (6-Stem)")
    ap.add_argument("--skip-installer", action="store_true")
    ap.add_argument("--skip-pyinstaller", action="store_true")
    ap.add_argument("--skip-rbshift", action="store_true", help="키 조절 도구 빌드 생략 (이미 빌드됨)")
    a = ap.parse_args()
    RELEASE.mkdir(parents=True, exist_ok=True)
    print(f"edition: {EDITION} → {APP_NAME} ({EXE_NAME}, 설정 {SETTINGS_ID})", flush=True)
    if not a.skip_pyinstaller:
        if not a.skip_rbshift:
            build_rbshift()
        pyinstaller(a.bundle_models, a.variant)
    if sys.platform == "win32":
        package_windows(a.variant, a.skip_installer)
    elif sys.platform == "darwin":
        package_macos()
    else:
        package_linux()
    print("\n결과물:")
    for f in sorted(RELEASE.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1048576:.0f} MB)")


if __name__ == "__main__":
    main()
