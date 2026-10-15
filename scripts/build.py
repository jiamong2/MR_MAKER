"""배포 파일 빌드 스크립트 (로컬 / GitHub Actions 공용).

사용 (프로젝트 루트에서, 빌드용 가상환경 활성화 후):
  python scripts/build.py                          # 현재 OS 용 빌드
  python scripts/build.py --bundle-models build/models   # AI 모델 포함 빌드 (오프라인 즉시 사용)
  python scripts/build.py --variant cuda           # (Windows) 이름표만 cuda 로 — torch CUDA 빌드는 미리 설치

결과물: dist/release/
  Windows : MRMaker-<ver>-windows-x64-<variant>-setup.exe, ...-portable.zip
  macOS   : MRMaker-<ver>-macos-<arch>.dmg
  Linux   : MRMaker-<ver>-linux-x86_64.tar.gz  (개발 확인용)
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
from music_separator import APP_NAME, EXE_NAME, __version__  # noqa: E402

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
    info.write_text(json.dumps({"variant": variant}), encoding="utf-8")
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
    zip_path = RELEASE / f"{EXE_NAME}-{__version__}-windows-x64-{variant}-portable.zip"
    print(f"portable zip -> {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, Path(EXE_NAME) / f.relative_to(src))
    if skip_installer:
        return
    iscc = find_iscc()
    if not iscc:
        print("!! Inno Setup(ISCC.exe)이 없어 설치 프로그램은 건너뜁니다. https://jrsoftware.org/isdl.php")
        return
    run([iscc, f"/DAppVersion={__version__}", f"/DSourceDir={src}", f"/DVariant={variant}",
         f"/DOutputDir={RELEASE}", ROOT / "packaging" / "windows" / "installer.iss"])
    win_sign(sorted(RELEASE.glob(f"{EXE_NAME}-{__version__}-windows-x64-{variant}-setup*.exe")))
    if variant == "cpu":
        build_msi(src, variant)


def build_msi(src: Path, variant: str):
    """WiX v5 로 MSI 설치 파일 생성 (CPU 버전만: MSI 한 파일은 2GB 를 넘을 수 없어 CUDA 는 제외)."""
    wix = shutil.which("wix")
    if not wix:
        print("!! WiX Toolset(wix)이 없어 MSI 는 건너뜁니다. dotnet tool install --global wix --version 5.0.2")
        return
    out = RELEASE / f"{EXE_NAME}-{__version__}-windows-x64-{variant}.msi"
    run([wix, "build", ROOT / "packaging" / "windows" / "msi" / "MRMaker.wxs", "-arch", "x64",
         "-d", f"Version={__version__}", "-d", f"SourceDir={src.resolve()}",
         "-d", f"IconPath={(ROOT / 'assets' / 'icon.ico').resolve()}", "-o", out])
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
    ap.add_argument("--skip-installer", action="store_true")
    ap.add_argument("--skip-pyinstaller", action="store_true")
    ap.add_argument("--skip-rbshift", action="store_true", help="키 조절 도구 빌드 생략 (이미 빌드됨)")
    a = ap.parse_args()
    RELEASE.mkdir(parents=True, exist_ok=True)
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
