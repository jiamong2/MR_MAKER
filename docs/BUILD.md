# 개발 · 빌드 · 배포 가이드

이 문서는 Music Separator 를 수정하고 다시 빌드·배포하는 방법을 설명합니다.

- [1. 구조 한눈에 보기](#1-구조-한눈에-보기)
- [2. 소스에서 실행하기](#2-소스에서-실행하기)
- [3. 테스트](#3-테스트)
- [4. GitHub Actions 로 Windows·macOS 자동 빌드 (권장)](#4-github-actions-로-windowsmacos-자동-빌드-권장)
- [5. 내 컴퓨터에서 직접 빌드](#5-내-컴퓨터에서-직접-빌드)
- [6. macOS 코드 서명과 공증](#6-macos-코드-서명과-공증)
- [7. Windows 코드 서명](#7-windows-코드-서명)
- [8. 자주 하는 수정](#8-자주-하는-수정)
- [9. 버전 선택 이유와 제약](#9-버전-선택-이유와-제약)

---

## 1. 구조 한눈에 보기

```
music-separator/
├─ run.py                         진입점 (소스 실행 / PyInstaller 공용)
├─ src/music_separator/
│  ├─ app.py                      앱 시작, 로깅, 명령행 옵션(--selftest, --gui-smoke)
│  ├─ engine.py                   Demucs 분리 엔진: 모드 정의, GPU→CPU 폴백, 진행률, 취소
│  ├─ models.py                   모델 탐색/다운로드(HuggingFace → Meta 서버)/로드/캐시
│  ├─ audio_io.py                 FFmpeg 디코딩, WAV/MP3 저장
│  ├─ player.py                   멀티트랙 믹서(뮤트/솔로/볼륨) + sounddevice 재생
│  ├─ dsp.py                      파형 피크 계산, 믹스다운(MR)
│  ├─ system.py                   GPU 감지, 메모리/디스크 요구량 계산
│  ├─ paths.py                    OS 별 데이터/캐시/리소스 경로
│  ├─ selftest.py / smoke.py      빌드 결과물 자동 검증 (CI 에서 사용)
│  └─ ui/                         PySide6 화면 (main_window, widgets, workers, theme)
├─ tests/                         pytest 테스트 (GPU·인터넷·오디오 장치 없이 실행)
├─ requirements/                  플랫폼별 의존성 (torch 는 OS 마다 다름)
├─ packaging/
│  ├─ music_separator.spec        PyInstaller 설정
│  ├─ windows/installer.iss       Inno Setup 설치 프로그램
│  └─ macos/                      DMG 생성, 서명·공증 스크립트, entitlements
├─ scripts/
│  ├─ build.py                    빌드 + 패키징 (로컬/CI 공용)
│  ├─ fetch_models.py             모델 미리 받기 (번들/오프라인용)
│  └─ make_icons.py               아이콘 생성
├─ assets/                        아이콘, 체크 이미지
└─ .github/workflows/build.yml    자동 빌드 · 테스트 · 릴리스
```

동작 흐름: 파일 열기 → `LoadThread` 가 FFmpeg 로 44.1kHz 스테레오 float32 로 디코딩해 캐시 폴더에 `.npy` 저장 →
`SeparationThread` 가 (필요시 모델 다운로드 후) `engine.separate()` 실행 → 결과 트랙을 `.npy` 로 저장하고
메모리 매핑(memmap)으로 재생/저장 → `ExportThread` 가 WAV/MP3 로 저장. 무거운 작업은 모두 별도 스레드라 UI 가 멈추지 않습니다.

## 2. 소스에서 실행하기

Python **3.11** 을 권장합니다 (3.10~3.12 동작).

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1) PyTorch — 내 환경에 맞는 것 하나
pip install -r requirements/torch-macos-arm64.txt     # Apple Silicon Mac
pip install -r requirements/torch-macos-intel.txt     # Intel Mac
pip install -r requirements/torch-windows-cpu.txt     # Windows (CPU)
pip install -r requirements/torch-windows-cuda.txt    # Windows (NVIDIA GPU)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # Linux (CPU)

# 2) 앱 의존성
pip install -r requirements/common.txt -r requirements/dev.txt
pip install --no-deps -r requirements/demucs.txt

# 3) 실행
python run.py
python run.py "노래.mp3"            # 파일을 열면서 시작
```

유용한 환경 변수

| 변수 | 용도 |
|---|---|
| `MUSIC_SEPARATOR_MODELS_DIR` | 추가 모델 폴더 (예: 공유 폴더) |
| `MUSIC_SEPARATOR_DATA_DIR` / `MUSIC_SEPARATOR_CACHE_DIR` | 사용자 데이터/캐시 위치 변경 |
| `HF_ENDPOINT` | HuggingFace 미러 주소 (사내 미러 등) |
| `MUSIC_SEPARATOR_FFMPEG` | 다른 ffmpeg 실행 파일 사용 |
| `MUSIC_SEPARATOR_FORCE_MODEL=demucs_unittest` | 초소형 테스트 모델로 UI 흐름 확인 (분리 품질 없음) |

## 3. 테스트

```bash
pytest -q tests                                        # 19개 핵심 테스트
python run.py --selftest --report selftest.json        # 실제 모델로 분리→저장 전 과정 점검 (모델 다운로드)
python run.py --selftest --model demucs_unittest       # 모델 없이 빠른 점검
QT_QPA_PLATFORM=offscreen python run.py --gui-smoke shot.png   # GUI 자동 조작 + 스크린샷
```

`--selftest` 는 FFmpeg 디코딩, 모델 로드, 분리, 각 형식 저장, 저장 파일 재검증,
그리고 실제 모델일 때 *분리된 트랙의 합 ≈ 원본* 인지까지 확인합니다. 빌드된 실행 파일에도 같은 옵션이 들어 있습니다.

## 4. GitHub Actions 로 Windows·macOS 자동 빌드 (권장)

Windows 실행 파일은 Windows 에서, macOS 앱은 macOS 에서만 만들 수 있으므로(PyInstaller 는 교차 빌드를 지원하지 않음)
GitHub 이 제공하는 Windows·macOS 가상 머신에서 빌드합니다.

1. GitHub 에 새 저장소를 만들고 이 폴더 전체를 push 합니다.
   ```bash
   cd music-separator
   git init && git add . && git commit -m "Music Separator 1.0.0"
   git branch -M main
   git remote add origin https://github.com/<아이디>/music-separator.git
   git push -u origin main
   ```
2. push 하면 **Actions** 탭에서 `Build` 워크플로가 자동 실행됩니다. 직접 실행하려면 *Actions → Build → Run workflow*.
3. 완료되면 실행 결과 페이지 아래 **Artifacts** 에서 받습니다.
   - `MusicSeparator-windows-x64-cpu` / `-cuda` / `macos-arm64` / `macos-x86_64` : 설치 파일
   - `test-…` : 각 OS 에서 **빌드된 실행 파일로 실제 실행한** 자가 진단 결과(JSON)와 GUI 스크린샷
4. 정식 배포: 버전 태그를 push 하면 빌드 후 GitHub Release(초안)에 파일이 올라갑니다.
   ```bash
   git tag v1.0.0 && git push origin v1.0.0
   ```

워크플로가 하는 일

| 작업 | 내용 |
|---|---|
| test | Linux 에서 단위 테스트 + 오프스크린 GUI 테스트 |
| models | htdemucs·htdemucs_6s 모델 다운로드(캐시) |
| build ×4 | Windows CPU / Windows CUDA / macOS arm64(macos-14) / macOS Intel(macos-15-intel) 빌드 → 각 OS 에서 단위 테스트, 빌드된 앱 `--selftest`(실제 모델로 분리), `--gui-smoke`, Windows 는 설치 프로그램 무인 설치 후 재검증, macOS 는 DMG 마운트 후 재검증 |
| release | 태그일 때 Release 초안 생성 (2GB 초과 파일은 제외) |

**Run workflow 옵션**: AI 모델 포함 여부, CUDA 빌드 여부와 CUDA 버전(cu126/cu128/cu130), Intel Mac 빌드 여부.

참고
- GitHub 러너에는 GPU 가 없으므로 CI 테스트는 CPU 경로로 실행됩니다. GPU 가속은 실제 PC 에서 확인해야 합니다.
- **비공개 저장소 + 무료 플랜**은 Artifacts 저장 용량이 작아(500MB) 업로드가 실패할 수 있습니다.
  공개 저장소로 하거나, 태그를 push 해 Release 로 받으세요 (Release 파일은 이 용량에 포함되지 않음).
- CUDA 빌드는 PyTorch CUDA 라이브러리 때문에 2~3GB 입니다. 설치 파일은 `setup.exe` + `setup-1.bin` … 으로 나뉘며 같은 폴더에 두고 실행합니다.
- Intel Mac 러너(`macos-15-intel`)는 GitHub 이 2027년 가을까지 제공 예정입니다.

## 5. 내 컴퓨터에서 직접 빌드

2장의 환경을 만든 뒤:

```bash
python scripts/fetch_models.py build/models            # (선택) 모델 미리 받기 ≈135MB
python scripts/build.py --bundle-models build/models   # 모델 포함 빌드
python scripts/build.py                                # 모델 미포함 (첫 실행 시 다운로드)
```

결과물은 `dist/release/` 에 생깁니다.

- **Windows**: [Inno Setup 6](https://jrsoftware.org/isdl.php) 이 설치되어 있으면 설치 프로그램까지 만들고, 없으면 portable zip 만 만듭니다.
  CUDA 빌드는 `torch-windows-cuda.txt` 로 torch 를 설치한 뒤 `python scripts/build.py --variant cuda`.
- **macOS**: Xcode Command Line Tools 필요(`xcode-select --install`). Apple Silicon Mac 에서 빌드하면 arm64,
  Intel Mac 에서 빌드하면 x86_64 앱이 만들어집니다. (torch 가 범용(universal2) 빌드를 제공하지 않아 아키텍처별로 따로 빌드합니다.)

빌드 후 확인:
```bash
dist/MusicSeparator/MusicSeparator --selftest                               # Linux
"dist/Music Separator.app/Contents/MacOS/MusicSeparator" --selftest         # macOS
dist\MusicSeparator\MusicSeparator.exe --selftest --report r.json           # Windows (창 모드 앱이라 결과는 JSON 으로)
```

## 6. macOS 코드 서명과 공증

서명·공증 없이 배포하면 다른 Mac 에서 "확인되지 않은 개발자" 경고가 나옵니다 (Control+클릭 → 열기 로 우회 가능).
경고를 없애려면 **Apple Developer Program**(유료, 연간) 가입 후 *Developer ID Application* 인증서로 서명하고 Apple 공증을 받아야 합니다.

### 준비
1. Xcode → Settings → Accounts 또는 developer.apple.com 에서 **Developer ID Application** 인증서를 만들고,
   키체인 접근에서 인증서+개인키를 `.p12` 로 내보냅니다 (암호 설정).
2. 공증용 인증 정보 (둘 중 하나)
   - Apple ID + **앱 암호**(appleid.apple.com → 로그인 및 보안 → 앱 암호) + Team ID
   - App Store Connect **API 키**(.p8, Key ID, Issuer ID)

### GitHub Actions 에서 자동 서명·공증
저장소 *Settings → Secrets and variables → Actions* 에 등록:

| Secret | 값 |
|---|---|
| `MACOS_CERT_P12_BASE64` | `base64 -i cert.p12 \| pbcopy` 결과 |
| `MACOS_CERT_PASSWORD` | p12 암호 |
| `MACOS_CODESIGN_ID` | `Developer ID Application: 이름 (TEAMID)` (`security find-identity -v -p codesigning` 으로 확인) |
| `APPLE_ID` / `APPLE_TEAM_ID` / `APPLE_APP_PASSWORD` | 공증용 |

등록하면 워크플로가 자동으로: 임시 키체인에 인증서 등록 → 앱 내부의 모든 Mach-O 파일(파이썬 확장, dylib, ffmpeg)과
프레임워크를 Hardened Runtime 으로 서명 → 앱 공증 + 스테이플 → DMG 생성·서명 → DMG 공증 + 스테이플.
(`packaging/macos/sign_and_notarize.sh`)

### 로컬에서
```bash
export MS_CODESIGN_ID="Developer ID Application: Hong Gildong (ABCDE12345)"
export APPLE_ID=you@example.com APPLE_TEAM_ID=ABCDE12345 APPLE_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
python scripts/build.py --bundle-models build/models
# 확인
spctl -a -vv "dist/Music Separator.app"         # → accepted, source=Notarized Developer ID
xcrun stapler validate dist/release/*.dmg
```

`packaging/macos/entitlements.plist` 의 예외 3가지(JIT, 서명 안 된 실행 메모리, 라이브러리 검증 해제)는
PyTorch·Python 확장 모듈 로드에 필요합니다. 공증이 거부되면 `xcrun notarytool log <제출 ID> ...` 로 원인을 확인하세요.

## 7. Windows 코드 서명

서명하지 않은 설치 파일은 SmartScreen 경고("Windows의 PC 보호")가 나옵니다. 사용자는 *추가 정보 → 실행* 으로 설치할 수 있습니다.

- **방법 A — Azure Artifact Signing** (구 Trusted Signing, Microsoft 의 클라우드 서명 서비스, 2026년 1월 정식 출시).
  인증서 파일을 직접 관리하지 않아 CI 에 가장 적합합니다. 자격 요건·요금은
  [제품 페이지](https://azure.microsoft.com/products/artifact-signing)를 확인하세요.
  워크플로의 `Build` 단계 전에 Microsoft 가 제공하는 서명 액션을 추가해 `dist/MusicSeparator/MusicSeparator.exe` 와 설치 파일을 서명하면 됩니다.
- **방법 B — .pfx 인증서 파일**: Secrets 에 `WINDOWS_CERT_PFX_BASE64`, `WINDOWS_CERT_PASSWORD` 를 등록하면
  `scripts/build.py` 가 signtool 로 exe 와 설치 파일을 서명합니다.
  (최근 발급되는 코드 서명 인증서는 하드웨어 토큰/HSM 보관이 의무라 .pfx 로 내보낼 수 없는 경우가 많습니다.)

서명을 해도 SmartScreen 평판은 다운로드 수가 쌓이면서 생기므로 초기에는 경고가 남을 수 있습니다.

## 8. 자주 하는 수정

| 하고 싶은 것 | 수정할 곳 |
|---|---|
| 버전 올리기 | `src/music_separator/__init__.py` 의 `__version__` (설치 파일 이름·앱 정보에 자동 반영) → 태그 `vX.Y.Z` push |
| 분리 모드/모델 바꾸기 | `engine.py` 의 `MODES` (모델, 출력 트랙), 새 모델은 `models.py` 의 `MODELS` 에 시그니처 추가 |
| 트랙 이름·색상 | `engine.py` 의 `STEM_STYLE` |
| 화면 색·글꼴 | `ui/theme.py` (색 상수와 스타일시트) |
| 화면 배치 | `ui/main_window.py` (`_build_control_bar`, `_build_inspector`, `_build_main_area`, `_build_export_bar`) |
| 저장 형식 추가 | `audio_io.py` 의 `EXPORT_FORMATS`, `write_audio()` |
| 아이콘 | `assets/icon.png`(1024×1024) 교체 후 `python scripts/make_icons.py --from-png` |
| 번들에 넣을 모델 | `.github/workflows/build.yml` 의 models 작업 (`fetch_models.py` 인자, 캐시 키) |
| 설치 프로그램 설정 | `packaging/windows/installer.iss` (설치 경로, 바로가기, 권한) |
| 포함/제외 모듈 | `packaging/music_separator.spec` 의 `hiddenimports` / `excludes` |

수정 후 순서: `pytest -q tests` → `python run.py` 로 확인 → push (CI 가 4개 OS 빌드와 실제 실행 테스트) → 태그로 릴리스.

## 9. 버전 선택 이유와 제약

| 항목 | 선택 | 이유 |
|---|---|---|
| Python | 3.11 | 모든 대상(Intel Mac 의 torch 2.2.2 포함) 휠이 있는 공통 버전 |
| PyTorch (Win, Apple Silicon) | 2.11.0 | torch 2.12 부터 macOS arm64 휠이 macOS 14+ 전용 → 2.11 로 macOS 11+ 유지 |
| PyTorch (Intel Mac) | 2.2.2 | PyTorch 가 Intel Mac 을 지원한 마지막 버전 (numpy < 2 필요) |
| PySide6-Essentials | 6.9.3 | macOS 12+ 지원 (6.10 은 13+, 6.12 는 14+). QtMultimedia 등 큰 모듈 제외로 용량 절감 |
| Demucs | 4.1.0 (`--no-deps`) | 진행률 콜백·취소 지원. 사용하지 않는 `sphn`/`huggingface-hub` 의존성 제외 |
| 재생 | sounddevice(PortAudio) | Windows/macOS 휠에 PortAudio 포함, 콜백 기반 저지연 멀티트랙 믹싱 |
| FFmpeg | imageio-ffmpeg | OS 별 정적 ffmpeg 바이너리를 휠로 제공 → 번들 간단 |

최소 사양: Windows 10 1809+ / 11 (x64), macOS 12+, RAM 8GB 권장(4분 곡 기준 약 3GB 사용), 디스크 여유 2GB 이상.

GPU 동작 방식: 시작 시 CUDA(NVIDIA) 또는 MPS(Apple)를 감지해 자동 사용합니다. GPU 처리 중 오류, GPU 메모리 부족,
또는 결과에 NaN 이 생기면 모델을 다시 불러와 CPU 로 자동 재처리하고 화면에 안내합니다. 설정에서 CPU 를 강제로 고를 수도 있습니다.
