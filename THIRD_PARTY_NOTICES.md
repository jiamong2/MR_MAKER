# 사용한 오픈소스와 라이선스

MR MAKER for ui.clover 배포 파일에는 다음 오픈소스가 포함됩니다. 배포·상업적 이용 전에 각 라이선스를 확인하세요.

| 구성 요소 | 용도 | 라이선스 |
|---|---|---|
| [Demucs](https://github.com/adefossez/demucs) (코드, 사전학습 모델 htdemucs / htdemucs_ft) | AI 음원 분리 (속도 우선, htdemucs_ft 보컬 모델은 배포 파일에 포함) | MIT |
| BS-RoFormer 구현 (`roformer.py`) | 고급 엔진 모델 구조 | 자체 구현. 구조는 [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) (MIT) 와 Lu et al., *Music Source Separation with Band-Split RoPE Transformer* (2023) 를 따름 |
| BS-RoFormer SW 가중치 (`BS-Roformer-SW.ckpt`, 배포 파일에 **포함하지 않음**, 처음 사용할 때 내려받음) | 품질 우선 보컬 분리 (기본) | 커뮤니티 공개 모델(jarredou 등 공유, [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 릴리스에서 내려받음). **사용 조건이 명시되어 있지 않으므로 개인 용도로만** 사용하고, 상업적 이용 전에는 권리자 확인이 필요합니다 |
| RMVPE 구현 (`rmvpe.py`) / 가중치 [`rmvpe.pt`](https://huggingface.co/lj1995/VoiceConversionWebUI) (배포 파일에 **포함하지 않음**, 실험 기능을 처음 쓸 때 내려받음) | [실험] 화음 따기 — 보컬 두 성부 음정 추적 (MIDI) | MIT ([RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 구현과 같은 구조로 자체 구현, 가중치 저장소 MIT). 논문: Wei et al., *RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music* (2023) |
| [Rubber Band Library](https://breakfastquay.com/rubberband/) 4.0 (`native/rbshift/vendor/rubberband`) | 키 조절(피치 시프트, 기본) — 별도 실행 파일 `rbshift` 로 포함 | **GPL-2.0-or-later** (rbshift 도 GPL-2.0-or-later, 소스: `native/rbshift`) |
| [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) / [python-stretch](https://github.com/gregogiudici/python-stretch) | 키 조절 대체 엔진 (rbshift 가 없을 때) | MIT |
| [soxr](https://github.com/dofuuz/python-soxr) / libsoxr | 고품질 리샘플링 | LGPL-2.1-or-later (동적 링크된 확장 모듈) |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) / [yt-dlp-ejs](https://github.com/yt-dlp/ejs) | YouTube 오디오 받기 | Unlicense / Unlicense AND MIT AND ISC |
| [Deno](https://deno.com) (PyPI `deno` 패키지의 실행 파일) | YouTube 챌린지 해석용 JavaScript 엔진 — 별도 실행 파일 | MIT (포함된 V8 등은 각자의 BSD 계열 라이선스) |
| [PyTorch](https://pytorch.org) | 딥러닝 실행 | BSD-3-Clause (CUDA 빌드는 NVIDIA CUDA 런타임 재배포 조건 포함) |
| [Qt for Python / PySide6](https://doc.qt.io/qtforpython/) | GUI | LGPLv3 (Qt 라이브러리는 동적 링크된 별도 파일로 포함되어 교체 가능) |
| [FFmpeg](https://ffmpeg.org) (imageio-ffmpeg 가 제공하는 정적 빌드) | 오디오 디코딩, MP3·FLAC 인코딩 | **GPLv3** 빌드 (`--enable-gpl --enable-version3`, libx264 · libmp3lame 포함). 별도 실행 파일로 호출됨 |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | FFmpeg 바이너리 제공 | BSD-2-Clause |
| [LAME](https://lame.sourceforge.io) / [lameenc](https://github.com/chrisstaite/lameenc) | MP3 인코딩 (ffmpeg 에 libmp3lame 이 없을 때 대체) | LGPL |
| [PortAudio](http://www.portaudio.com) / [sounddevice](https://github.com/spatialaudio/python-sounddevice) | 오디오 재생 | MIT |
| NumPy, PyYAML, einops, julius, safetensors, tqdm, psutil, certifi, truststore | 보조 라이브러리 | BSD / MIT / Apache-2.0 / MPL-2.0 |
| [Pretendard](https://github.com/orioncactus/pretendard) | 화면 글꼴 (assets/fonts) | SIL Open Font License 1.1 (assets/fonts/OFL-Pretendard.txt) |
| [PyInstaller](https://pyinstaller.org) | 실행 파일 생성 (부트로더) | GPLv2 + 예외 조항 (생성된 실행 파일은 자유롭게 배포 가능) |
| [Inno Setup](https://jrsoftware.org/isinfo.php) | Windows 설치 프로그램 | Inno Setup License |

### FFmpeg(GPL) 관련
번들된 FFmpeg 는 GPLv3 조건으로 배포됩니다. 프로그램을 외부에 배포한다면 FFmpeg 소스 코드 입수 방법을 함께 안내해야 합니다
(https://ffmpeg.org/download.html, imageio-ffmpeg 빌드 정보: https://github.com/imageio/imageio-binaries).
GPL 구성 요소를 피하고 싶다면 LGPL 로 빌드한 ffmpeg 로 교체하고 `MUSIC_SEPARATOR_FFMPEG` 환경 변수나
`audio_io.find_ffmpeg()` 가 그 파일을 찾도록 수정하면 됩니다. 디코딩·FLAC 은 LGPL 빌드로 충분하며,
MP3 는 LGPL 빌드에 libmp3lame(LGPL)을 포함하거나, 없으면 자동으로 lameenc 로 인코딩합니다.

### Rubber Band(GPL) 관련
키 조절 도구 `rbshift` 는 Rubber Band Library(GPL-2.0-or-later)를 포함하는 별도 프로그램으로, 프로그램 본체와는 표준 입출력으로만
데이터를 주고받습니다 (FFmpeg 와 같은 방식). 소스 코드 전체가 이 저장소의 `native/rbshift` 에 있으며 GPL-2.0-or-later 로 배포됩니다.
GPL 구성 요소 없이 배포하려면 `scripts/build.py --skip-rbshift` 로 빌드하세요 (키 조절은 Signalsmith Stretch 로 대체).
Rubber Band 를 GPL 이 아닌 조건으로 쓰려면 Particular Programs Ltd. 의 상용 라이선스가 필요합니다.

### YouTube 관련
yt-dlp 는 영상 사이트에서 미디어를 받는 범용 도구입니다. 이 프로그램은 로그인·쿠키·DRM 우회 기능을 쓰지 않으며,
사용자가 권리를 가졌거나 다운로드가 허락된 콘텐츠만 처리하도록 안내합니다. 각 콘텐츠의 이용 조건과 YouTube 서비스 약관은 사용자가 지켜야 합니다.

### 상업적 이용
속도 우선 엔진(Demucs)과 앱 코드는 위 라이선스 조건을 지키면 상업적으로 이용할 수 있습니다.
품질 우선 엔진(BS-RoFormer SW)은 가중치의 사용 조건이 명시되어 있지 않으므로 상업적 용도로는 사용하지 마세요
(음악 모임 안에서 개인 연습용으로 쓰는 것을 전제로 합니다).

### 분리 결과물
분리된 음원과 MR 의 저작권은 원곡 권리자에게 있습니다.
