# 사용한 오픈소스와 라이선스

MR MAKER for ui.clover 배포 파일에는 다음 오픈소스가 포함됩니다. 배포·상업적 이용 전에 각 라이선스를 확인하세요.

| 구성 요소 | 용도 | 라이선스 |
|---|---|---|
| [Demucs](https://github.com/adefossez/demucs) (코드, 사전학습 모델 htdemucs / htdemucs_ft / htdemucs_6s) | AI 음원 분리 | MIT |
| BS-RoFormer 구현 (`roformer.py`) | 고급 엔진 모델 구조 | 자체 구현. 구조는 [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) (MIT) 와 Lu et al., *Music Source Separation with Band-Split RoPE Transformer* (2023) 를 따름 |
| BS-RoFormer SW 가중치 (`BS-Roformer-SW.ckpt`, 배포 파일에 **포함하지 않음**) | 고급 엔진 6-Stem 분리 | 커뮤니티 공개 모델(jarredou 등 공유, [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 릴리스에서 내려받음). **사용 조건이 명시되어 있지 않으므로 개인 용도로만** 사용하고, 상업적 이용 전에는 권리자 확인이 필요합니다 |
| [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) / [python-stretch](https://github.com/gregogiudici/python-stretch) | 키 조절(피치 시프트) | MIT |
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

### 상업적 이용
기본 엔진(Demucs)과 앱 코드는 위 라이선스 조건을 지키면 상업적으로 이용할 수 있습니다.
고급 엔진(BS-RoFormer SW)은 가중치의 사용 조건이 명시되어 있지 않으므로 상업적 용도로는 사용하지 마세요.

### 분리 결과물
분리된 음원과 MR 의 저작권은 원곡 권리자에게 있습니다.
