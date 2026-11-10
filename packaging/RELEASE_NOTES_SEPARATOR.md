## MR SEPARATOR for ui.clover

곡을 **보컬 · 드럼 · 베이스 · 기타 · 피아노 · 나머지 악기** 6개 트랙으로 나눠, 듣고 싶은 악기만 골라 듣거나 원하는 조합으로 저장하는 프로그램입니다.
악기를 연주하거나 카피·편곡을 하는 분들을 위한 버전입니다. (보컬 / 반주만 필요하면 [MR MAKER](https://github.com/jiamong2/MR_MAKER/releases/latest)를 쓰세요. 두 프로그램은 함께 설치할 수 있고, 내려받은 AI 모델도 함께 씁니다.)

사용 방법: [MR SEPARATOR 사용 설명서 (PDF)](https://github.com/jiamong2/MR_MAKER/blob/main/docs/MR_SEPARATOR_사용설명서.pdf)

### 설치 파일 선택
| 컴퓨터 | 파일 |
|---|---|
| Windows, NVIDIA 그래픽카드(RTX 20·GTX 16 시리즈 이상) | `MRSeparator-…-windows-x64-cuda.msi` + 같은 이름으로 시작하는 `.cab` 파일들을 같은 폴더에 받은 뒤 .msi 실행 |
| Windows, 그 외 | `MRSeparator-…-windows-x64-cpu.msi` (권장) 또는 `MRSeparator-…-windows-x64-cpu-setup.exe` |
| Mac (M1~M4, Apple Silicon) | `MRSeparator-…-macos-arm64.dmg` |
| Mac (Intel) | `MRSeparator-…-macos-x86_64.dmg` |

계산량이 많아 그래픽카드가 없는 PC(CPU 버전)에서는 4분 곡에 15~40분 정도 걸릴 수 있습니다. 처음 분리할 때 AI 모델(약 667MB)을 한 번 내려받습니다.

### v1.0.1
- 왼쪽 화면 순서를 바꿨습니다: **① 곡 불러오기 → ② 키 조절 → ③ 템포 조절 → ④ AI 음원 분리**. 화면 위에 있던 [파일 열기] · [YouTube] 는 ① 카드로 옮겼습니다. 곡을 불러오면 [AI 분리 시작] 버튼이 보이도록 자동으로 내려갑니다.
- "골라 듣기 · 믹스 저장" 카드를 없앴습니다. 골라 듣기는 오른쪽 트랙의 S(솔로) · M(음소거) · 볼륨으로, 저장은 아래 막대의 [선택한 트랙 저장] · [전체 트랙 저장] 으로 합니다.

### v1.0.0 (첫 버전)
- 곡을 열고 [AI 분리 시작] 을 누르면 6개 트랙으로 나눕니다. 여섯 트랙을 모두 더하면 원곡과 정확히 같습니다.
- 기본 모델 BS-RoFormer SW — 내려받아 쓸 수 있는 모델 가운데 피아노 분리 점수가 가장 높습니다 (MVSEP 피아노 SDR 7.80dB).
- **골라 듣기**: 트랙의 S(솔로)로 한 악기만, M(음소거)으로 한 악기만 빼고 듣습니다. 트랙마다 볼륨(0~150%)을 바꿀 수 있습니다.
- 트랙별 저장 · 전체 트랙 저장.
- 키(−12 ~ +12)·템포(50 ~ 200%) 조절, YouTube 입력, 화음 따기(실험).
