"""MSI 설치 화면용 파일 생성: 이용 안내(license.rtf), 상단 배너(banner.bmp 493x58), 시작/완료 화면(dialog.bmp 493x312).

    python scripts/make_msi_assets.py   → packaging/windows/msi/ (MR MAKER) 와 packaging/windows/msi/separator/
                                          (MR SEPARATOR) 에 저장 (생성된 파일도 저장소에 포함)
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packaging" / "windows" / "msi"
FONT = ROOT / "assets" / "fonts"
SAGE, SAGE_DEEP, INK, DIM = (234, 244, 241), (47, 111, 98), (31, 39, 35), (101, 112, 106)

EDITIONS = {
    # 폴더, 짧은 이름, 아이콘, 소개 문장
    "maker": ("", "MR MAKER", "icon.png",
              "MR MAKER는 누구나 쉽게 필요한 MR을 제작할 수 있는 AI 음원 분리 프로그램입니다."),
    "separator": ("separator", "MR SEPARATOR", "icon_separator.png",
                  "MR SEPARATOR는 음악을 보컬·드럼·베이스·기타·피아노·나머지 악기로 나누어 듣고 연습할 수 있는 "
                  "AI 음원 분리 프로그램입니다."),
}
SHORT, ICON, INTRO = "MR MAKER", "icon.png", EDITIONS["maker"][3]

NOTICE = [
    ("{short} for ui.clover 이용 안내", True),
    ("", False),
    ("{intro}", False),
    ("", False),
    ("1. 본 프로그램의 배포·공유는 제한하지 않으나, 가급적 클로버 활동 내에서 사용해 주시면 좋겠습니다.", False),
    ("2. 해당 프로그램과 생성 음원을 영리적 목적으로 이용하지 말아주세요. (판매, 부가 수익 창출 등)", False),
    ("3. 모든 음원 처리는 사용자의 컴퓨터 안에서만 이루어지며, 음원과 개인 정보는 외부로 전송되지 않습니다.", False),
    ("4. 전체 소스 코드는 https://github.com/jiamong2/MR_MAKER 에 공개되어 있습니다.", False),
    ("", False),
    ("포함된 오픈소스(Demucs, FFmpeg, Rubber Band, PyTorch, Qt 등)의 라이선스는 설치 폴더의 "
     "THIRD_PARTY_NOTICES 와 위 소스 저장소에서 확인할 수 있습니다.", False),
]


def rtf_text(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch in "\\{}":
            out.append("\\" + ch)
        elif o < 128:
            out.append(ch)
        else:
            out.append(f"\\u{o if o < 32768 else o - 65536}?")
    return "".join(out)


def make_rtf():
    lines = [r"{\rtf1\ansi\ansicpg949\deff0{\fonttbl{\f0\fnil\fcharset129 Malgun Gothic;}}", r"\viewkind4\uc1\pard\sa60\f0\fs18"]
    for text, bold in NOTICE:
        t = rtf_text(text.format(short=SHORT, intro=INTRO))
        lines.append((r"{\b\fs22 " + t + r"}" if bold else t) + r"\par")
    lines.append("}")
    (OUT / "license.rtf").write_text("\n".join(lines), encoding="ascii")


def font(size: int, weight: str = "Regular"):
    return ImageFont.truetype(str(FONT / f"Pretendard-{weight}.otf"), size)


def clover(px: int) -> Image.Image:
    return Image.open(ROOT / "assets" / ICON).convert("RGBA").resize((px, px), Image.LANCZOS)


def make_banner():
    # 상단 배너: 왼쪽은 설치 프로그램이 제목/설명을 쓰는 자리 → 오른쪽 끝에만 로고
    im = Image.new("RGB", (493, 58), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([380, 0, 493, 58], fill=SAGE)
    d.polygon([(360, 0), (380, 0), (380, 58), (340, 58)], fill=SAGE)
    c = clover(42)
    im.paste(c, (437, 8), c)
    d.line([(0, 57), (493, 57)], fill=(214, 222, 216))
    im.save(OUT / "banner.bmp")


def make_dialog():
    # 시작/완료 화면: 왼쪽 164px 은 그림, 오른쪽은 설치 프로그램이 글자를 쓰는 흰 영역
    im = Image.new("RGB", (493, 312), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 164, 312], fill=SAGE)
    c = clover(96)
    im.paste(c, (34, 70), c)
    title = font(20, "Bold") if d.textlength(SHORT, font=font(20, "Bold")) <= 148 else font(17, "Bold")
    for text, f, y, col in ((SHORT, title, 186, INK), ("for ui.clover", font(12, "SemiBold"), 214, SAGE_DEEP)):
        w = d.textlength(text, font=f)
        d.text(((164 - w) / 2, y), text, font=f, fill=col)
    d.line([(164, 0), (164, 312)], fill=(214, 222, 216))
    im.save(OUT / "dialog.bmp")


if __name__ == "__main__":
    BASE = OUT
    for _ed, (_sub, SHORT, ICON, INTRO) in EDITIONS.items():
        OUT = BASE / _sub if _sub else BASE
        OUT.mkdir(parents=True, exist_ok=True)
        make_rtf()
        make_banner()
        make_dialog()
        print(f"{_ed}: written:", *(p.name for p in sorted(OUT.iterdir()) if p.is_file()))
