#!/usr/bin/env bash
# .app 을 드래그 설치용 DMG 로 묶습니다 (Applications 바로가기 포함).
# 사용: packaging/macos/make_dmg.sh "dist/Music Separator.app" dist/release/MusicSeparator.dmg
set -euo pipefail
APP="$1"
OUT="$2"
VOLNAME="${3:-Music Separator}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/설치 방법.txt" <<'EOF'
Music Separator 설치 방법

1. 'Music Separator' 아이콘을 'Applications' 폴더로 끌어다 놓으세요.
2. 응용 프로그램 폴더에서 Music Separator 를 실행하세요.

서명/공증되지 않은 빌드라면 처음 실행할 때 경고가 나올 수 있습니다:
  - 앱을 Control+클릭(우클릭) → [열기] → [열기]
  - 또는 시스템 설정 → 개인정보 보호 및 보안 → [그래도 열기]
EOF
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
# APFS 이미지에 압축(ULMO: lzma, macOS 10.15+) 사용, 실패하면 UDZO 로 대체
hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" -ov -fs HFS+ -format ULMO "$OUT" \
  || hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" -ov -fs HFS+ -format UDZO "$OUT"
echo "DMG: $OUT ($(du -h "$OUT" | cut -f1))"
