#!/usr/bin/env bash
# macOS 코드 서명 + 공증(Notarization) + 스테이플링 + DMG 생성
#
# 필요 환경 변수
#   MS_CODESIGN_ID   "Developer ID Application: 이름 (TEAMID)"  (키체인에 인증서가 있어야 함)
# 공증(선택) — 둘 중 하나
#   A) APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD   (appleid.apple.com 에서 만든 앱 암호)
#   B) NOTARY_KEY_PATH(.p8), NOTARY_KEY_ID, NOTARY_ISSUER_ID   (App Store Connect API 키)
#
# 사용: packaging/macos/sign_and_notarize.sh "dist/Music Separator.app" dist/release/MusicSeparator.dmg
set -euo pipefail
APP="$1"
DMG="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
ENT="$HERE/entitlements.plist"
ID="${MS_CODESIGN_ID:?MS_CODESIGN_ID 가 필요합니다}"

sign() {
  codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$@"
}

echo "== 1/5 내부 바이너리 서명 (Mach-O 파일 전체)"
# 실행 파일, dylib, so, 그리고 데이터로 들어간 ffmpeg 같은 Mach-O 파일까지 모두 서명
while IFS= read -r -d '' f; do
  if file -b "$f" | grep -q "Mach-O"; then
    chmod u+w "$f"
    sign "$f" >/dev/null
  fi
done < <(find "$APP/Contents" -type f ! -path "*/_CodeSignature/*" -print0)

echo "== 2/5 프레임워크 번들 서명 (깊은 것부터)"
find "$APP/Contents" -type d -name "*.framework" | awk '{ print length, $0 }' | sort -rn | cut -d" " -f2- |
while IFS= read -r fw; do sign "$fw" >/dev/null; done

echo "== 3/5 앱 번들 서명"
sign "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

notarize() {
  local target="$1"
  if [[ -n "${NOTARY_KEY_PATH:-}" ]]; then
    xcrun notarytool submit "$target" --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" \
      --issuer "$NOTARY_ISSUER_ID" --wait --timeout 60m
  else
    xcrun notarytool submit "$target" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
      --password "$APPLE_APP_PASSWORD" --wait --timeout 60m
  fi
}

CAN_NOTARIZE=0
if [[ -n "${NOTARY_KEY_PATH:-}" || ( -n "${APPLE_ID:-}" && -n "${APPLE_APP_PASSWORD:-}" ) ]]; then
  CAN_NOTARIZE=1
fi

if [[ $CAN_NOTARIZE == 1 ]]; then
  echo "== 4/5 앱 공증 + 스테이플"
  ZIP="$(mktemp -d)/app.zip"
  ditto -c -k --keepParent "$APP" "$ZIP"
  notarize "$ZIP"
  xcrun stapler staple "$APP"
else
  echo "== 4/5 공증 정보 없음 → 서명만 수행 (Gatekeeper 경고는 남을 수 있음)"
fi

echo "== 5/5 DMG 생성/서명"
"$HERE/make_dmg.sh" "$APP" "$DMG"
codesign --force --timestamp --sign "$ID" "$DMG"
if [[ $CAN_NOTARIZE == 1 ]]; then
  notarize "$DMG"
  xcrun stapler staple "$DMG"
  spctl -a -t open --context context:primary-signature -v "$DMG" || true
fi
spctl -a -vv "$APP" || true
echo "완료: $DMG"
