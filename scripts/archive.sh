#!/usr/bin/env bash
#
# Archive Vigorly for the App Store without opening Xcode.
#
# The Xcode GUI is only a front-end for these commands. Everything here —
# signing, archiving, exporting — runs through the same toolchain the
# Organizer uses, so a broken Xcode window does not block a release.
#
#   First run (shows your signing identities and team ID):
#     ./scripts/archive.sh
#
#   Then:
#     TEAM_ID=XXXXXXXXXX ./scripts/archive.sh
#
set -euo pipefail

SCHEME="Vigorly"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IOS_DIR="$ROOT/mobile/ios"
WORKSPACE="$IOS_DIR/$SCHEME.xcworkspace"
OUT="$HOME/Desktop/$SCHEME-release"
ARCHIVE="$OUT/$SCHEME.xcarchive"

TEAM_ID="${TEAM_ID:-}"

if [ ! -d "$WORKSPACE" ]; then
  echo "No workspace at $WORKSPACE"
  echo "Run 'npx expo prebuild --clean' in mobile/ first."
  exit 1
fi

if [ -z "$TEAM_ID" ]; then
  echo
  echo "Code-signing identities installed on this Mac:"
  echo
  security find-identity -v -p codesigning | sed -n 's/^ *[0-9]*) [A-F0-9]* "\(.*\)"$/  \1/p'
  echo
  echo "Your team ID is the 10 characters in brackets above."
  echo "Re-run with it set, for example:"
  echo
  echo "  TEAM_ID=ABCDE12345 $0"
  echo
  exit 1
fi

echo "==> Archiving $SCHEME for team $TEAM_ID"
rm -rf "$OUT"
mkdir -p "$OUT"

# -allowProvisioningUpdates lets xcodebuild fetch or create the distribution
# certificate and profile from the Apple ID already signed in to Xcode, which
# is what Signing & Capabilities does when you pick a team in the GUI.
xcodebuild \
  -workspace "$WORKSPACE" \
  -scheme "$SCHEME" \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -archivePath "$ARCHIVE" \
  -allowProvisioningUpdates \
  DEVELOPMENT_TEAM="$TEAM_ID" \
  CODE_SIGN_STYLE=Automatic \
  archive

echo "==> Exporting a signed .ipa"

cat > "$OUT/ExportOptions.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key><string>app-store-connect</string>
  <key>teamID</key><string>$TEAM_ID</string>
  <key>signingStyle</key><string>automatic</string>
  <key>uploadSymbols</key><true/>
  <key>stripSwiftSymbols</key><true/>
  <key>destination</key><string>export</string>
</dict>
</plist>
PLIST

xcodebuild \
  -exportArchive \
  -archivePath "$ARCHIVE" \
  -exportOptionsPlist "$OUT/ExportOptions.plist" \
  -exportPath "$OUT" \
  -allowProvisioningUpdates

echo
echo "==> Done"
echo
ls -lh "$OUT"/*.ipa 2>/dev/null || echo "No .ipa produced — read the export output above."
echo
echo "Version check (should read 1.0.0 / 2):"
/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
  "$ARCHIVE/Products/Applications/$SCHEME.app/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' \
  "$ARCHIVE/Products/Applications/$SCHEME.app/Info.plist" 2>/dev/null || true
echo
echo "Upload the .ipa with Transporter (free, Mac App Store):"
echo "  open -a Transporter '$OUT'"
echo "Sign in as the Advaice Limited account, drag the .ipa in, press Deliver."
