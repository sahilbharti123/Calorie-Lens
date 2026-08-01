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

# Prints "<team name> | <team id>" for every provisioning profile on this Mac.
# Certificate common names are unreliable for this: an Apple Development cert is
# named after the developer, so its bracketed code can belong to a personal team.
# Profiles carry TeamName explicitly, which is the only way to be sure which
# bracketed code is Advaice Limited.
list_teams() {
  # Xcode 16 moved provisioning profiles out of ~/Library/MobileDevice into its
  # own UserData directory. Scan both, or the table reads empty on a modern
  # Xcode and the team check silently has nothing to check against.
  local dirs=(
    "$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles"
    "$HOME/Library/MobileDevice/Provisioning Profiles"
  )
  local found=0
  for dir in "${dirs[@]}"; do
    [ -d "$dir" ] || continue
  for profile in "$dir"/*.mobileprovision; do
    [ -e "$profile" ] || continue
    local plist name team tid
    plist="$(security cms -D -i "$profile" 2>/dev/null)" || continue
    name="$(printf '%s' "$plist" | plutil -extract Name raw - 2>/dev/null || true)"
    team="$(printf '%s' "$plist" | plutil -extract TeamName raw - 2>/dev/null || true)"
    tid="$(printf '%s' "$plist" | plutil -extract TeamIdentifier.0 raw - 2>/dev/null || true)"
    [ -n "$tid" ] && { printf '  %-28s  %-24s  %s\n' "$team" "$tid" "$name"; found=1; }
  done
  done
  [ "$found" = 1 ] || echo "  (no provisioning profiles installed yet)"
}

has_distribution_cert() {
  security find-identity -v -p codesigning | grep -q "Apple Distribution"
}

if [ -z "$TEAM_ID" ]; then
  echo
  echo "Signing identities on this Mac:"
  echo
  security find-identity -v -p codesigning | sed -n 's/^ *[0-9]*) [A-F0-9]* "\(.*\)"$/  \1/p'
  echo
  echo "Teams, from the provisioning profiles (team | team ID | profile):"
  echo
  list_teams
  echo

  if has_distribution_cert; then
    echo "An Apple Distribution certificate is installed."
  else
    echo "NOTE: no Apple Distribution certificate is installed — only Development."
    echo "      App Store builds must be signed with a distribution certificate."
    echo "      xcodebuild will try to create one for the team you pass below;"
    echo "      that needs your Apple ID signed in to Xcode with Account Holder,"
    echo "      Admin or App Manager on that team."
  fi

  echo
  echo "Find the row that says Advaice Limited and use ITS team ID:"
  echo
  echo "  TEAM_ID=XXXXXXXXXX $0"
  echo
  echo "If no Advaice Limited row appears, the Mac has never signed for that"
  echo "team. Sign in under Xcode > Settings > Accounts first, or read the ID"
  echo "from developer.apple.com > Membership details."
  echo
  exit 1
fi

# Refuse to sign for a team that is not the one the release belongs to.
TEAM_NAME="$(list_teams | awk -v id="$TEAM_ID" '$0 ~ id { $NF=""; print $1" "$2" "$3 }' | head -1 | xargs || true)"
if [ -n "$TEAM_NAME" ]; then
  echo "==> Team $TEAM_ID resolves to: $TEAM_NAME"
  case "$TEAM_NAME" in
    *Advaice*) : ;;
    *) echo
       echo "That is not Advaice Limited. Stopping — signing a release with the"
       echo "wrong team puts the app under the wrong developer account."
       echo "Set ALLOW_ANY_TEAM=1 to override deliberately."
       [ "${ALLOW_ANY_TEAM:-}" = "1" ] || exit 1 ;;
  esac
else
  echo "==> No local profile matches $TEAM_ID yet; xcodebuild will request one."
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
