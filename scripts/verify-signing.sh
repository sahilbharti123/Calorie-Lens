#!/usr/bin/env bash
#
# Check an exported .ipa is signed for App Store distribution under Advaice
# Limited, before it goes anywhere near App Store Connect.
#
# Why this is a separate check: `xcodebuild archive` signs the archive with
# whatever identity the Release configuration happens to resolve to — usually a
# development one — and `-exportArchive` then RE-SIGNS for distribution on the
# way out. So the "Signing Identity: Apple Development ..." line in the archive
# log says nothing about the .ipa. Only the .ipa does, and that is what this
# reads.
#
#   ./scripts/verify-signing.sh                       # default export path
#   ./scripts/verify-signing.sh path/to/Some.ipa
#
set -euo pipefail

EXPECTED_TEAM="${EXPECTED_TEAM:-95R5R7A683}"
IPA="${1:-$HOME/Desktop/Vigorly-release/Vigorly.ipa}"

if [ ! -f "$IPA" ]; then
  echo "No .ipa at $IPA"
  echo "Run ./scripts/archive.sh first, or pass the path as an argument."
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unzip -q "$IPA" -d "$TMP"

APP="$(find "$TMP/Payload" -maxdepth 1 -name '*.app' -print -quit)"
if [ -z "$APP" ]; then
  echo "No .app inside $IPA — the export produced something unexpected."
  exit 1
fi

echo
echo "Checking $(basename "$IPA")  ($(du -h "$IPA" | cut -f1))"
echo

# ---------------------------------------------------------------- identity ---
AUTHORITY="$(codesign -dv --verbose=4 "$APP" 2>&1 | sed -n 's/^Authority=//p' | head -1)"
echo "  Signing identity     $AUTHORITY"

# ----------------------------------------------------------------- profile ---
PROFILE="$(security cms -D -i "$APP/embedded.mobileprovision" 2>/dev/null || true)"
read_profile() { printf '%s' "$PROFILE" | plutil -extract "$1" raw - 2>/dev/null || true; }

PROFILE_NAME="$(read_profile Name)"
TEAM_NAME="$(read_profile TeamName)"
TEAM_ID="$(read_profile TeamIdentifier.0)"
GET_TASK_ALLOW="$(read_profile 'Entitlements.get-task-allow')"
BETA_REPORTS="$(read_profile 'Entitlements.beta-reports-active')"
HAS_DEVICE_LIST=no
printf '%s' "$PROFILE" | plutil -extract ProvisionedDevices raw - >/dev/null 2>&1 && HAS_DEVICE_LIST=yes

echo "  Provisioning profile $PROFILE_NAME"
echo "  Team                 ${TEAM_NAME:-unknown} (${TEAM_ID:-unknown})"
echo "  Bundle ID            $(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Info.plist" 2>/dev/null || echo unknown)"
echo "  Version              $(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Info.plist" 2>/dev/null) ($(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP/Info.plist" 2>/dev/null))"
echo

# ----------------------------------------------------------------- verdict ---
fail=0

case "$AUTHORITY" in
  "Apple Distribution"*|"iPhone Distribution"*)
    echo "  ok    signed with a distribution certificate" ;;
  *)
    echo "  FAIL  signed with '$AUTHORITY', not a distribution certificate"
    fail=1 ;;
esac

# A development-signed build carries get-task-allow, which lets a debugger
# attach. App Store Connect rejects any binary that has it.
if [ "$GET_TASK_ALLOW" = "true" ]; then
  echo "  FAIL  get-task-allow is set — this is a development build"
  fail=1
else
  echo "  ok    no get-task-allow entitlement"
fi

# App Store profiles cover every device, so they carry no device list.
# An ad-hoc or development profile pins a specific set of UDIDs.
if [ "$HAS_DEVICE_LIST" = yes ]; then
  echo "  FAIL  the profile lists specific devices — ad-hoc or development, not App Store"
  fail=1
else
  echo "  ok    profile is not limited to specific devices"
fi

if [ "$BETA_REPORTS" = "true" ]; then
  echo "  ok    beta-reports-active is set — TestFlight will accept it"
fi

if [ "$TEAM_ID" = "$EXPECTED_TEAM" ]; then
  echo "  ok    team is $TEAM_NAME ($TEAM_ID)"
else
  echo "  FAIL  team is ${TEAM_ID:-unknown}, expected $EXPECTED_TEAM (Advaice Limited)"
  fail=1
fi

echo
if [ "$fail" = 0 ]; then
  echo "  Ready to upload."
  echo "    open -a Transporter '$(dirname "$IPA")'"
  echo "  Sign in as the Advaice Limited account, drag the .ipa in, press Deliver."
  echo
  exit 0
fi

cat <<'NOTE'
  Do not upload this build.

  The usual cause is that xcodebuild could not obtain a distribution
  certificate for the team, and quietly fell back. Fixing it needs your Apple
  Account signed in under Xcode > Settings > Accounts with Account Holder,
  Admin or App Manager on Advaice Limited — a Developer-level role cannot
  create the distribution certificate.

  Once that is true, re-run the export alone (seconds, no rebuild):

    xcodebuild -exportArchive \
      -archivePath ~/Desktop/Vigorly-release/Vigorly.xcarchive \
      -exportOptionsPlist ~/Desktop/Vigorly-release/ExportOptions.plist \
      -exportPath ~/Desktop/Vigorly-release \
      -allowProvisioningUpdates

  then run this script again.
NOTE
exit 1
