# Vigorly — TestFlight runbook (Advaice Limited)

Getting build **1.0.0 (2)** into TestFlight under the **Advaice Limited** team
(`95R5R7A683`), built on your own Mac. No third-party build service is involved
and no signing credential leaves your machine.

**Scope of this run:** internal testing only. No testers are added, no external
group is created, and no public link is generated. The build lands in TestFlight
and waits there until you decide otherwise.

---

## Where things stand

| Item | Value |
| --- | --- |
| App Store Connect record | Created — Vigorly, Apple ID `6796998952` |
| Bundle ID | `com.advaice.calorielens`, App ID registered with HealthKit |
| Team | Advaice Limited, `95R5R7A683` |
| Version / build | `1.0.0` (`2`) |
| Build 1 | Uploaded, then rejected by Apple's static scan — see the note at the end |
| Build 2 | Archived and exported to `~/Desktop/Vigorly-release/Vigorly.ipa` |
| Backend | None. `EXPO_PUBLIC_API_URL` is empty, so the app runs entirely on-device |
| Account deletion | In-app, under Account & privacy — required by guideline 5.1.1(v) |

Internal testing does **not** go through Beta App Review, so the build is
installable by team members as soon as processing finishes.

---

## 1. Check what you are about to upload

```bash
cd ~/Desktop/Docs/Apps/Calorie-Lens
./scripts/verify-signing.sh
```

This unpacks the .ipa and reads the signature and the embedded provisioning
profile out of the binary itself. It has to, because the archive log is
misleading on exactly this point: `xcodebuild archive` signs with whatever
identity the Release configuration resolves to — for you that was
`Apple Development: Sahil Bharti` — and then the export step re-signs the payload
for distribution. So a development identity in the archive log is normal and
says nothing about the .ipa. Only the .ipa does.

Four things have to be true, and the script prints each one:

- an **Apple Distribution** certificate, not a development one
- **no `get-task-allow`** entitlement, which is what marks a build as debuggable
  and is an automatic rejection
- **no device list** in the profile, which would mean ad-hoc rather than App Store
- team **95R5R7A683**

If any line reads FAIL, the script tells you what to do; do not upload.

`scripts/archive.sh` now runs this check itself at the end of every export, so
future builds are verified without you remembering to.

## 2. Upload

```bash
open -a Transporter ~/Desktop/Vigorly-release
```

Transporter is Apple's free uploader from the Mac App Store. Sign in as the
Apple Account that holds your Advaice Limited role, drag `Vigorly.ipa` into the
window, and press **Deliver**.

Expect a warning that dSYMs are missing for `React.framework`,
`ReactNativeDependencies.framework` and `hermes.framework`. That is a warning,
not an error, and it does not block the upload. It means crash reports inside
those prebuilt frameworks arrive unsymbolicated; crashes in your own JavaScript
and native code are unaffected.

Processing on Apple's side takes a few minutes to about an hour, and you get an
email either way. A build that fails the static scan is rejected at this stage
rather than at upload — build 1 is the example.

## 3. Mark the build internal-only

Once it appears under **TestFlight → iOS Builds**, open it and switch
**Internal Only** on. Do this before anything else on the build.

This is the safety rail for what you asked for. A build flagged Internal Only
can never be added to an external group or submitted for external distribution,
even by accident.

Leave **Internal Testing** groups empty. Add no testers, create no external
group, generate no public link. The build sits in TestFlight, ready, until you
decide who gets it.

Apple asks for **Test Information** (feedback email, description) only when you
first invite external testers, so you can skip it.

## 4. When you are ready to add testers

Internal testers must already be users on the Advaice Limited App Store Connect
team — invite them under Users and Access first, then add them to an internal
group under TestFlight. Up to 100 internal testers, and each build stays
installable for 90 days.

External testers are a different path and *do* require Beta App Review. That is
also the point at which the App Store privacy questionnaire, the support URL and
the privacy policy URL become mandatory. `PRIVACY.md` and `APP_STORE.md` have
the content ready for that when you get there.

---

## Every build after this one

`buildNumber` must strictly increase for each upload to the same version, or App
Store Connect rejects it. In `mobile/app.json`:

```json
"ios": { "buildNumber": "3" }
```

Then:

```bash
cd ~/Desktop/Docs/Apps/Calorie-Lens/mobile
npx expo prebuild --clean
cd ..
TEAM_ID=95R5R7A683 ./scripts/archive.sh
```

`archive.sh` archives, exports, and verifies the signature. Bump `version`
(`1.0.1`, `1.1.0`) only when the release itself changes, not per upload.

The `--clean` matters. `prebuild` regenerates `mobile/ios/` from `app.json`;
without it you can archive a stale native project. That is how build 1 shipped a
bundle still called `CalorieLens.app` after the rename.

---

## Why build 1 was rejected

`ITMS-90683: Missing purpose string in Info.plist` — specifically
`NSPhotoLibraryUsageDescription`.

Apple's static scan rejects a binary that *links* a protected API regardless of
whether the app ever calls it, and React Native's image loader plus
`expo-document-picker` both reference the Photos frameworks. Vigorly never opens
a photo picker. Build 2 declares `NSPhotoLibraryUsageDescription`,
`NSPhotoLibraryAddUsageDescription` and `NSFaceIDUsageDescription`, each saying
plainly that the app does not use the capability — which is the truth, and is
what a user would read if a bundled framework ever did prompt.

If a future build genuinely adds photo attachments or biometric unlock, rewrite
these to describe the real feature. A purpose string that misdescribes what the
app does is a review problem in its own right.

That rejection is also useful evidence about signing: ITMS-90683 comes from
*processing*, which runs after signature validation. A badly signed build never
reaches it. So the upload path itself is known to work.

## Two things worth your judgement

**The encryption declaration.** `ITSAppUsesNonExemptEncryption` is set to
`false`, which means the upload will not stop to ask you about export
compliance. The app does bundle AES-256-GCM (`@noble/ciphers`) to encrypt local
data at rest. The common reading is that this qualifies as exempt — encryption
limited to protecting the user's own data on their own device, plus standard
HTTPS — but it is a legal declaration you are signing, not a technical setting,
and I am not the right party to make it for you. Section 10 of `APP_STORE.md`
lays out the facts in full. If you would rather answer Apple's questionnaire
yourself, remove that line from `app.json` and the upload will prompt you.

**HealthKit and TestFlight.** Internal testing skips review, so nothing blocks
you now. When you go external or to the App Store, HealthKit apps get closer
scrutiny: the privacy policy must specifically describe health data handling,
and health data must not be used for advertising or sold. `PRIVACY.md` already
covers this, but read it against what the app actually does before you submit.
