# Vigorly — TestFlight runbook (Advaice Limited)

Build 1.0.0 (1) to TestFlight under the **Advaice Limited** team, built locally in
Xcode and uploaded through Organizer. No third-party build service is involved and
no signing credential leaves your Mac.

**Scope of this run:** internal testing only. No testers are added, no external
group is created, and no public link is generated. The build lands in TestFlight
and waits.

---

## What is already configured

| Item | Value |
| --- | --- |
| Bundle ID | `com.advaice.calorielens` |
| Version / build | `1.0.0` (`1`) |
| Encryption declaration | `ITSAppUsesNonExemptEncryption: false` in `Info.plist` — see the note at the end |
| Privacy manifest | `PrivacyInfo.xcprivacy`, four required API reasons declared |
| Purpose strings | Microphone, speech, HealthKit, photo library, Face ID — see the note below |
| Backend | None. `EXPO_PUBLIC_API_URL` is empty, so the app runs entirely on-device |
| Account deletion | In-app, under Account & privacy — required by guideline 5.1.1(v) |

Internal testing does **not** go through Beta App Review, so this build is
installable by team members as soon as processing finishes.

---

## 1. Confirm your role on the team

In [App Store Connect → Users and Access](https://appstoreconnect.apple.com/access/users),
check your own row while the **Advaice Limited** team is selected in the top-right
account switcher. You need **Account Holder**, **Admin**, or **App Manager** to
create an app record and to upload builds. Developer alone can upload but cannot
create the record.

If the switcher does not offer Advaice Limited, you are not on the team yet — the
Account Holder must invite your Apple Account first, and everything below waits on
that.

## 2. Register the App ID with HealthKit

Go to [Certificates, Identifiers & Profiles → Identifiers](https://developer.apple.com/account/resources/identifiers/list),
make sure **Advaice Limited** is the selected team, and add an identifier:

- Type: **App IDs → App**
- Description: `Vigorly`
- Bundle ID: **Explicit**, `com.advaice.calorielens`
- Capabilities: tick **HealthKit**

HealthKit matters. The app reads steps, activity, sleep and weight through
`@kingstinct/react-native-healthkit`, whose config plugin writes the
`com.apple.developer.healthkit` entitlement into the target. If the App ID does
not carry that capability, signing fails at archive time with a provisioning
error that does not name HealthKit — so set it now rather than debugging it later.

Nothing else needs enabling. Microphone and speech recognition are `Info.plist`
usage strings, not capabilities.

## 3. Create the App Store Connect record

In [App Store Connect → Apps](https://appstoreconnect.apple.com/apps) → **+** → **New App**:

- Platform: **iOS**
- Name: `Vigorly` (must be unique across the App Store; if it is taken, pick
  the alternative from `APP_STORE.md` and tell me so I can align the metadata)
- Primary language: **English (U.K.)**
- Bundle ID: `com.advaice.calorielens` — pick the identifier you just registered
- SKU: `vigorly-ios` (internal reference only, never shown to users)
- User Access: **Full Access**

You do not need to fill in pricing, screenshots, or the App Store listing to use
TestFlight. Leave all of it alone for now.

## 4. Point the project at the new bundle ID

The identifier changed from `com.sahilbharti.calorielens`, so the native project
has to be regenerated — editing it in Xcode alone will be overwritten next prebuild.

```bash
cd ~/Desktop/Docs/Apps/Calorie-Lens/mobile
npx expo prebuild --clean
```

Then confirm it took:

```bash
grep -r PRODUCT_BUNDLE_IDENTIFIER ios/Vigorly.xcodeproj/project.pbxproj | head -3
```

Every line should read `com.advaice.calorielens`.

## 5. Set the signing team in Xcode

```bash
open ios/Vigorly.xcworkspace
```

Select the **Vigorly** target → **Signing & Capabilities** → **Release** tab:

- **Automatically manage signing**: on
- **Team**: Advaice Limited
- **Bundle Identifier**: `com.advaice.calorielens`

Xcode will fetch or create the distribution certificate and provisioning profile
against the Advaice team. Check that HealthKit appears in the capabilities list —
if it is missing, step 2 did not save.

Do this on the **Release** configuration specifically. Debug is what the simulator
uses and its settings are irrelevant here.

## 6. Archive

In Xcode, set the run destination to **Any iOS Device (arm64)** — you cannot
archive with a simulator selected — then **Product → Archive**.

This runs a Release build, which is a different code path from everything you have
run so far: JavaScript is bundled into the app rather than served by Metro, the
dev client is inert, and the React Compiler output is minified. Expect it to take
several minutes and to be the first place a release-only problem would surface.

If it fails, the error is almost always one of: signing (step 2 or 5), or a stale
Pods directory. For the latter:

```bash
cd ios && pod install && cd ..
```

## 7. Upload

The Organizer opens on a successful archive (or **Window → Organizer**). Select
the archive → **Distribute App** → **TestFlight & App Store** → **Upload**.

Accept the defaults: upload symbols (you want symbolicated crash reports), manage
signing automatically. Xcode validates before sending; validation failures are
reported precisely and are usually a missing icon size or an entitlement mismatch,
neither of which should apply here.

Processing on Apple's side typically takes a few minutes to about an hour. You get
an email when it finishes.

## 8. Mark the build internal-only

Once the build appears under **TestFlight → iOS Builds**, open it and set
**Internal Only** to on.

This is the safety rail for what you asked: a build flagged Internal Only can
never be added to an external group or submitted for external distribution, even
by accident. Do this before anything else on the build.

Leave **Internal Testing** groups empty. Add no testers, create no external group,
generate no public link. The build will sit in TestFlight, ready, until you decide
who gets it.

Apple asks for **Test Information** (feedback email, description) only when you
first invite external testers, so you can skip it.

## 9. When you are ready to add testers

Internal testers must already be users on the Advaice Limited App Store Connect
team — you invite them under Users and Access first, then add them to an internal
group under TestFlight. Up to 100 internal testers, and each build stays
installable for 90 days.

External testers are a different path and *do* require Beta App Review. That is
also the point at which the App Store privacy questionnaire, the support URL and
the privacy policy URL become mandatory — `PRIVACY.md` and `APP_STORE.md` have the
content ready for that when you get there.

---

## Every build after this one

`buildNumber` must strictly increase for each upload to the same version, or App
Store Connect rejects it. In `mobile/app.json`:

```json
"ios": { "buildNumber": "2" }
```

Then `npx expo prebuild --clean`, archive, upload. Bump `version` (`1.0.1`,
`1.1.0`) only when the release itself changes, not per upload.

---

## Purpose strings you did not ask for

`NSPhotoLibraryUsageDescription`, `NSPhotoLibraryAddUsageDescription` and
`NSFaceIDUsageDescription` are declared even though Vigorly never opens a photo
picker or a Face ID prompt.

Apple's static scan (`ITMS-90683`) rejects a binary that *links* a protected API
regardless of whether the app calls it, and React Native's image loader plus
`expo-document-picker` both reference the Photos frameworks. The strings say
plainly that the app does not use them, which is the truth and is what a user
would read if a bundled framework ever did prompt.

If a future build genuinely adds photo attachments or biometric unlock, rewrite
these to describe the real feature — a purpose string that misdescribes what the
app does is a review problem.

## Two things worth your judgement

**The encryption declaration.** `ITSAppUsesNonExemptEncryption` is set to `false`,
which means the upload will not stop to ask you about export compliance. The app
does bundle AES-256-GCM (`@noble/ciphers`) to encrypt local data at rest. The
common reading is that this qualifies as exempt — encryption limited to protecting
the user's own data on their own device, plus standard HTTPS — but it is a legal
declaration you are signing, not a technical setting, and I am not the right party
to make it for you. If you would rather answer Apple's questionnaire yourself,
remove that line from `app.json` and the upload will prompt you.

**HealthKit and TestFlight.** Internal testing skips review, so nothing blocks you
now. When you go external or to the App Store, HealthKit apps get closer scrutiny:
the privacy policy must specifically describe health data handling, and health data
must not be used for advertising or sold. `PRIVACY.md` already covers this, but
read it against what the app actually does before you submit.
