# Vigorly physical-device release acceptance

Use the exact TestFlight build intended for release. Record device/OS/build and attach a screen recording or screenshot for every row. A release owner must mark all rows Pass; “not tested” is not a pass.

Candidate tracking (19 August 2026): EAS production build `587289eb-10f0-4dd7-bb95-acf74a3c2eba` completed successfully as Vigorly `1.0.0 (14)`, with its embedded Watch target synchronized to build 14. Submission `8a1df61e-c436-4d8f-acb0-d690968149ce` is attached to that exact artifact and remains `IN_QUEUE`. Expo's live status API reports the unresolved incident **“iOS Submissions hanging on App Store Connect build uploads”** with EAS Submit degraded. Do not send a duplicate build while this submission is active, and do not execute or mark P01–P15 against build 13; begin only after App Store Connect finishes processing build 14.

Signed-artifact verification: the downloaded build-14 IPA has SHA-256 `99b5050cc14f4ac16e2628e0a3a11668b5651eb9a1cdce1bebd58d9ab9bf1e8d`. `codesign --verify --deep --strict` passes. Its phone and Watch Info.plists contain the expected bundle IDs and matching `1.0.0 (14)` versions; the Watch declares `WKCompanionAppBundleIdentifier = com.advaice.calorielens`. Both production signatures have `get-task-allow = false`, `beta-reports-active = true`, the Advaice team identifier, and HealthKit entitlement. The IPA's Hermes bundle and Watch executable contain the build-14 sync-status and corrected-empty-state markers, confirming that the signed artifact contains the audited source rather than the earlier build-13 implementation.

| ID | Setup | Acceptance procedure | Pass condition | Result |
|---|---|---|---|---|
| P01 | Fresh iPhone + real inbox | Create an account, wait for confirmation, open the link cold, sign in, request recovery, open that link cold, change password. | Every deep link returns to the correct screen; old and new password behavior matches the UI. | ☐ |
| P02 | Returning account + guest logs | Sign out, create guest meal/water/workout data, then sign into an account that already has cloud history. | Guest and account histories both remain, with no duplicate IDs. | ☐ |
| P03 | Two iPhones | Edit a meal and lower water on phone A. Foreground phone B without making a local edit. | Phone B pulls the correction; the meal exists on exactly one date and the lower water value remains. | ☐ |
| P04 | iPhone + paired Watch | Start a routine on iPhone while Watch app is closed. | HealthKit opens Vigorly on Watch, the same session appears on both, and Watch metrics return to phone. | ☐ |
| P05 | Watch away from phone | Start, edit KG/reps/RPE, time a hold, finish offline, then reconnect. | Exactly one complete session syncs; no set, timer, RPE, or superset is lost. | ☐ |
| P06 | iPhone + Watch | Start a timed set, background/reopen each device, pause/resume on the other device, and finish early. Repeat for rest. | Both devices converge on one deadline/state and the correct set completes. | ☐ |
| P07 | Watch | Complete the wrong set, reopen it, correct it; try to complete a blank/zero-rep strength set. | Reopen works and blank strength completion is rejected with visible/haptic feedback. | ☐ |
| P08 | 40 mm and 45/49 mm Watch | Search exercises, scroll KG/reps/duration directly, run eight sets with largest text. | Controls remain reachable; values can be changed without first tapping a number box; primary actions do not clip. | ☐ |
| P09 | Health partial sharing | Share steps and energy but not sleep/weight, then sync. | Shared metrics update; unshared old values are preserved; UI names source, newest sample, and empty categories. | ☐ |
| P10 | Mixed workout + Watch | Complete weights plus treadmill intervals and immediately finish after the last set. | Health records cross training; final measured calories are nonzero when Watch measured them and never show an undefined range. | ☐ |
| P11 | Voice + poor network | Dictate English/Hinglish food, “current weight is 89,” then disable network and repeat with installed/offline speech and keyboard. | Supported voice paths parse; failures explain fallback; keyboard always logs through the same review flow. | ☐ |
| P12 | VoiceOver iPhone + Watch | Traverse auth, onboarding, meal correction, custom exercise, KG/reps/duration, completed-set reopen, and destructive confirmations. | Every control has a meaningful label, focus order is usable, and no required action depends only on a hidden gesture. | ☐ |
| P13 | iPhone SE, 200% text | Complete onboarding, Settings preferences, Food correction, Progress, routine editing, and live workout. | Save/close/delete actions remain reachable; critical values do not overlap or disappear. | ☐ |
| P14 | Backup + populated account | Export, alter data, restore, and select a malformed JSON file. | Valid backup merges deterministically; malformed backup changes nothing and shows an error. | ☐ |
| P15 | Supabase account | Export, clear fitness history, then delete the account. | Clear preserves profile/goals/routines/learned foods; delete removes cloud identity/data and clears the local signed-in vault. | ☐ |

## Automated gates before uploading the candidate

Run from `mobile/`:

```sh
npm run typecheck
npm run lint
npm test
npm run sync:native-versions
npm run check:phrasings
npm run check:catalog
npm run check:logging
npx expo export --platform ios --output-dir /tmp/vigorly-ios-export
xcrun --sdk watchos swiftc -typecheck -target arm64-apple-watchos10.0 ios/VigorlyWatch/WatchWorkoutManager.swift ios/VigorlyWatch/WorkoutView.swift ios/VigorlyWatch/VigorlyWatchApp.swift
xcodebuild -workspace ios/Vigorly.xcworkspace -scheme Vigorly -configuration Release -destination 'generic/platform=iOS Simulator' -derivedDataPath /tmp/vigorly-release-derived CODE_SIGNING_ALLOWED=NO build
xcodebuild test -workspace ios/Vigorly.xcworkspace -scheme VigorlyUITests -destination '<small-iPhone-simulator-id>' -derivedDataPath /tmp/vigorly-release-derived CODE_SIGNING_ALLOWED=NO
```

Latest native simulator evidence (19 August 2026): the full 126-target Release graph built successfully, Xcode validated the embedded Watch binary, and the paired iPhone 17 / Apple Watch Series 11 simulators both launched the generated apps. The Watch displayed `Phone connected · Synced now` only after receiving the iPhone library and returning its explicit receipt. The phone and Watch bundles were verified as `1.0.0 (13)` with companion bundle IDs `com.advaice.calorielens` and `com.advaice.calorielens.watchkitapp`. This is build/launch/delivery-receipt evidence only; it does not replace P01–P15 on physical devices.

The Release auth surface was also rendered on the smallest available iPhone 17e simulator at `accessibility-extra-extra-extra-large`. A first pass exposed runaway display scaling; decorative branding, the hero heading, form controls, and action labels were then bounded independently. The rebuilt screen keeps the mode selector and fields readable without overlap while preserving up to 200% scaling for body/input text and scroll access to every action. Two Release XCUITests pass on that configuration: core auth controls remain reachable and dragging the signup form dismisses the keyboard after text entry. P13 remains unchecked until the full journey is recorded on a physical small-screen device.

The rebuilt Watch library was also rendered on standalone 40 mm and 49 mm simulators. The sync state, Quick workout action, and corrected `No routines yet` empty state remain readable and scrollable on both sizes. The watchOS 26.5 simulator does not support changing Dynamic Type through `simctl`, so largest-text Watch behavior remains part of physical P08/P12 rather than being inferred from these screenshots.

Also run `npx expo-doctor`; its checked-in-native-project and Health Connect plugin warnings are documented in `QA-30-AGENT-REPORT.md` and are not runtime test passes.

## Dashboard gates before public release

- Supabase Auth now enforces the same 10-character minimum as the shipped app (live setting verified 19 August 2026). Leaked-password protection remains unavailable while the project is on Supabase Free; after an owner intentionally upgrades to Pro, enable it and verify a known compromised password is rejected. Do not purchase a plan merely as part of this checklist.
- Confirm email confirmation and recovery redirect URLs include the production `vigorly://` scheme.
- Confirm the intended tester group has the candidate TestFlight build and the embedded Watch app is listed in build metadata.
