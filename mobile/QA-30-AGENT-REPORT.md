# Vigorly 30-agent product audit

Date: 19 August 2026

## What “30 test users” means

This audit uses 30 named, real-life **simulated product agents**. It does not create fake people or silently add 30 identities to the production Supabase project. Each agent has a device setup, life context, end-to-end journey, feature coverage, and a product-feedback question in `qa/agent-personas.mts`.

- 30/30 unique personas are defined and validated by a regression test.
- 28/28 feature areas have at least one owner.
- 15 journeys can be substantially exercised with code-level tests and static inspection.
- 15 require a physical iPhone, Apple Watch, email inbox, Health permissions, accessibility settings, or multiple devices. Those remain an explicit human-device test matrix; they are not falsely marked as physically passed.
- No production test accounts were created. The production project still has the three existing accounts observed during this audit.

## Coverage

The cohort covers onboarding, account creation, email confirmation, password recovery, offline use, guest/account migration, Supabase sync, Today, keyboard and voice quick-log, food estimates, learned foods, saved meals, water, weight, Health, goals, Progress, Coach, settings, backup/restore, account deletion, routines, exercise discovery, live workouts, timed sets, rest timers, history, Apple Watch, VoiceOver, and Dynamic Type.

Notable real-life cases include an Indian vegetarian speaking Hinglish, a vegan with allergies, a celiac label-reader, a night-shift nurse, a privacy-conscious offline user, a powerlifter using RPE and supersets, a rehab patient using timed holds, an older user with intermittent internet, an offline Watch user, a runner granting partial Health permissions, an account with an older password, and a VoiceOver user.

## Verification results

| Check | Result |
|---|---:|
| Unit and regression tests | 144/144 pass |
| Agent registry | 30/30 agents, 28/28 feature areas |
| Phrase handling sweep | 180/180 pass |
| Food/catalog sweep | 1,672/1,672 pass |
| Realistic logging sweep | 100/100 internally consistent |
| TypeScript | Pass |
| Expo lint | Pass |
| iOS production JS bundle | Pass, 1,873 modules |
| Watch Swift typecheck | Pass |
| Full iPhone + embedded Watch Release build | Pass; 126 native targets, embedded binary validated |
| Paired simulator launch | Pass; iPhone 17 + Watch Series 11, both bundles `1.0.0 (13)`, Watch receipt confirmed |
| Watch size smoke test | Pass at default text on 40 mm and 49 mm simulators; largest text remains physical P08 |
| Signed IPA audit | Pass; strict nested signature valid, phone/Watch `1.0.0 (14)`, production HealthKit entitlements, source markers verified |
| Native iOS UI tests | 2/2 pass in Release at maximum Dynamic Type on iPhone 17e simulator |
| Expo Doctor | 16/18; SDK versions pass, two documented architecture warnings remain |
| Supabase RLS | Enabled; own-row policies verified |
| Supabase performance advisor | No findings |

## Defects fixed during the audit

### Accounts, local data, and cloud sync

- Guest logs now merge into an account vault instead of being ignored when the returning account already has data.
- Foregrounding the app now pulls remote changes; sync is no longer push-only until relaunch/manual sync.
- Changes made while another sync is in flight are queued for another pass instead of being silently skipped.
- Water corrections can move downward, and zero is a valid value for steps/sleep. Per-metric timestamps replace `Math.max` and truthy-value merging.
- Existing accounts with a valid password shorter than the new ten-character signup rule can sign in.
- “Clear fitness history” now clears logs/history while preserving profile, goals, routines, custom items, and learned foods, as its confirmation promises.
- Backup restore now rejects unrelated or structurally damaged JSON before it can replace live state.
- Sign-out copy now states its all-device scope and local encrypted-copy behavior.

### Supabase security

- A production migration revoked anonymous and public execution of `save_user_app_data` and `delete_own_account`, retaining execution for authenticated users only.
- Live verification confirms anonymous execution is denied and authenticated execution is allowed.
- Row-level security remains enabled with authenticated own-row select/insert/update/delete policies.
- The remaining intentional advisor warning is the authenticated, security-definer self-delete RPC. It checks `auth.uid()` and must remain callable for self-service account deletion.
- The live Auth minimum was raised from 6 to 10 characters on 19 August 2026 and re-opened in the dashboard to verify the saved value, aligning direct API signups with the app's existing rule.
- Leaked-password protection remains disabled because the live project is on Supabase Free and the feature is available only on Pro and above. An owner must make the commercial upgrade decision before this can be enabled; no subscription purchase was inferred from the release-remediation scope. See [Supabase password security](https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection).

### Nutrition and quick logging

- A partly unknown meal is no longer silently saved as a partial meal. For example, rice plus an unknown edamame item now asks for clarification rather than dropping the edamame.
- Almond milk, gluten-free bread, and vegan “chicken” are protected from unsafe generic dairy/wheat/meat substitutions.
- Explicit label calories and protein/carbs/fat override a generic catalog estimate.
- Repeating a saved/recent meal from Today now uses the current meal period rather than silently restoring its original slot.
- Saved/recent repeats on Today and saved-meal repeats on Food now use the current meal period and expose immediate Undo.
- Logged food can be corrected after saving: name, quantity, calories, macros, meal slot, and date. Moving a food between days is conflict-safe and cannot duplicate it on the old day.
- The widest-estimate action now opens that exact logged food for correction rather than an empty logger.
- The estimate-review screen can return to the original transcript for correction before anything is saved.
- Active calories imported from Health are visible on Today with their source.
- Voice copy no longer claims guaranteed offline/on-device recognition or full Hinglish support when the operating system controls recognition and may use a network service.

### Workouts and Apple Watch

- A routine/settings edit can no longer generate a generic tombstone that deletes a newer offline Watch workout. Tombstones identify the exact cleared workout.
- RPE and superset IDs survive phone–Watch round trips.
- Measured Watch energy no longer renders an `undefined–undefined` estimate range in workout detail.
- A Watch timed set now retains exercise and set identity, so navigation or plan edits cannot make the timer complete a different set.
- The phone rest timer is stored in the active workout, so leaving and reopening the workout screen no longer resets it.
- Timed-set and rest-timer state now round-trips between phone and Watch, including absolute deadlines and set identity.
- Watch supports exercise search, duration wheels, completed-set reopening, and rejects a strength set with no reps.
- Custom exercises now have create, edit, search, sync, and safe-delete flows.
- A mixed strength-and-cardio workout launches and records as HealthKit cross training; single-mode sessions retain their specific activity type.
- The phone uses HealthKit's workout launch API when an active workout exists, which is the Apple-supported way to foreground the companion Watch app.
- Phone and Watch targets now receive the same version/build number from `app.json` before local iOS and EAS builds, preventing TestFlight companion-version drift.
- Phone and Watch now expose delivery state and the last successful synchronization time. The phone shows cloud and Watch status separately, and only marks Watch data synced after an explicit receipt from the Watch; the Watch shows connected/offline-ready state in both its routine library and workout overview.

### UX, Coach, and accessibility

- Coach is no longer a placeholder. It provides deterministic private/offline replies, prompt chips, persistence, and a clear general-guidance disclaimer.
- Shared auth, recovery, account, and goal inputs now expose meaningful accessibility labels.
- A misleading Face ID usage description was removed because the app does not implement biometric authentication.
- Zero weekly strength days now persists instead of being replaced by a default.
- Onboarding now asks activity, pace, training availability, experience, diet, meal pattern, allergies, injuries, challenge, and coaching tone before calculating recommended targets.
- The same plan preferences can be edited later in Settings without restarting onboarding.
- Coach answers trend/progress and “why this target?” questions from logged data and the plan method, while retaining its non-medical boundary.
- Health cards now show provider, newest sample time, and which of steps, energy, sleep, and weight actually returned samples.
- Workout KG, reps, and duration fields now have explicit VoiceOver labels.
- Speech uses the device locale (including `en-IN`) and prefers an installed offline recognition pack when available; privacy copy states when the OS may use its speech service.
- Auth typography now separates decorative display scaling from readable content scaling, preventing maximum Dynamic Type from turning the header into clipped partial words on small screens while preserving scroll access and 200% body/input scaling.
- Editable screens dismiss the software keyboard when the user drags the form; auth fields also expose a Done return action, and Coach dismisses after sending. Native UI coverage verifies the signup behavior.

## Agent feedback: what is still missing

These are the most valuable next product changes, ordered by risk and impact.

### Build and dependency notes

- Expo, Expo Constants, and Expo File System were aligned to the exact SDK 54 patch versions requested by Expo Doctor.
- Doctor still warns that app config is not automatically synchronized because checked-in native projects exist. That is expected for the custom Watch target, but it means future `app.json` capability/privacy changes must also be reviewed in Xcode rather than assuming EAS prebuild will apply them.
- Doctor flags the `expo-health-connect` config plugin as unmaintained. The actual Android runtime API is `react-native-health-connect` 3.6.0, but replacing the config path requires native Android validation before removal.
- A non-breaking `npm audit fix` reduced the audit from 31 to 28 transitive findings (14 high, 14 moderate, zero critical). The remaining advisories are in Expo/Metro/config build dependencies; npm's proposed automatic repair is a breaking jump to Expo 57. No forced framework upgrade or unsafe downgrade was made in this audit.

### Release-critical device validation

1. Run the 15 hardware journeys on the exact TestFlight build using at least two iPhones and two Watch sizes. This must include email deep links, offline Watch completion/reconnection, partial Health permissions, VoiceOver, 200% Dynamic Type, backgrounding during set/rest timers, and immediate finish after the last set.
2. Expand the native suite beyond its auth reachability and keyboard-dismissal coverage. Watch taps, HealthKit delivery, WatchConnectivity timing, and OS permission sheets still require physical-device/integration coverage.

### Remaining architectural limits, not hidden defects

1. **Physical-device evidence is still required.** WatchConnectivity timing, HealthKit delivery, permission sheets, email deep links, VoiceOver focus order, and 200% Dynamic Type cannot be truthfully certified by Node/TypeScript/Swift compilation.
2. **Clock-skew-resistant sync is a future backend evolution.** Common corrections and moves are protected by per-field timestamps and tombstones, but a server-ordered operation log would be stronger when two devices have badly incorrect clocks.
3. **Coach remains deterministic.** It now uses goals, constraints, current logs, weekly insights, and the plan calculation. It deliberately does not imitate a clinician or invent external facts.
4. **Dependency upgrade requires its own native release.** Remaining npm advisories are transitive Expo/Metro build tooling. The automated repair is a breaking Expo 57 jump and must not be mixed into this stability release.

## Persona-by-persona acceptance focus

| Agents | Primary acceptance focus |
|---|---|
| A01–A05 | Personalized goals, culturally accurate nutrition, dietary safety, advanced sets, rehab timers |
| A06–A10 | Night-shift meal slots, offline privacy, two-device merge, undo/corrections, label macros |
| A11–A15 | Voice fallback, VoiceOver, Dynamic Type, Watch corrections, measured Health energy |
| A16–A20 | Offline Watch sync, no silent food omissions, backup conflicts, mixed workouts, midnight/zero corrections |
| A21–A25 | Partial Health permissions, older passwords, email deep links, timer identity, estimate editing |
| A26–A30 | Zero-value goals, Coach usefulness, accurate deletion wording, data deletion, full-product friction sweep |

The executable persona definitions and exact journeys are in `qa/agent-personas.mts`. The registry and route/native-file checks are in `tests/agent-personas.test.mts`.

## Release judgment

The implementation backlog identified by the audit is closed and all automated gates pass. Production candidate `1.0.0 (14)` completed on EAS as build `587289eb-10f0-4dd7-bb95-acf74a3c2eba`; its TestFlight submission is tracked as `8a1df61e-c436-4d8f-acb0-d690968149ce` and remains `IN_QUEUE` during Expo's unresolved “iOS Submissions hanging on App Store Connect build uploads” incident. The app is **ready for the controlled TestFlight acceptance run in `RELEASE-ACCEPTANCE.md` once Apple finishes processing build 14**, not yet for an unqualified public release. Public release requires recorded passes for the 15 physical-device journeys—particularly WatchConnectivity, HealthKit, email deep links, VoiceOver, and large-text behavior. Supabase’s Expo guide also requires native deep-link handling for email verification and recovery, which is implemented but still needs real inbox/device confirmation: [Supabase Expo user-management guide](https://supabase.com/docs/guides/getting-started/tutorials/with-expo-react-native).
