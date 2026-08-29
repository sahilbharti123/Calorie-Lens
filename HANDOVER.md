# Vigorly engineering and product handover

- **Prepared:** 29 August 2026
- **Repository:** `https://github.com/sahilbharti123/Calorie-Lens`
- **Local project root:** `/Users/sahilbharti/Desktop/Docs/Apps/Calorie-Lens`
- **Active branch at handover:** `codex/release-readiness-and-calorie-fixes`
- **Mobile application root:** `/Users/sahilbharti/Desktop/Docs/Apps/Calorie-Lens/mobile`

This is the authoritative continuity document for another coding model or engineer. It consolidates the work completed across the long Vigorly product session, the current architecture, operational details, release evidence, unresolved risks, and the exact rules that must not be violated.

## 1. Read this first

1. The customer-facing product is **Vigorly**, even though the repository and several persisted keys retain the old `Calorie-Lens` name for compatibility.
2. The primary deliverable is the Expo/React Native app in `mobile/`. The root Python/Streamlit application and FastAPI account service are legacy/optional companion code. Current native account creation and cloud sync use **Supabase**, not the Python account service.
3. The iOS project is intentionally committed because it contains a manually maintained native HealthKit bridge, UI tests, and an Apple Watch app. **Never run `expo prebuild --clean`.** It can destroy the Watch target and native bridge.
4. Current source metadata is Vigorly `1.0.0 (15)`. Repository evidence proves a signed build-14 IPA and records a build-number bump to 15, but it does not by itself prove build 15 finished App Store Connect processing. Check EAS and App Store Connect before making that claim.
5. Automated gates are strong, but all physical-device acceptance rows in `mobile/RELEASE-ACCEPTANCE.md` still require recorded passes on the exact TestFlight candidate. Do not convert simulator/unit evidence into claims about real WatchConnectivity, HealthKit, email links, VoiceOver, or maximum Dynamic Type.
6. Do not add a `service_role` Supabase key, Apple password, two-factor code, signing private key, or any other secret to Git. The mobile app needs only the Supabase project URL and publishable key.
7. The native app does **not** use AWS Bedrock. Normal parsing and coaching are deterministic and local. Optional legacy remote parsing uses the repository's API path only when explicitly enabled.

## 2. Product intent

Vigorly is an offline-first private fitness ledger combining:

- food, calories, macros, recipes, meal history, saved meals, and user-taught foods;
- water, steps, sleep, active energy, and weight;
- personalized calorie/macro/activity targets based on profile and target weight;
- routines, strength sessions, timed exercises, rest timers, records, and charts;
- Apple Health, Health Connect, and a full Apple Watch workout logger;
- optional Supabase accounts for backup and cross-device synchronization;
- deterministic private coaching based on the user's plan and logs.

The operating product principle is **“glance first, evidence always.”** Estimates should expose their range, source, assumptions, and confidence. Unknown data must trigger clarification or manual input, not a fabricated precise value.

The competitive direction came from StepsApp and Hevy:

- Borrow StepsApp's glanceable daily state, visual progress, calm motivation, adjustable goals, continuity, and Watch usefulness.
- Avoid review complaints around opaque sync, declining totals, background failures, aggressive monetization, ads, and unexplained wearable latency.
- Borrow Hevy's fast routine/session model, previous-performance display, weight/reps controls, rest timers, records, and true Watch-side logging.
- Do not visually clone either product.

`VIGORLY_PRODUCT_PLAN.md` contains the original product operating plan and StepsApp research. Treat it as product context; use this handover plus current code for implementation truth.

## 3. What was completed during the session

### 3.1 Initial product/UI audit and competitor research

- The codebase and primary mobile flows were audited for information hierarchy, excessive text, weak visual prioritization, permissions, and logging friction.
- StepsApp's positive and negative review patterns were converted into product principles in `VIGORLY_PRODUCT_PLAN.md`.
- The Midnight Athlete design system was established around a near-black canvas, electric-lime accent, high-contrast metrics, rings/bars/charts, restrained cards, and fewer competing text blocks.
- Shared primitives live in `mobile/src/components/ui.tsx`; tokens live in `mobile/src/theme.ts`.
- App-store screenshots and visual references live in `mobile/store/` and the conversation attachments, but screenshots are reference material, not specifications.

### 3.2 Onboarding, recommendations, and targets

The original onboarding did not gather enough information to justify personalized targets. It now collects:

- primary goal;
- equation option, age, height, current weight, and target weight;
- goal pace and activity level;
- workout preference, experience, training days, and available minutes;
- diet style, meal count, allergies, injuries, main challenge, and coaching tone;
- optional portion calibration and Health/voice setup.

Target-weight validation respects goal direction. Recommended calories, macros, water, steps, weekly workout minutes, and strength days are calculated in `mobile/src/lib/personalization.ts`. Users can override targets; an override changes plan provenance to “Manually adjusted in profile settings.” A later weight update recalculates the plan unless it was manually adjusted.

The voice setup action now changes from **Turn on voice** to **Voice configured** when permission/configuration succeeds. It no longer behaves like a dead “Test voice” button.

### 3.3 Progress and weight logging repairs

- Dedicated water and weight routes exist rather than routing every action through an ambiguous generic chat.
- Phrases such as “current weight is 89” use dedicated weight parsing and reject implausible values.
- Keyboard and voice input pass through the same deterministic logging path.
- Clarification suggestions that choose “Weight” navigate to the weight logger instead of re-asking “What would you like me to log?”
- Progress presents weight trend, target weight context, Health provenance/freshness, and correction paths.
- Trends distinguish missing data from zero.

Relevant files: `mobile/app/(tabs)/progress.tsx`, `mobile/app/weight-log.tsx`, `mobile/src/lib/weight.ts`, and `mobile/tests/weight.test.mts`.

### 3.4 Supabase accounts and cloud storage

The mobile app now supports:

- email/password signup;
- optional email confirmation;
- sign-in;
- resend-confirmation flow;
- password-recovery deep links;
- password changes after reauthentication;
- global sign-out with local fallback;
- self-service account deletion;
- explicit guest/offline mode;
- merging existing guest data into the signed-in account rather than discarding either side.

- Auth implementation: `mobile/src/store/auth-store.tsx` and `mobile/src/lib/supabase-auth.ts`.
- Client setup: `mobile/src/lib/supabase.ts`.
- Cloud persistence: `mobile/src/lib/supabase-sync.ts`.
- Database migrations: `supabase/migrations/`.

The database has one `public.user_app_data` row per Auth user. RLS limits rows to `auth.uid()`. `save_user_app_data` implements optimistic version concurrency. `delete_own_account` removes the Auth user, with the fitness row removed by foreign-key cascade. The second migration revokes public/anonymous RPC execution.

Supabase settings last verified during the release audit on 19 August 2026:

- minimum password length matched the app's 10-character signup rule;
- own-row RLS was enabled;
- anonymous execution of the save/delete RPCs was denied;
- authenticated execution remained allowed;
- leaked-password protection was unavailable on the then-current Free plan.

Those are historical verified facts, not a guarantee that dashboard state has not changed. Re-check before release. Redirect URLs must include:

```text
vigorly://auth
vigorly://auth-reset
```

Custom SMTP is strongly recommended before external/public testing so confirmation and recovery email are reliable. Testers do not need to be manually inserted into Supabase when signup is enabled.

### 3.5 Local storage and sync reliability

- The native vault is serialized into SQLite KV storage and encrypted with AES-GCM.
- The 256-bit device key is stored in SecureStore/Keychain as `WHEN_UNLOCKED_THIS_DEVICE_ONLY`.
- Persisted keys retain the old `calorie-lens.*` prefix intentionally. Renaming them would orphan installed-user data.
- Unreadable restored ciphertext is discarded and reported rather than freezing app hydration forever.
- Writes are debounced by 300 ms but flushed on background/inactive, scope changes, and provider teardown.
- Foregrounding triggers a cloud pull, so remote edits no longer stay stale until a local edit or relaunch.
- A second change made during an in-flight sync queues another pass.
- Version conflicts merge and retry instead of overwriting.
- Per-metric timestamps let water move downward and preserve valid zeros for steps/sleep.
- Meal relocation is conflict-safe; one meal cannot survive on two dates after a move.
- Deleted meals, workouts, saved meals, learned foods, routines, sessions, and custom exercises use bounded tombstone lists to prevent resurrection.

Core logic is in `mobile/src/store/app-store.tsx`, `mobile/src/lib/secure-storage.ts`, `mobile/src/lib/sync-merge.ts`, and `mobile/src/lib/training.ts`.

### 3.6 Nutrition and quick logging

Current nutrition behavior is deterministic and local by default:

- 170 bundled food references: 29 USDA-tier records and 141 intentionally broad “typical” records.
- Natural-language logging supports English, common Hindi/Hinglish phrasing, counts, grams, kilograms, millilitres, litres, cups, bowls/katoris, glasses, handfuls, packets, pints, pegs, and common implicit servings.
- Food output carries calories, macros, range, confidence, basis, source, and assumptions.
- Unknown foods ask for label calories/macros or ingredient amounts instead of being silently discarded or mapped to a generic substitute.
- Protected dietary qualifiers prevent almond milk becoming dairy milk, gluten-free bread becoming wheat bread, and vegan “chicken” becoming chicken.
- Explicit label calories/macros override a generic catalog match.
- Longer compound aliases win so mango lassi, coconut water, rajma chawal, and similar phrases are not double-counted as their component words.
- User-supplied label facts become learned foods and can later be renamed, corrected, or forgotten.
- Saved/recent meal repeats use the current meal period and offer immediate undo.
- Existing logged foods can be edited for name, amount, calories, macros, meal slot, and date.
- The estimate-review screen can return to the original transcript before saving.

#### Composite dishes and the poha/coffee fixes

The calorie engine now treats recipe dishes as one parent dish rather than logging every word as a separate food:

- “poha with onion” asks for the amount of finished poha instead of logging poha and onion separately;
- a bare “poha” remains quick to log using the broad dish estimate;
- review exposes **Ingredients for the portion you ate**;
- ingredient refinement requires amounts and sums calories/macros into one parent meal item;
- ingredient evidence is stored in `MealItem.recipe` and remains editable;
- omitted ingredient amounts are not invented;
- “coffee without sugar,” “coffee no sugar,” and “unsweetened coffee” stay near zero;
- “coffee with milk without sugar” maps to the unsweetened milk-coffee entry rather than adding sugar;
- negative ingredient modifiers generalize beyond sugar.

After a successful log, the same quick-log screen presents **Log another dish** or **Log another update**, so the user does not need to return to the main tab between dishes.

Core files: `mobile/src/lib/nutrition.ts`, `mobile/src/lib/food-catalog.ts`, `mobile/src/lib/food-typical.ts`, `mobile/app/quick-log.tsx`, `mobile/app/edit-meal.tsx`, and `mobile/tests/logging.test.mts`.

Remote parsing is disabled unless `EXPO_PUBLIC_ENABLE_AI_PARSING=1`. Even then it requires the optional API URL and a signed-in session. No remote model is allowed to directly invent a calorie. There is no AWS Bedrock integration. The root Python/Streamlit companion may use Gemini only under its separate explicit legacy flags.

### 3.7 Voice logging

- Uses `expo-speech-recognition` and the operating system recognizer.
- Vigorly does not upload audio to its own server.
- The selected device locale is used; Indian locales use `en-IN`.
- Installed on-device recognition is required when the OS reports a matching installed locale; otherwise the OS may use its speech service.
- The recognizer receives a bounded vocabulary built from foods, exercises, and domain units.
- One tap produces one update; final and end events are guarded from double submission.
- Dictation stops after 40 seconds or approximately 2.6 seconds of silence after speech.
- Network, permission, no-speech, and unsupported-device failures all expose keyboard fallback.

Do not claim voice is universally offline. The operating system decides based on device, locale, installed speech packs, and platform behavior.

### 3.8 Coach

Coach was initially a placeholder and is now implemented as a deterministic, private, offline feature:

- prompt chips;
- persisted message history;
- answers based on profile, targets, current day, weekly logs, diet, allergies, injuries, and coaching tone;
- explanations for targets and trend questions;
- a clear general-guidance/non-medical boundary.

Relevant code: `mobile/app/(tabs)/coach.tsx`, `mobile/src/lib/personalization.ts`, `mobile/src/lib/insights.ts`, and coach fields in `mobile/src/types.ts`.

### 3.9 Workout system on iPhone

The Train tab supports:

- reusable routines and template routines;
- routine folders, notes, rep ranges, weight defaults, rest duration, supersets, and set types;
- live sessions started empty or from a routine;
- weight/reps, reps-only, and duration-set logging;
- previous performance and prefill;
- warm-up, failure, drop, and normal sets;
- optional RPE;
- rest countdown with adjustment and skip;
- absolute-deadline timed sets that survive navigation/backgrounding;
- adding, replacing, reordering, or deleting exercises and sets;
- custom exercises with create/edit/search/safe deletion;
- PR detection for heaviest weight, estimated 1RM, best set volume, reps, and duration;
- workout history, per-exercise charts, records, weekly muscle volume, and energy estimates;
- measured Apple Watch active energy when available, with Compendium-based estimation otherwise;
- mixed strength/cardio classification as HealthKit cross training.

Phone rest and timed-set state live in the active workout model, not component-only state. Active workouts use revision and timestamp fields for phone/Watch convergence.

Core files: `mobile/app/(tabs)/train.tsx`, `mobile/app/routine-editor.tsx`, `mobile/app/workout-session.tsx`, `mobile/app/workout-history.tsx`, `mobile/app/workout/[id].tsx`, `mobile/src/store/workout-store.tsx`, `mobile/src/lib/training.ts`, `mobile/src/lib/set-timer.ts`, and `mobile/src/lib/workout-activity.ts`.

### 3.10 Exercise library expansion

The built-in library grew from 100 to **262** curated movements. Current primary-equipment counts:

| Equipment | Exercises |
|---|---:|
| Barbell | 42 |
| Dumbbell | 45 |
| Machine | 36 |
| Bodyweight | 58 |
| Cable | 32 |
| Kettlebell | 16 |
| Resistance band | 16 |
| Other/specialty | 17 |

The extension includes broad push/pull/legs coverage, unilateral work, bands, kettlebells, suspension/TRX aliases, sleds, Air Bike, SkiErg, Olympic-lift variants, core progressions, and many machine/free-weight variations.

`Biceps Curl 21s` includes aliases for `21s`, `21's`, `bicep 21s`, `barbell 21s`, `curl 21s`, and `twenty ones`, with correct 7 lower-half + 7 upper-half + 7 full-range instructions.

Exercise search now normalizes punctuation and token-matches across:

- canonical name;
- aliases;
- primary and secondary muscles;
- equipment;
- logging kind (`timed`, `reps`, `weight`, `kg`).

An unmatched search can immediately open a prefilled custom-exercise form. The full built-in and custom catalog is sent to Apple Watch.

Important content limitation: the original 100 exercises have the richest photo/how-to/video content. The 162 extended entries have concise instructions, coaching tips, and a reusable animated figure fallback, but most do not yet have individual photo pairs or the large generated guide/video record in `exercise-guides.ts`. Do not claim all 262 have bespoke videos and photo demonstrations.

Core files: `mobile/src/lib/exercises.ts`, `mobile/src/lib/exercise-catalog-extra.ts`, `mobile/app/exercise-picker.tsx`, `mobile/app/custom-exercise.tsx`, and `mobile/tests/exercises.test.mts`.

### 3.11 Apple Watch implementation

The Watch app is not a metrics-only companion. It can:

- receive routines, previous performance, active workout state, custom exercises, and the exercise catalog;
- start a synced routine or a quick empty workout on Watch;
- search/add exercises;
- add/remove sets;
- edit kg, reps, duration, and RPE with always-present Digital Crown controls;
- complete and reopen sets;
- reject an empty/zero-rep strength completion;
- run, pause, resume, adjust, or finish timed sets and rest timers;
- retain timer identity so plan edits cannot complete the wrong set;
- see notes and prior values;
- stream live heart rate and active calories from HealthKit;
- finish offline, persist the snapshot, and replay after reconnecting.

Phone and Watch exchange complete, revisioned snapshots rather than relying on a fragile stream of individual field mutations. RPE and `supersetId` round-trip. A workout-specific tombstone prevents unrelated routine/settings edits from deleting a newer offline Watch workout. Delivery UI differentiates “queued” from an explicit Watch receipt.

Opening the iPhone app cannot arbitrarily open an unrelated Watch app because watchOS prohibits that. When an active workout exists, Vigorly uses Apple's supported HealthKit workout-launch handoff; otherwise foregrounding the phone refreshes Watch application context.

Native files:

- `mobile/ios/VigorlyWatch/VigorlyWatchApp.swift`
- `mobile/ios/VigorlyWatch/WatchWorkoutManager.swift`
- `mobile/ios/VigorlyWatch/WorkoutView.swift`
- `mobile/ios/Vigorly/VigorlyWorkoutBridge.swift`
- `mobile/ios/Vigorly/VigorlyWorkoutBridge.m`
- `mobile/src/components/watch-workout-sync.tsx`
- `mobile/src/lib/live-workout.ts`
- `mobile/src/lib/watch-workout-contract.ts`

- Phone bundle ID: `com.advaice.calorielens`.
- Watch bundle ID: `com.advaice.calorielens.watchkitapp`.
- Apple team: Advaice Limited, `95R5R7A683`.

### 3.12 Visual polish, accessibility, and keyboard behavior

- Major surfaces were reduced from dense prose toward rings, charts, compact metrics, progress visuals, icons, and clearer hierarchy.
- Forms and editable screens dismiss the keyboard on drag where appropriate.
- Auth inputs expose Done behavior and remain scrollable.
- Display typography and content typography scale separately so maximum Dynamic Type does not turn headings into clipped fragments on small screens.
- Shared `Tap` behavior and controls target 44-point interaction areas.
- Reduced-motion preferences are honored by shared animation primitives.
- VoiceOver labels were added to auth, goals, health, workout kg/reps/duration, destructive actions, and key status controls.
- Color is not intended to be the only status signal.

The UI is significantly improved but should still be judged on real devices. The product owner has repeatedly emphasized that small interaction bugs and text-heavy screens are unacceptable.

### 3.13 Thirty-agent product audit

`mobile/qa/agent-personas.mts` defines 30 named simulated real-life product agents covering 28 feature areas. These are test personas, not fake production accounts. No 30 Supabase users were created.

The cohort includes Indian/Hinglish logging, vegan/allergy/celiac cases, night-shift meal timing, offline privacy, powerlifting/RPE/supersets, rehab timers, older users, offline Watch workouts, partial Health sharing, VoiceOver, and maximum Dynamic Type.

The audit fixed cloud merge, nutrition, Watch snapshot, timer, onboarding, Coach, accessibility, and release problems. `mobile/QA-30-AGENT-REPORT.md` provides the full defect ledger and feedback. Its historical test counts and build status may be older than this handover.

## 4. Current architecture

### 4.1 Runtime stack

- Expo SDK `~54.0.37`
- React Native `0.81.5`
- React `19.1.0`
- Expo Router with typed routes
- React Native Reanimated
- Expo SQLite KV storage
- Expo SecureStore
- Supabase JS `^2.111.0`
- `@kingstinct/react-native-healthkit` for iOS HealthKit and workout launch
- `react-native-health-connect` plus Expo configuration on Android
- `expo-speech-recognition`
- Native Swift watchOS target and iOS React Native bridge

See `mobile/package.json` for exact dependency versions. Do not apply npm's breaking automatic audit repair or upgrade Expo in the same change as a product fix.

### 4.2 Provider and navigation hierarchy

`mobile/app/_layout.tsx` owns the root navigation shell and provider order. At runtime:

1. auth state determines signed-in vs guest scope;
2. `AppProvider` hydrates the encrypted local vault and cloud version;
3. `WorkoutProvider` exposes training mutations over the same `AppData` object;
4. `WatchWorkoutSync` stays mounted at app-shell level so Watch-originated events are accepted outside the workout screen;
5. Expo Router selects auth, onboarding, tabs, or modal/detail routes.

Primary tabs are Today, Food, Train, Progress, and Coach.

### 4.3 Data model

The complete persisted schema is `AppData` in `mobile/src/types.ts`. Major branches:

- `goals`
- `profile`
- `plan`
- `estimation`
- `days`
- `weights`
- `coachMemory` / `coachMessages`
- saved meals and learned foods
- deletion tombstones
- `training`
- Health freshness/status

`TrainingData` contains routines, completed sessions, one active session, custom exercises, delete tombstones, default rest, RPE preference, and the active-workout tombstone.

All additions must normalize missing old fields in `normalizeData` or `normalizeTraining`. Never assume every installed vault was written by the newest schema.

### 4.4 Sync precedence

- Collections merge by stable ID plus updated timestamp and deletion tombstones.
- Day metrics select the newest per-field timestamp; legacy values without timestamps preserve local behavior.
- Meal movement resolves one canonical location.
- Training merges routines/customs by newest update, sessions by ID, and active workouts by revision then edit time.
- Supabase row version protects the whole cloud snapshot from stale writes.

This is robust for ordinary devices. It is not a server-ordered operation log and remains vulnerable to severe device clock skew in sufficiently complex simultaneous edits. A server sequence/operation log is a future architecture improvement, not something to improvise inside a UI fix.

### 4.5 Health behavior

- iOS reads/writes through Apple Health and launches the Watch workout using HealthKit.
- Android reads through Health Connect.
- Health cards expose provider, status, last attempt/success, latest sample, and per-category sample counts.
- Partial sharing preserves old values for unshared categories rather than replacing them with zero.
- Imported active calories appear separately from consumed calories.
- Manual logging always remains available.

Simulator Health data is not proof of physical behavior.

## 5. Repository map

| Path | Purpose |
|---|---|
| `mobile/app/` | Expo Router screens and navigation |
| `mobile/src/components/` | Shared UI, charts, figures, Watch sync shell |
| `mobile/src/lib/` | Nutrition, exercises, personalization, sync, health, speech, workout math |
| `mobile/src/store/` | Auth, application data, and workout providers |
| `mobile/src/types.ts` | Persisted/domain schema |
| `mobile/tests/` | Node/TypeScript regression suite |
| `mobile/qa/` | Thirty-agent simulated journey registry |
| `mobile/scripts/` | Catalog sweeps and native-version synchronization |
| `mobile/ios/Vigorly/` | iPhone native target and HealthKit/Watch bridge |
| `mobile/ios/VigorlyWatch/` | Watch application |
| `mobile/ios/VigorlyUITests/` | Native iOS UI tests |
| `mobile/assets/exercises/` | Original exercise demonstration photos |
| `supabase/migrations/` | Auth-scoped cloud-vault schema/security |
| `calorie_engine/` | Legacy/shared Python calorie engine |
| `app.py` | Streamlit companion |
| `api.py`, `account_store.py` | Legacy optional FastAPI/SQLite account service |
| `tests/` | Python engine/account/health tests |
| `scripts/archive.sh` | Local Apple archive/export flow |
| `scripts/verify-signing.sh` | IPA signature/profile verification |

## 6. Environment and local setup

### 6.1 Required mobile environment

`mobile/.env` exists locally and is ignored by Git. It currently defines these names; values must never be copied into documentation or commits:

```dotenv
EXPO_PUBLIC_SUPABASE_URL=...
EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY=...
EXPO_PUBLIC_API_URL=...
```

The Supabase publishable key is intended for the client, while security is enforced by Auth and RLS. Never use the service-role key in an `EXPO_PUBLIC_*` variable.

`EXPO_PUBLIC_API_URL` is optional legacy functionality. `EXPO_PUBLIC_ENABLE_AI_PARSING=1` is also optional and off by default.

### 6.2 Install and run

```bash
cd /Users/sahilbharti/Desktop/Docs/Apps/Calorie-Lens/mobile
npm install
npm run ios
# or
npm run android
```

Expo Go is insufficient for HealthKit, Health Connect, speech, and the Watch bridge. Use a development build/native run.

### 6.3 Before editing Expo/native configuration

Read the exact Expo SDK 54 documentation, as required by `mobile/AGENTS.md`. Because iOS is checked in and hand-maintained, any `app.json` capability/privacy/plugin change must be reviewed against the native project. Do not assume a prebuild will safely synchronize it.

## 7. Automated verification

Run from `mobile/`:

```bash
npm run sync:native-versions
npm test
npm run lint
npm run typecheck
npm run check:phrasings
npm run check:catalog
npm run check:logging
npx expo export --platform ios --output-dir /tmp/vigorly-ios-export
```

At this handover, the latest complete run produced:

- 158/158 Node/TypeScript tests passing;
- 180/180 phrasing cases;
- 1,691/1,691 catalog sentences;
- 100/100 realistic logging examples with self-consistent output;
- Expo lint passing;
- TypeScript passing;
- 23/23 Python account, calorie-engine, and Health-import tests passing;
- the Watch Swift sources typechecking against the watchOS 10 SDK;
- iOS Metro production export passing with 1,873 modules.

Test domains include auth links, backup validation, food plausibility, learned foods, recipe behavior, negative modifiers, meal repeats/moves, personalization, water/zero merge semantics, timers, Watch snapshots, native version sync, weight parsing, workout activity classification, and the 30-agent registry.

For native release candidates also run the commands in `mobile/RELEASE-ACCEPTANCE.md`: Swift typecheck, Xcode Release build, and UI tests. Then execute every P01–P15 row on physical devices.

## 8. Build, signing, and TestFlight

Current identifiers:

| Item | Value |
|---|---|
| Display name | Vigorly |
| Expo slug | `vigorly` |
| URL scheme | `vigorly` |
| iOS version/build in source | `1.0.0 (15)` |
| Android package | `com.advaice.calorielens` |
| Android version code | `2` |
| iPhone bundle ID | `com.advaice.calorielens` |
| Watch bundle ID | `com.advaice.calorielens.watchkitapp` |
| Apple team | Advaice Limited `95R5R7A683` |
| App Store Connect Apple ID | `6796998952` |
| Expo owner | `sahilbhartis-team` |
| EAS project ID | `e90be1b1-ccef-45bb-a981-b2aeb8b62aab` |

`mobile/eas.json` contains a production build profile with auto-increment and a production submit profile with the ASC app ID.

The last signed artifact fully documented in the repository is build 14. `mobile/RELEASE-ACCEPTANCE.md` records its EAS build ID, submission ID, signature checks, hash, entitlements, and the then-current Expo submission incident. Commit `8560bd3` later changed source metadata to build 15. Before any new upload:

1. inspect EAS build/submission history;
2. inspect App Store Connect/TestFlight processing state;
3. confirm whether build 15 already exists;
4. increment to a number Apple has not seen if making a new binary;
5. run `npm run sync:native-versions` so phone and Watch match;
6. run automated gates;
7. verify the signed IPA itself, not merely the archive log;
8. avoid duplicate submissions while another submission is active.

Never expose Apple credentials or two-factor codes in logs or handover documents.

`TESTFLIGHT.md` describes the early local build-2 workflow and is historical. Its advice to run `expo prebuild --clean` is no longer safe now that the Watch target exists. `TESTFLIGHT-TEST-INFO.md` also contains stale product text saying Coach is a placeholder. Use current code, this handover, and `mobile/RELEASE-ACCEPTANCE.md` instead.

## 9. Known limitations and unfinished acceptance

### Release-critical human validation

The following still require real hardware and accounts:

- account creation with a real external inbox, confirmation, sign-in, recovery, and new password;
- guest-to-existing-account merge across two phones;
- foreground cloud pull and downward corrections on a second phone;
- phone-started workout auto-launch on paired Watch;
- Watch-only workout with kg/reps/RPE/timed sets, reconnect, and exactly-once completion;
- cross-device pause/resume/finish of timed sets and rest timers;
- 40 mm and large Watch layouts with largest text;
- partial Health permissions and sample freshness;
- live measured calories and heart rate;
- VoiceOver traversal on phone and Watch;
- 200% Dynamic Type on a small physical iPhone;
- backup/restore, malformed-backup rejection, clear history, and account deletion.

Do not call the app public-release-ready until those rows are recorded as passing.

### Product/content limitations

- Nutrition estimates remain estimates. Recipe ingredients, cooked state, oils, and actual portion size can dominate error.
- Only 29 food records are the narrow USDA tier; typical entries deliberately carry wider uncertainty.
- The deterministic Coach is useful but is not a clinician or an open-ended knowledge model.
- Speech can depend on the OS/network when no on-device language model is installed.
- The extended exercise catalog lacks bespoke photo/video guide assets for most new entries.
- WatchConnectivity delivery timing remains platform-controlled.
- No widget/complication work should begin until freshness and sync are measured on real users.
- Android Health Connect configuration still carries a documented Expo Doctor warning and needs a dedicated native upgrade/validation project.
- Remaining npm audit findings are transitive Expo/Metro build-tool issues; the automated fix proposes a breaking framework jump.

### Documentation debt

Some older documents still say “85+” or “100” exercises, describe the Python service as the account backend, or describe Coach as a placeholder. They are historical. Update or archive them in a focused documentation pass rather than assuming they describe current behavior.

## 10. Non-negotiable engineering rules

1. Preserve user changes in a dirty worktree and inspect `git diff` before editing overlapping files.
2. Use `apply_patch` for source/document edits.
3. Never run destructive Git commands or `expo prebuild --clean`.
4. Never silently guess an unknown food, missing ingredient amount, body weight, or workout detail.
5. Typed and spoken logging must use the same parser/review path.
6. Never let a partly unknown multi-food command save only the known fragment.
7. Preserve ranges, confidence, assumptions, source IDs, recipe evidence, and provenance through save/repeat/edit/sync.
8. Any new persisted field needs backward-compatible normalization and merge semantics.
9. Any cross-device deletion or move needs a tombstone or equivalent conflict rule.
10. Phone/Watch changes must preserve stable workout, exercise, set, timer, RPE, and superset identity.
11. Queueing a Watch context is not proof the Watch received it; only an explicit receipt is “synced.”
12. Do not claim an app launch can arbitrarily force-open the Watch app. Use the supported workout handoff only when a workout is active.
13. Do not claim simulated agents are real production users or that simulator tests are physical-device tests.
14. Do not add advertising or sell health data.
15. Do not use guilt-based calorie streaks or punish users for eating over a target.
16. Do not add AWS Bedrock unless the product owner explicitly authorizes a new architecture; it is not part of the current app.

## 11. Recommended next sequence

1. Confirm this branch and handover are visible on GitHub.
2. Check EAS and App Store Connect to establish the exact build-15 state.
3. If needed, create the next unused build and submit one artifact only.
4. Run P01–P15 against that exact TestFlight build, attach evidence, and record device/OS/build.
5. Fix only defects reproduced by those journeys; do not mix an Expo major upgrade into the release.
6. Add bespoke guides/photos or reviewed video links for the 162 extended exercises in batches, with source and collision tests.
7. Refresh stale README/TestFlight/App Store copy after physical behavior is known.
8. Configure and verify production SMTP before external/public account testing.
9. Decide commercially whether Supabase Pro/leaked-password protection is warranted.
10. After acceptance passes, merge the release branch through the owner's preferred GitHub workflow and tag the shipped commit.

## 12. Useful evidence and historical documents

- `VIGORLY_PRODUCT_PLAN.md` — product promise and StepsApp findings.
- `mobile/QA-30-AGENT-REPORT.md` — detailed simulated-persona audit and defects fixed.
- `mobile/RELEASE-ACCEPTANCE.md` — physical-device matrix and build-14 signed-artifact evidence.
- `SUPABASE_SETUP.md` — schema and dashboard setup.
- `ACCURACY.md` — food/exercise sources and estimate methodology.
- `PRIVACY.md` — privacy posture.
- `APP_STORE.md` — listing/review preparation; some feature copy may be stale.
- `TESTFLIGHT-TEST-INFO.md` — historical beta copy and email-delivery notes; update before reuse.
- `TESTFLIGHT.md` — historical early local archive workflow; never follow its prebuild-clean instruction now.
- Git commits `6b4936f` and `8560bd3` — release-readiness implementation and build-15 metadata.

## 13. Handover completion checklist

Before declaring a future task complete, the next model should answer:

- Did I inspect the current branch and dirty worktree?
- Did I preserve the manually maintained iOS/Watch targets?
- Did I avoid exposing secrets?
- Did I update normalization and merge behavior for persisted fields?
- Did I add a regression test for the exact user-reported sentence/action?
- Did I run unit tests, lint, TypeScript, phrase/catalog/logging sweeps, and iOS export?
- If native behavior changed, did I build/typecheck the native targets?
- If I claimed a release/device result, do I have evidence from that exact build and device?
- Did I distinguish implemented behavior from remaining physical acceptance?
- Did I update this handover when architecture or release state materially changed?

That discipline is how this project moved from a visually ambitious but fragile prototype into a substantially tested offline-first phone-and-Watch fitness product. Keep the trust model intact.
