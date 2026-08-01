# Calorie Lens mobile

The native iOS and Android client for Calorie Lens, built with Expo SDK 54 and
React Native.

## Included

- Today dashboard for calories, protein, water, steps, sleep, and meal rhythm
- Voice-first logging with typed input as a fallback, transcribed on the device
  by the operating system's own speech recogniser
- Evidence-backed food estimates with visible ranges, confidence, and calculation basis
- A reviewed offline catalog sourced from USDA FoodData Central
- Breakfast, lunch, evening snack, and dinner logs
- Weight-personalized workout burn from the 2024 Adult Compendium of Physical Activities
- A short follow-up question when a portion, bowl size, duration, or weight is missing
- Workout, water, steps, sleep, and weight logging
- A full strength-training module in the Train tab (see below)
- SQLite-backed local persistence
- Native Apple Health/Apple Watch sync on iOS
- Native Health Connect sync on Android
- Optional account signup, login, password recovery, and deletion
- Device-encrypted offline storage, with encrypted cross-device sync when an
  account is used
- Portable backup export and restore
- A Coach tab that shows the day's focus, your plan, and insights calculated on
  this device from your own entries
- Deterministic on-device parsing for every entry; remote AI parsing is opt-in
  and off by default
- One shared design system and codebase for both platforms
- Ten-step first-run onboarding for primary goal, body inputs, target direction,
  activity, training availability, diet, constraints, coaching style, portion
  calibration, Health, voice, and account setup
- Personal calorie and macro targets calculated from the Mifflin–St Jeor
  resting-energy estimate, daily activity, goal, and selected pace
- Goal-aware dashboard nudges and coach insights calculated on the device; the
  backend is optional and the app runs with no server at all

## Train tab (strength training)

The Train tab is a complete gym logger in the style of dedicated trackers
like Hevy:

- **Exercise library** — 100 built-in exercises across barbell, dumbbell,
  machine, cable, bodyweight, kettlebell, and cardio. Every exercise has
  primary/secondary muscles and equipment, real start/finish demonstration
  photos (public-domain, from free-exercise-db), a full researched how-to
  (setup, execution with range-of-motion standards, breathing, tempo, common
  mistakes with fixes, safety), a one-tap technique video from an established
  channel (every link verified against YouTube metadata), and an animated
  movement-path figure. See ../ACCURACY.md for the content provenance and
  review process.
- **Routines** — build reusable workouts with target sets, exact reps or rep
  ranges (`8-12`), starting weights, per-exercise rest timers, notes,
  supersets, and optional folders. Template routines (full-body, push/pull/
  legs, home bodyweight) can be imported and edited. A finished workout can be
  saved back as a routine.
- **Live logging** — start a routine or an empty workout. Each set row shows
  your previous performance, prefills the last weight × reps, and is checked
  off with one tap. Set types (warm-up · failure · drop), optional RPE
  tracking (enable in the tab settings), an automatic rest countdown with
  −15s/+15s/skip, and mid-workout add/replace/remove of exercises are all
  supported.
- **Records & charts** — live PR detection on every completed set (heaviest
  weight, best estimated 1RM by Epley, best set volume; most reps or longest
  hold for bodyweight and timed movements), plus per-exercise history and
  trend charts and weekly sets-per-muscle-group counts.
- **Energy integration** — a finished session logs its duration and a
  transparent active-energy range into the day (set-weighted Compendium METs
  × body weight × duration, resting energy excluded), so training and
  nutrition stay in one ledger.

Workouts, routines, and settings live inside the same encrypted vault as the
rest of your data and sync across devices with the existing offline-first
merge.

## Design system

The app uses one dark design system, "Midnight Athlete", defined entirely in
[`src/theme.ts`](src/theme.ts). No screen declares its own colour, radius, or
type size; everything reads from those tokens.

- **Surfaces** — a near-black canvas (`#07090A`) with a stack of raised, inset,
  and pressed card surfaces, plus hairline dividers.
- **Accent** — a single electric lime (`#C6FF3C`) for progress, primary
  actions, and live state. Macro and chart hues (lime, blue, orange) are
  distinct at a glance and meet AA contrast on the dark canvas.
- **Tokens** — `space`, `radius`, `shadow`, `motion`, and `gradient` are
  exported alongside the palette. On dark surfaces a glow reads better than a
  drop shadow, so `shadow.glow` is used where a card needs lift.
- **Components** — the shared kit lives in
  [`src/components/ui.tsx`](src/components/ui.tsx): `Screen`, `ScreenHeader`,
  `Card`, `Well`, `Ring`, `Bar`, `Metric`, `ListRow`, `Segmented`, `Chip`,
  `Pill`, `PrimaryButton`, `GhostButton`, `CountUp`, `Reveal`, `VoiceBar`, and
  others. Icons are drawn by `src/components/glyph.tsx`.

Typography is two families, loaded with `expo-font` in
[`app/_layout.tsx`](app/_layout.tsx) and held behind the splash screen until
they are ready:

- **Space Grotesk** (`@expo-google-fonts/space-grotesk`) for display numerals,
  screen titles, and card headlines.
- **Inter** (`@expo-google-fonts/inter`) for everything you actually read —
  body copy, rows, values, captions, and labels.

The `text` export defines the whole type scale from `hero` down to `micro`, and
`tabular` renders digits at a fixed width so counting numbers do not shift the
layout as they animate.

## Local setup

```bash
npm install
npm run ios
# or: npm run android
```

That is the whole setup. The app has no required backend: logging, estimates,
voice, the Train tab, Health sync, and the Coach tab all run on the device.

A `.env` is only needed if you want the optional server (see
[`../DEPLOY.md`](../DEPLOY.md)):

```bash
cp .env.example .env
```

Two variables are read, both inlined at build time:

- `EXPO_PUBLIC_API_URL` (in `.env.example`) — the address of the optional sync
  API, reachable by the device. For a physical phone use your computer's LAN IP
  rather than `localhost`. In development the app also discovers the API from
  the Metro bundler address, so this is usually unnecessary until you make a
  production build.
- `EXPO_PUBLIC_ENABLE_AI_PARSING` (not in `.env.example`; add it yourself) —
  set to `1` to allow the remote language parser to be consulted for foods the
  local catalog does not recognize. Off by default.

To run the optional API from the repository root:

```bash
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Account sessions and recovery codes use iOS Keychain or Android Keystore via
Expo SecureStore. Fitness records are encrypted before being written to the
local SQLite key-value store. When signed in, changes are queued offline and
automatically synchronized when the API is reachable.

## Voice and parsing

Speech is transcribed by the operating system — Apple's Speech framework on
iOS, the platform recogniser on Android — through `expo-speech-recognition`
(see [`src/lib/speech.ts`](src/lib/speech.ts)). No audio is uploaded anywhere,
there is no per-use cost, and voice logging needs no account, no API key, and
no `GOOGLE_API_KEY`. It works offline wherever the OS has an on-device model
installed. The app hands the recogniser a vocabulary of Indian dish names and
gym movements, which are exactly the words a general dictation model gets
wrong.

The transcript then goes through the same deterministic parser typed input uses
([`src/lib/nutrition.ts`](src/lib/nutrition.ts)). That parser extracts food,
amount, unit, activity, duration, and intensity, and the reference engine turns
those facts into numbers. No model produces a calorie.

Remote AI parsing is opt-in and off by default: `AI_PARSING_ENABLED` reads
`EXPO_PUBLIC_ENABLE_AI_PARSING === '1'`. With the flag off — the shipped
default — an unrecognized food produces a clarifying question asking for label
calories or the main parts with amounts, rather than a model call. With the
flag on, and only when a signed-in session and an API URL both exist, the app
may consult `/v1/parse-command` for the part it could not resolve; the local
result is still the fallback.

The Coach tab calls no service either. Its insights are computed arithmetically
from your own logs by [`src/lib/insights.ts`](src/lib/insights.ts) — protein
and calorie adherence, training volume against plan, hydration, how wide your
estimates are running, weight trend, and logging streak. Each one states the
number it is based on. The conversational AI coach is not part of this release.

## Accuracy setup

Onboarding asks for the inputs that actually change the plan:

- age, height, current weight, and energy-equation option;
- primary goal, target direction, pace, and normal daily activity;
- realistic training days, available time, experience, and preferred movement;
- diet style, meal rhythm, allergies, injuries, main challenge, and coaching tone;
- the capacity of your usual bowl in ml.

The app calculates a starting maintenance estimate, goal adjustment, calories,
protein, carbohydrates, fat, water, steps, weekly training minutes, and strength
frequency. Weight updates from logging or an approved health source recalculate
the plan. These values remain estimates; review them against a 2–4 week trend.

To measure a bowl, fill it with water and read the volume in a measuring jug.
The app intentionally asks for this value instead of treating every Indian
katori or bowl as identical. Its default cup measure is 200 ml.

For best results, say the cooked weight or package serving:

- `Lunch: 2 rotis and one 200 ml bowl rajma`
- `100 grams grilled chicken and 150 grams cooked rice`
- `30 minute brisk walk, moderate effort`

## Run native builds

HealthKit, Health Connect, and speech recognition are native modules, so this
app uses an Expo development client instead of Expo Go.

```bash
npm run prebuild
npm run ios
# or
npm run android
```

The first Health sync asks for operating-system permissions. Apple Watch data
already written to Apple Health is read through HealthKit. Android reads
compatible health and wearable data through Health Connect.

### What to test where

- **iOS Simulator:** onboarding, offline mode, typed food/workout/water
  logging, the Train tab, the Coach tab, and the Apple Health permission flow.
  **Speech recognition does not work reliably in the Simulator** — it has no
  real microphone of its own, and the on-device recogniser is frequently
  unavailable or returns nothing there. Treat a voice failure in the Simulator
  as inconclusive and retest it on hardware. The Simulator may also contain no
  fitness samples and cannot receive Watch records.
- **Physical iPhone development build:** the only honest test of dictation, and
  of real Apple Health records. Apple Watch data reaches the app after the
  Watch syncs it to Health on that iPhone.
- **Android emulator:** onboarding, typed logging, and Health Connect after
  Health Connect and sample data are installed. Dictation may work if the
  emulator image ships Google's speech services and a recognition language is
  downloaded, but it is not dependable — confirm on a device.
- **Physical Android development build:** dictation and approved Health Connect
  records.
- **Anywhere, only with a server running:** signup, login, and cross-device
  sync. Without one, the auth screen offers "Continue without an account" and
  everything else still works.

The app reports these conditions in the UI instead of presenting a generic
failed connection. Voice needs microphone and speech permission and nothing
else — no account, no API key, and no Calorie Lens server. Where the OS has no
on-device recognition model it may fall back to its own network recogniser, and
the app says so rather than failing silently. Typed logging and the offline
fitness tracker incur no cost of any kind.

## Quality checks

```bash
npm run typecheck
npm run lint
cd ..
python -m unittest discover -s tests -v
```

## Build with EAS

```bash
npx eas build --profile development --platform ios
npx eas build --profile development --platform android
```

App identifiers are configured as `com.advaice.calorielens`. Store release
builds still require your Apple Developer and Google Play accounts, signing
credentials, privacy disclosures, and Health Connect declaration.

See [`../ACCURACY.md`](../ACCURACY.md) for calculation rules, source records,
benchmark cases, and limitations, and [`../DEPLOY.md`](../DEPLOY.md) for what
running this actually costs. Nutrition and exercise calories are estimates, not
medical advice.
