# Calorie Lens mobile

The native iOS and Android client for Calorie Lens, built with Expo SDK 54 and
React Native.

## Included

- Today dashboard for calories, protein, water, steps, sleep, and meal rhythm
- Voice-first logging with typed input as a fallback
- Evidence-backed food estimates with visible ranges, confidence, and calculation basis
- A reviewed offline catalog sourced from USDA FoodData Central
- Breakfast, lunch, evening snack, and dinner logs
- Weight-personalized workout burn from the 2024 Adult Compendium of Physical Activities
- Follow-up voice questions when a portion, bowl size, duration, or weight is missing
- Workout, water, steps, sleep, and weight logging
- A full strength-training module in the Train tab (see below)
- SQLite-backed local persistence
- Native Apple Health/Apple Watch sync on iOS
- Native Health Connect sync on Android
- Secure account signup, login, password recovery, and deletion
- Device-encrypted offline storage plus encrypted cross-device cloud sync
- Portable backup export and restore
- A personal AI coach with editable long-term memory and offline fallback
- Local-first typed logging and coaching with explicit `AI:` opt-in
- One shared design system and codebase for both platforms
- Ten-step first-run onboarding for primary goal, body inputs, target direction,
  activity, training availability, diet, constraints, coaching style, portion
  calibration, Health, voice, and account setup
- Personal calorie and macro targets calculated from the Mifflin–St Jeor
  resting-energy estimate, daily activity, goal, and selected pace
- Goal-aware offline dashboard nudges and coaching; paid AI is optional

## Train tab (strength training)

The Train tab is a complete gym logger in the style of dedicated trackers
like Hevy:

- **Exercise library** — 85+ built-in exercises across barbell, dumbbell,
  machine, cable, bodyweight, kettlebell, and cardio. Every exercise has
  primary/secondary muscles, equipment, step-by-step instructions, form tips,
  and an animated skeleton demo of the movement.
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

## Local setup

```bash
npm install
cp .env.example .env
```

Start the AI API from the repository root:

```bash
pip install -r requirements.txt
GOOGLE_API_KEY=your_key uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Set `EXPO_PUBLIC_API_URL` in `.env` to the API address reachable by the device.
For a physical phone, use your computer's LAN IP rather than `localhost`.

Account sessions and recovery codes use iOS Keychain or Android Keystore via
Expo SecureStore. Fitness records are encrypted before being written to the
local SQLite key-value store. When signed in, changes are queued offline and
automatically synchronized when the API is reachable.

The AI service extracts facts from speech; it does not supply calories. Food
and exercise numbers are calculated after transcription by the deterministic
reference engine.

To keep costs near zero, typed updates use the on-device parser whenever it can
fully understand the command. Normal coach messages are answered on-device;
start one with `AI:` to opt into the online coach. Voice transcription is
online, cached on the server, and protected by a daily account limit.

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

HealthKit and Health Connect are native modules, so this app uses an Expo
development client instead of Expo Go.

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

- **iOS Simulator:** onboarding, signup/login, encrypted offline mode, typed
  food/workout/water logging, and the Apple Health permission flow. The
  simulator may contain no fitness samples and cannot receive Watch records.
- **Physical iPhone development build:** microphone recording and real Apple
  Health records. Apple Watch data reaches the app after the Watch syncs it to
  Health on that iPhone.
- **Android emulator:** onboarding, accounts, typed logging, and Health Connect
  after Health Connect and sample data are installed.
- **Physical Android development build:** microphone recording and approved
  Health Connect records.

The app now identifies these conditions in the UI instead of presenting a
generic failed connection. Voice additionally needs a signed-in account, a
reachable API, microphone permission, and `GOOGLE_API_KEY`. Typed logging and
the offline fitness tracker do not need that key and incur no AI cost.

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

App identifiers are configured as `com.sahilbharti.calorielens`. Store release
builds still require your Apple Developer and Google Play accounts, signing
credentials, privacy disclosures, and Health Connect declaration.

See [`../ACCURACY.md`](../ACCURACY.md) for calculation rules, source records,
benchmark cases, and limitations. Nutrition and exercise calories are
estimates, not medical advice.
