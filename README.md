# Vigorly

A light personal fitness tracker with native iOS and Android apps plus a
Streamlit web companion. Try the web version here:
https://calorie-lens-count.streamlit.app/

The native app lives in [`mobile/`](mobile/) and uses one React Native/Expo
codebase for both platforms. It needs no backend: the Python API in this
repository is optional, and is only there for cross-device sync and the legacy
AI endpoints. See [`DEPLOY.md`](DEPLOY.md).

## What it does

- Logs breakfast, lunch, evening snacks, and dinner
- Accepts natural English, Hindi, or Hinglish updates such as
  `Lunch was 2 rotis and dal, plus 500 ml water`
- Records the same mixed update by voice, transcribed on the phone by the
  operating system's own speech recogniser
- Parses every update with a deterministic on-device parser; remote AI parsing
  is opt-in and off by default
- Calculates known foods from reviewed USDA FoodData Central records
- Calculates active workout energy from body weight, duration, intensity, and 2024 Compendium MET values
- Full strength training in the Train tab: routines with folders, supersets,
  rep ranges and per-exercise rest timers; a live workout logger with the
  previous performance beside every set, set types (warm-up/failure/drop),
  optional RPE, automatic rest countdown and live PR detection; workout
  history, per-exercise records and charts; and an 85+ exercise library with
  step-by-step instructions and animated form demos
- A native Apple Watch workout logger: start a synced routine or quick workout,
  enter weight and reps with the Digital Crown, run duration sets and rest
  timers, add exercises or sets, see notes and previous performance, and finish
  the session without opening the iPhone. Heart rate and active calories come
  from the live HealthKit workout and Watch-originated sessions replay to the
  phone after an offline workout.
- Shows a range, confidence, source, and calculation basis before saving
- Asks a short follow-up instead of guessing when a portion or workout detail is missing
- Tracks water, steps, sleep, weight, exercise, and daily notes
- Imports Apple Health XML/ZIP snapshots, including Apple Watch steps and workouts
- Provides optional signup, login, recovery codes, password rotation, and account deletion
- Encrypts the local mobile vault with a device-only key and the server vault with AES-GCM
- Syncs fitness history across devices with offline-first conflict recovery when
  an account is used
- Reads your plan and your logs back to you in the Coach tab, with insights
  calculated arithmetically on the device
- Exports and restores portable JSON backups
- Saves web companion data locally in `data/fitness_logs.json`
- Exports your current day as JSON or CSV

## Voice logging

In the native app, speech is transcribed by the operating system — Apple's
Speech framework on iOS, the platform recogniser on Android — via
`expo-speech-recognition`. No audio leaves the phone, no account is required,
no `GOOGLE_API_KEY` is involved, and there is no per-use cost.

The transcript goes through the same deterministic parser typed input uses.
That parser extracts facts — food, amount, unit, activity, duration, intensity —
and the reference engine turns them into numbers. No model produces a calorie.
When the local catalog does not recognize a food, the app asks for label
calories or the main parts with amounts instead of guessing.

Remote AI parsing exists but is opt-in and off by default; set
`EXPO_PUBLIC_ENABLE_AI_PARSING=1` in `mobile/.env` to allow it. See
[`DEPLOY.md`](DEPLOY.md) for what that costs and what it buys.

The Streamlit web companion (`app.py`) is separate: it still uses Gemini audio
input for its own voice box, and only when `GOOGLE_API_KEY` is set and
`CALORIE_LENS_ENABLE_WEB_AI=true`. It ships with that disabled.

Examples:

- `I ate 3 boiled eggs and 2 slices toast for breakfast`
- `Drank two glasses of water and slept 7.5 hours`
- `Did 45 minutes of hard strength training`
- `Lunch was one 200 ml bowl rajma and 150 grams cooked rice`

## Apple Health and Apple Watch

The native iOS build includes a watchOS companion. Open Vigorly on the iPhone
once to cache routines and the exercise catalog on the Watch; after that a
workout can be started, logged, and finished from the Watch while the phone is
away or offline. The Watch runs the HealthKit workout session for live heart
rate and active-energy readings, persists the active log locally, and sends
complete revisioned snapshots back to the phone so individual set messages are
not lost.

The web companion separately supports Apple Health XML/ZIP import. Open Health
on iPhone, tap your profile, then **Export All Health Data** and upload the
generated `export.zip`. The importer supports steps, water, sleep, body weight,
and workouts and prevents duplicate imports.

## Run the native app

```bash
cd mobile
npm install
npm run ios
# or: npm run android
```

That is all of it. **Running the Python API is optional.** The native app is
fully functional with no server: logging, estimates, voice, the Train tab,
Health sync, the Coach tab, and the encrypted local vault all work on the
device alone. The auth screen offers "Continue without an account".

See [`mobile/README.md`](mobile/README.md) for the design system, HealthKit,
Apple Watch, Health Connect, development-build, and EAS setup, and
[`DEPLOY.md`](DEPLOY.md) for what running any of this costs.

## Run the optional API

A server is needed for exactly two things: cross-device sync of the encrypted
vault, and the legacy AI endpoints if you deliberately turn remote parsing on.

```bash
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

In development the app discovers the API automatically from the Metro
bundler address, so a physical phone reaches your computer without any
`.env` as long as both are on the same Wi-Fi and uvicorn is bound to
`0.0.0.0`. Set `EXPO_PUBLIC_API_URL` (see `mobile/.env.example`) only to
override this or for production builds. The auth screen shows a live
reachability indicator with the exact URL it is trying.

`uvicorn api:app` starts without the Google AI SDK: accounts, sync, and typed
logging never needed it, and a partially installed `google-*` package no longer
crashes startup.

Mount the database directory on persistent storage in any deployment, so
accounts survive a restart, and keep the master key in your deployment secret
manager. [`DEPLOY.md`](DEPLOY.md) has concrete steps for a couple of cheap
hosts.

## Run the web companion

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment

Every variable below is server-side and optional; the native app reads none of
them. Create a `.env` file or Streamlit secret:

```bash
# Persistence. Point the database at a mounted volume in production.
CALORIE_LENS_DB_PATH=data/calorie_lens.db
CALORIE_LENS_KEY_PATH=data/calorie_lens.master.key
# Set this to a stable URL-safe base64 32-byte key in production:
CALORIE_LENS_MASTER_KEY=
CALORIE_LENS_ALLOWED_ORIGINS=*

# Only needed for the legacy AI endpoints and the web companion's voice box.
GOOGLE_API_KEY=your_google_ai_key
GEMINI_TEXT_MODEL=gemini-3.1-flash-lite
CALORIE_LENS_AI_DAILY_LIMIT=8
CALORIE_LENS_AI_TEXT_DAILY_LIMIT=4
CALORIE_LENS_AI_AUDIO_DAILY_LIMIT=4
CALORIE_LENS_AI_COACH_DAILY_LIMIT=2
CALORIE_LENS_ENABLE_WEB_AI=false
```

`GOOGLE_API_KEY` is not needed for anything the native app does by default —
not for voice, not for typed logging, not for the Coach tab. The API starts and
serves accounts, sync, and backup without it.

When the AI endpoints are used, results are encrypted and cached, repeated
identical requests cost no additional model calls, prompts and outputs are
capped, and per-account daily limits prevent an open-ended bill. The limits
reset each UTC day and can be lowered with the environment variables above.

The public Streamlit companion keeps AI disabled even when a key exists. Set
`CALORIE_LENS_ENABLE_WEB_AI=true` only if you deliberately want model-backed
fallbacks there; known foods and typed fitness commands are local-first.

The API creates a development encryption key in `data/` when no master key is
configured. In production, mount the database on persistent storage and set a
stable `CALORIE_LENS_MASTER_KEY`; losing that key makes encrypted vaults
unrecoverable.

## Notes

- Nutrition values are estimates, not medical advice.
- Exact intake cannot be inferred from a phrase such as “one bowl rajma”
  without bowl capacity and recipe information. The app makes those
  assumptions visible.
- The conversational AI coach is not part of this release. The `/v1/coach`
  endpoint still exists server-side, but the app no longer calls it; the Coach
  tab computes its insights on the device.
- See [`ACCURACY.md`](ACCURACY.md) for sources, formulas, benchmark cases, and
  known limits, and [`DEPLOY.md`](DEPLOY.md) for hosting tiers and real costs.
