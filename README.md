# Calorie Lens

A light personal fitness tracker with native iOS and Android apps plus a
Streamlit web companion. Try the web version here:
https://calorie-lens-count.streamlit.app/

The native app lives in [`mobile/`](mobile/) and uses one React Native/Expo
codebase for both platforms.

## What it does

- Logs breakfast, lunch, evening snacks, and dinner
- Accepts natural English, Hindi, or Hinglish updates such as
  `Lunch was 2 rotis and dal, plus 500 ml water`
- Records the same mixed update by voice
- Uses the on-device parser first and only calls Gemini for voice or an unrecognized typed command
- Calculates known foods from reviewed USDA FoodData Central records
- Calculates active workout energy from body weight, duration, intensity, and 2024 Compendium MET values
- Shows a range, confidence, source, and calculation basis before saving
- Asks a short follow-up instead of guessing when a portion or workout detail is missing
- Tracks water, steps, sleep, weight, exercise, and daily notes
- Imports Apple Health XML/ZIP snapshots, including Apple Watch steps and workouts
- Provides secure signup, login, recovery codes, password rotation, and account deletion
- Encrypts the local mobile vault with a device-only key and the server vault with AES-GCM
- Syncs fitness history across devices with offline-first conflict recovery
- Saves coach preferences, limitations, notes, and recent conversations as long-term memory
- Exports and restores portable JSON backups
- Saves web companion data locally in `data/fitness_logs.json`
- Exports your current day as JSON or CSV

## Voice logging

The quick logger accepts text without an API key. Voice transcription and
multi-part command understanding use Gemini audio input and therefore require
`GOOGLE_API_KEY`.

Everyday coach prompts also stay on-device. Prefix a coach message with `AI:`
to deliberately request a deeper online answer.

Examples:

- `I ate 3 boiled eggs and 2 slices toast for breakfast`
- `Drank two glasses of water and slept 7.5 hours`
- `Did 45 minutes of hard strength training`
- `Lunch was one 200 ml bowl rajma and 150 grams cooked rice`

## Apple Health and Apple Watch

Open Health on iPhone, tap your profile, then **Export All Health Data**. Upload
the generated `export.zip` in the Apple Health tab. The importer supports steps,
water, sleep, body weight, and workouts and prevents the same export from being
imported twice.

Direct background HealthKit sync requires a native iOS companion because
HealthKit is not exposed to web apps. Snapshot import is the working web-app
integration boundary.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the native app and its AI/voice service:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
cd mobile
cp .env.example .env
npm install
npm run ios
# or: npm run android
```

See [`mobile/README.md`](mobile/README.md) for HealthKit, Apple Watch, Health
Connect, development-build, and EAS setup.

## Run the account API in a container

The included container exposes the account, sync, backup, AI parsing, voice,
and coach endpoints. Mount `/data` so accounts survive restarts and keep the
master key in your deployment secret manager.

```bash
docker build -t calorie-lens-api .
docker run --rm -p 8000:8000 \
  -v calorie-lens-data:/data \
  -e CALORIE_LENS_MASTER_KEY=your_urlsafe_base64_32_byte_key \
  -e GOOGLE_API_KEY=your_google_ai_key \
  calorie-lens-api
```

## Environment

Create a `.env` file or Streamlit secret:

```bash
GOOGLE_API_KEY=your_google_ai_key
GEMINI_TEXT_MODEL=gemini-3.1-flash-lite
CALORIE_LENS_AI_DAILY_LIMIT=8
CALORIE_LENS_AI_TEXT_DAILY_LIMIT=4
CALORIE_LENS_AI_AUDIO_DAILY_LIMIT=4
CALORIE_LENS_AI_COACH_DAILY_LIMIT=2
CALORIE_LENS_ENABLE_WEB_AI=false
CALORIE_LENS_DB_PATH=data/calorie_lens.db
# Set this to a stable URL-safe base64 32-byte key in production:
CALORIE_LENS_MASTER_KEY=
```

`GOOGLE_API_KEY` is optional for typed logging. Voice transcription requires
the API. The offline path uses the same conservative reference rules for its
reviewed food and activity catalog.

AI results are encrypted and cached, repeated identical requests are free of
additional model calls, prompts and outputs are capped, and authenticated
per-account daily limits prevent an open-ended bill. The limits reset each UTC
day and can be lowered with the environment variables above.

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
- See [`ACCURACY.md`](ACCURACY.md) for sources, formulas, benchmark cases, and
  known limits.
