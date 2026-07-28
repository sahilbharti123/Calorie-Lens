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
- Estimates meal calories and macros with Gemini 3.6 Flash when available
- Falls back to a built-in estimator if the Gemini model is missing or the API is unavailable
- Tracks water, steps, sleep, weight, exercise, and daily notes
- Imports Apple Health XML/ZIP snapshots, including Apple Watch steps and workouts
- Saves your data locally in `data/fitness_logs.json`
- Exports your current day as JSON or CSV

## Voice logging

The quick logger accepts text without an API key. Voice transcription and
multi-part command understanding use Gemini audio input and therefore require
`GOOGLE_API_KEY`.

Examples:

- `I ate 3 eggs and toast for breakfast`
- `Drank two glasses of water and slept 7.5 hours`
- `Did 45 minutes of strength training, around 280 calories`
- `Lunch was rajma chawal, walked 4,000 steps`

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

## Environment

Create a `.env` file or Streamlit secret:

```bash
GOOGLE_API_KEY=your_google_ai_key
GEMINI_TEXT_MODEL=gemini-3.6-flash
```

`GOOGLE_API_KEY` is optional now. Without it, the app still works using the built-in estimator.

## Notes

- Nutrition values are estimates, not medical advice.
- The built-in fallback is best for common foods and rough self-tracking.
