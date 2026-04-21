# Calorie Lens

A personal HealthifyMe-style fitness tracker built with Streamlit. Try it here: https://calorie-lens-count.streamlit.app/

## What it does

- Logs meals by meal slot: Breakfast, Lunch, Evening Snack, Dinner
- Lets you type food in plain English or Hinglish like `2 rotis + 1 cup dal`
- Estimates calories and macros with Gemini when available
- Falls back to a built-in estimator if the Gemini model is missing or the API is unavailable
- Tracks water, steps, sleep, weight, exercise, and daily notes
- Saves your data locally in `data/fitness_logs.json`
- Exports your current day as JSON or CSV

## Why the previous error happened

The `google.api_core.exceptions.NotFound` error usually means the configured Gemini model name is unavailable for the API key or the model identifier is outdated.

This app now handles that more safely:

- it tries multiple Gemini model names
- it catches model/API failures instead of crashing the app
- it falls back to local calorie estimates so you can still log meals

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment

Create a `.env` file or Streamlit secret:

```bash
GOOGLE_API_KEY=your_google_ai_key
GEMINI_TEXT_MODEL=gemini-2.0-flash
```

`GOOGLE_API_KEY` is optional now. Without it, the app still works using the built-in estimator.

## Notes

- Nutrition values are estimates, not medical advice.
- The built-in fallback is best for common foods and rough self-tracking.
