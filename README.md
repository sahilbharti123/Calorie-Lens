# 💪 Calorie Lens → Personal HealthifyMe-style Tracker

This app is now a **personal fitness tracker** built with **Streamlit + Gemini**.

It lets you:
- Log meals in plain English (or Hinglish) like: `2 rotis + 1 cup dal + 1 tsp ghee`
- Auto-estimate calories + macros from that text
- Automatic Gemini model fallback if one configured model is unavailable
- Track meal slots separately: **Breakfast, Lunch, Evening Snack, Dinner**
- Track **water intake, exercise, steps, sleep, and weight**
- Export your daily log as JSON/CSV

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment

Set your API key in `.env` or Streamlit secrets:

```bash
GOOGLE_API_KEY=your_google_ai_key
# optional override
GEMINI_TEXT_MODEL=gemini-2.0-flash
```

## Notes

- Nutrition values are estimates from an AI model.
- Use this for self-tracking convenience, not medical diagnosis.
