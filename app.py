import csv
import io
import json
import os
from datetime import date, datetime
from typing import Any, Dict, List

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound


load_dotenv()
st.set_page_config(page_title="HealthifyMe-style Fitness Tracker", page_icon="💪", layout="wide")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
TEXT_MODEL = st.secrets.get("GEMINI_TEXT_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash"))
MODEL_CANDIDATES = [
    TEXT_MODEL,
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Add it to Streamlit secrets or your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

MEAL_SLOTS = ["Breakfast", "Lunch", "Evening Snack", "Dinner"]

CALORIE_PROMPT = """You are a nutrition logging assistant.
User will provide one meal as free text with food names and quantities.
Estimate calories and macros for each item using common Indian and global food references.
Return STRICT JSON only:
{
  "items": [
    {
      "name": "string",
      "quantity_text": "string",
      "calories_kcal": number,
      "protein_g": number,
      "carbs_g": number,
      "fat_g": number
    }
  ],
  "meal_total": {
    "calories_kcal": number,
    "protein_g": number,
    "carbs_g": number,
    "fat_g": number
  },
  "confidence": 0-1,
  "notes": "short caveat on uncertainty"
}
Rules:
- Be practical and conservative when uncertain.
- Use quantity provided by user; infer only when needed.
- Never add non-JSON text.
"""




def _available_text_models() -> List[str]:
    try:
        names: List[str] = []
        for model in genai.list_models():
            methods = set(getattr(model, "supported_generation_methods", []) or [])
            if "generateContent" in methods:
                names.append(model.name.split("/")[-1])
        return sorted(set(names))
    except Exception:
        return []


def _clean_numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fallback_meal_estimate(user_text: str, reason: str) -> Dict[str, Any]:
    return {
        "items": [
            {
                "name": "manual_entry",
                "quantity_text": user_text,
                "calories_kcal": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
            }
        ],
        "meal_total": {"calories_kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0},
        "confidence": 0.0,
        "notes": f"AI calorie estimation unavailable: {reason}. Entry saved with 0 calories; edit manually later.",
    }

def _safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return {}
    return {}


def _today_state(day: str) -> Dict[str, Any]:
    if "fitness_log" not in st.session_state:
        st.session_state.fitness_log = {}
    if day not in st.session_state.fitness_log:
        st.session_state.fitness_log[day] = {
            "water_ml": 0,
            "sleep_hours": 0.0,
            "steps": 0,
            "weight_kg": None,
            "meals": {slot: [] for slot in MEAL_SLOTS},
            "exercises": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    return st.session_state.fitness_log[day]


def estimate_meal_from_text(user_text: str, model_name: str) -> Dict[str, Any]:
    last_error = "Unknown model error"
    for candidate in [model_name] + [m for m in MODEL_CANDIDATES if m != model_name]:
        try:
            model = genai.GenerativeModel(candidate)
            response = model.generate_content([CALORIE_PROMPT, f"Meal text: {user_text}"])
            payload = _safe_json_loads(response.text or "")
            payload.setdefault("items", [])
            payload.setdefault("meal_total", {"calories_kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0})
            payload.setdefault("confidence", 0.45)
            payload.setdefault("notes", f"Estimated using common references ({candidate}).")

            if not payload["meal_total"].get("calories_kcal") and payload["items"]:
                payload["meal_total"] = {
                    "calories_kcal": round(sum(_clean_numeric(i.get("calories_kcal", 0)) for i in payload["items"]), 1),
                    "protein_g": round(sum(_clean_numeric(i.get("protein_g", 0)) for i in payload["items"]), 1),
                    "carbs_g": round(sum(_clean_numeric(i.get("carbs_g", 0)) for i in payload["items"]), 1),
                    "fat_g": round(sum(_clean_numeric(i.get("fat_g", 0)) for i in payload["items"]), 1),
                }

            payload["model_used"] = candidate
            return payload
        except NotFound:
            last_error = f"model {candidate} not found for your API project"
            continue
        except Exception as exc:
            last_error = str(exc)
            continue

    return _fallback_meal_estimate(user_text, last_error)


def day_totals(day_log: Dict[str, Any]) -> Dict[str, float]:
    calories = protein = carbs = fat = exercise_burn = 0.0
    for slot in MEAL_SLOTS:
        for entry in day_log["meals"][slot]:
            meal_total = entry.get("meal_total", {})
            calories += float(meal_total.get("calories_kcal", 0))
            protein += float(meal_total.get("protein_g", 0))
            carbs += float(meal_total.get("carbs_g", 0))
            fat += float(meal_total.get("fat_g", 0))

    for ex in day_log["exercises"]:
        exercise_burn += float(ex.get("calories_burned", 0))

    return {
        "calories_in": round(calories, 1),
        "protein_g": round(protein, 1),
        "carbs_g": round(carbs, 1),
        "fat_g": round(fat, 1),
        "calories_out": round(exercise_burn, 1),
        "net_calories": round(calories - exercise_burn, 1),
    }


st.title("💪 HealthifyMe-style Personal Fitness Tracker")
st.caption(
    "Log food with plain text (e.g., '2 rotis + 1 cup dal'), track meals, water, exercise, steps, sleep, and calories in one place."
)

selected_date = st.date_input("Tracking date", value=date.today())
day_key = selected_date.isoformat()
log = _today_state(day_key)

with st.sidebar:
    st.header("Daily Goals")
    calorie_goal = st.number_input("Calorie goal (kcal)", min_value=800, max_value=6000, value=2200, step=50)
    water_goal = st.number_input("Water goal (ml)", min_value=500, max_value=8000, value=3000, step=100)
    step_goal = st.number_input("Step goal", min_value=1000, max_value=50000, value=8000, step=500)
    st.markdown("---")
    available_models = _available_text_models()
    default_model = TEXT_MODEL if TEXT_MODEL in available_models else (available_models[0] if available_models else TEXT_MODEL)
    selected_model = st.selectbox(
        "Gemini text model",
        options=available_models if available_models else MODEL_CANDIDATES,
        index=(available_models.index(default_model) if available_models and default_model in available_models else 0),
        help="If one model is unavailable in your project, the app auto-falls back to others.",
    )
    if not available_models:
        st.warning("Could not list models from API. Fallback model retry is enabled.")
    st.caption("All data is stored in the current app session.")

summary = day_totals(log)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Calories In", f"{summary['calories_in']:.0f} kcal")
m2.metric("Calories Out", f"{summary['calories_out']:.0f} kcal")
m3.metric("Net", f"{summary['net_calories']:.0f} kcal", delta=f"Goal {calorie_goal} kcal")
m4.metric("Water", f"{log['water_ml']} ml", delta=f"{log['water_ml']-water_goal} vs goal")
m5.metric("Steps", f"{log['steps']}", delta=f"{log['steps']-step_goal} vs goal")

st.progress(min(log["water_ml"] / max(water_goal, 1), 1.0), text="Water goal progress")

track_tab, meals_tab, exercise_tab, export_tab = st.tabs(["Daily Tracking", "Meal Logging", "Exercise", "Export / History"])

with track_tab:
    st.subheader("Body + Lifestyle")
    c1, c2, c3 = st.columns(3)
    with c1:
        log["water_ml"] = st.number_input("Water intake (ml)", min_value=0, max_value=10000, step=100, value=int(log["water_ml"]))
    with c2:
        log["sleep_hours"] = st.number_input("Sleep (hours)", min_value=0.0, max_value=24.0, step=0.5, value=float(log["sleep_hours"]))
    with c3:
        log["steps"] = st.number_input("Steps", min_value=0, max_value=100000, step=100, value=int(log["steps"]))

    weight_value = 60.0 if log["weight_kg"] is None else float(log["weight_kg"])
    log["weight_kg"] = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.1, value=weight_value)

with meals_tab:
    st.subheader("Log Meals with Natural Language")
    meal_slot = st.selectbox("Meal type", MEAL_SLOTS)
    meal_text = st.text_area(
        "What did you eat? Include quantity.",
        placeholder="Example: 2 egg omelette with 1 tsp butter, 2 slices whole wheat toast, 1 banana",
    )

    if st.button("Estimate and add meal", type="primary", use_container_width=True):
        if not meal_text.strip():
            st.warning("Please enter a meal description first.")
        else:
            with st.spinner("Estimating calories and macros..."):
                result = estimate_meal_from_text(meal_text.strip(), selected_model)
                log["meals"][meal_slot].append(
                    {
                        "logged_at": datetime.now().strftime("%H:%M"),
                        "input_text": meal_text.strip(),
                        **result,
                    }
                )
            if result.get("confidence", 0) == 0:
                st.warning("Added meal, but AI estimation failed. Saved as 0 kcal so you can still track continuity.")
            else:
                st.success(f"Added to {meal_slot} using {result.get('model_used', selected_model)}.")

    st.markdown("### Today's meal log")
    for slot in MEAL_SLOTS:
        entries = log["meals"][slot]
        with st.expander(f"{slot} ({len(entries)} entries)", expanded=(slot == "Breakfast")):
            if not entries:
                st.caption("No entries yet.")
                continue
            for idx, entry in enumerate(entries, start=1):
                mt = entry.get("meal_total", {})
                st.markdown(
                    f"**{idx}. {entry['logged_at']}** — {mt.get('calories_kcal', 0):.0f} kcal · "
                    f"P {mt.get('protein_g', 0):.1f}g · C {mt.get('carbs_g', 0):.1f}g · F {mt.get('fat_g', 0):.1f}g"
                )
                st.caption(f"Input: {entry.get('input_text', '')}")
                if entry.get("notes"):
                    st.caption(f"Note: {entry['notes']} (confidence {entry.get('confidence', 0):.2f})")
                if entry.get("items"):
                    st.table(entry["items"])

with exercise_tab:
    st.subheader("Track Exercise")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        exercise_name = st.text_input("Exercise name", placeholder="Brisk walking")
    with ex2:
        duration_min = st.number_input("Duration (minutes)", min_value=0, max_value=600, value=30, step=5)
    with ex3:
        calories_burned = st.number_input("Calories burned (kcal)", min_value=0, max_value=3000, value=150, step=10)

    if st.button("Add exercise"):
        if exercise_name.strip():
            log["exercises"].append(
                {
                    "name": exercise_name.strip(),
                    "duration_min": duration_min,
                    "calories_burned": calories_burned,
                    "logged_at": datetime.now().strftime("%H:%M"),
                }
            )
            st.success("Exercise added.")
        else:
            st.warning("Please provide an exercise name.")

    if log["exercises"]:
        st.table(log["exercises"])
    else:
        st.caption("No exercises added yet.")

with export_tab:
    st.subheader("Daily Summary")
    st.json({"date": day_key, "summary": summary, "log": log})

    payload = {"date": day_key, "summary": summary, "log": log}
    st.download_button(
        "Download day JSON",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name=f"fitness_log_{day_key}.json",
        mime="application/json",
    )

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["date", "meal", "logged_at", "input_text", "calories_kcal", "protein_g", "carbs_g", "fat_g"])
    for slot in MEAL_SLOTS:
        for entry in log["meals"][slot]:
            mt = entry.get("meal_total", {})
            writer.writerow(
                [
                    day_key,
                    slot,
                    entry.get("logged_at", ""),
                    entry.get("input_text", ""),
                    mt.get("calories_kcal", 0),
                    mt.get("protein_g", 0),
                    mt.get("carbs_g", 0),
                    mt.get("fat_g", 0),
                ]
            )

    st.download_button(
        "Download meals CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"meals_{day_key}.csv",
        mime="text/csv",
    )

st.info("Estimates can be inaccurate. Use as a personal tracking aid, not medical advice.")
