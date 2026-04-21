import csv
import io
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError


load_dotenv()
st.set_page_config(page_title="Calorie Lens", page_icon="🥗", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_FILE = DATA_DIR / "fitness_logs.json"
MEAL_SLOTS = ["Breakfast", "Lunch", "Evening Snack", "Dinner"]
DEFAULT_PROFILE = {
    "name": "My Fitness Tracker",
    "calorie_goal": 2200,
    "protein_goal": 120,
    "water_goal": 3000,
    "step_goal": 8000,
}

def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


GOOGLE_API_KEY = read_secret("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
TEXT_MODEL = read_secret("GEMINI_TEXT_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash"))
AI_ENABLED = bool(GOOGLE_API_KEY)

if AI_ENABLED:
    genai.configure(api_key=GOOGLE_API_KEY)

CALORIE_PROMPT = """You are a nutrition logging assistant for a personal fitness tracker.
The user gives one meal in natural language, often Indian food, with quantities.
Estimate calories and macros using practical real-world references.
Return STRICT JSON only with this exact shape:
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
  "confidence": number,
  "notes": "short string"
}
Rules:
- Never include markdown or extra commentary.
- Be conservative when uncertain.
- Respect the user's quantity text.
- If a quantity is unclear, make one practical assumption and mention it in notes.
"""

FOOD_DB = [
    {"name": "Roti", "aliases": ["roti", "roti", "chapati", "phulka"], "calories": 120, "protein": 3.5, "carbs": 18, "fat": 3},
    {"name": "Cooked Rice", "aliases": ["rice", "cooked rice", "plain rice"], "calories": 130, "protein": 2.5, "carbs": 28, "fat": 0.3},
    {"name": "Dal", "aliases": ["dal", "dhal", "lentils", "sambar"], "calories": 180, "protein": 9, "carbs": 24, "fat": 4},
    {"name": "Paneer", "aliases": ["paneer", "paneer bhurji", "paneer curry"], "calories": 265, "protein": 18, "carbs": 6, "fat": 20},
    {"name": "Chicken Curry", "aliases": ["chicken curry", "chicken gravy", "butter chicken"], "calories": 260, "protein": 24, "carbs": 8, "fat": 14},
    {"name": "Chicken Breast", "aliases": ["chicken breast", "grilled chicken"], "calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    {"name": "Egg", "aliases": ["egg", "boiled egg"], "calories": 78, "protein": 6.3, "carbs": 0.6, "fat": 5.3},
    {"name": "Omelette", "aliases": ["omelette", "omelet"], "calories": 154, "protein": 11, "carbs": 2, "fat": 11},
    {"name": "Idli", "aliases": ["idli", "idlis"], "calories": 58, "protein": 2, "carbs": 12, "fat": 0.4},
    {"name": "Dosa", "aliases": ["dosa", "masala dosa"], "calories": 170, "protein": 4, "carbs": 25, "fat": 5},
    {"name": "Poha", "aliases": ["poha"], "calories": 210, "protein": 5, "carbs": 34, "fat": 6},
    {"name": "Upma", "aliases": ["upma"], "calories": 220, "protein": 5, "carbs": 32, "fat": 8},
    {"name": "Bread Slice", "aliases": ["bread", "toast", "slice bread"], "calories": 80, "protein": 3, "carbs": 15, "fat": 1},
    {"name": "Peanut Butter", "aliases": ["peanut butter"], "calories": 95, "protein": 4, "carbs": 3, "fat": 8},
    {"name": "Banana", "aliases": ["banana"], "calories": 105, "protein": 1.3, "carbs": 27, "fat": 0.4},
    {"name": "Apple", "aliases": ["apple"], "calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3},
    {"name": "Curd", "aliases": ["curd", "yogurt", "dahi"], "calories": 98, "protein": 5, "carbs": 7, "fat": 5},
    {"name": "Milk", "aliases": ["milk"], "calories": 120, "protein": 6, "carbs": 12, "fat": 5},
    {"name": "Tea", "aliases": ["tea", "chai"], "calories": 70, "protein": 2, "carbs": 9, "fat": 3},
    {"name": "Coffee", "aliases": ["coffee"], "calories": 60, "protein": 2, "carbs": 7, "fat": 2},
    {"name": "Biscuit", "aliases": ["biscuit", "cookie"], "calories": 35, "protein": 0.5, "carbs": 5, "fat": 1.5},
    {"name": "Samosa", "aliases": ["samosa"], "calories": 250, "protein": 4, "carbs": 30, "fat": 12},
    {"name": "Salad", "aliases": ["salad"], "calories": 50, "protein": 2, "carbs": 10, "fat": 0.5},
    {"name": "Protein Shake", "aliases": ["protein shake", "whey shake"], "calories": 130, "protein": 24, "carbs": 4, "fat": 2},
]

NUMBER_WORDS = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "half": 0.5,
    "quarter": 0.25,
}


def default_day_log() -> Dict[str, Any]:
    return {
        "water_ml": 0,
        "sleep_hours": 0.0,
        "steps": 0,
        "weight_kg": None,
        "notes": "",
        "meals": {slot: [] for slot in MEAL_SLOTS},
        "exercises": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_store() -> Dict[str, Any]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"profile": DEFAULT_PROFILE.copy(), "days": {}}


def save_store(store: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def get_store() -> Dict[str, Any]:
    if "fitness_store" not in st.session_state:
        st.session_state.fitness_store = load_store()
    store = st.session_state.fitness_store
    store.setdefault("profile", DEFAULT_PROFILE.copy())
    store.setdefault("days", {})
    for key, value in DEFAULT_PROFILE.items():
        store["profile"].setdefault(key, value)
    return store


def ensure_day(store: Dict[str, Any], day_key: str) -> Dict[str, Any]:
    if day_key not in store["days"]:
        store["days"][day_key] = default_day_log()
    day_log = store["days"][day_key]
    day_log.setdefault("notes", "")
    day_log.setdefault("water_ml", 0)
    day_log.setdefault("sleep_hours", 0.0)
    day_log.setdefault("steps", 0)
    day_log.setdefault("weight_kg", None)
    day_log.setdefault("meals", {slot: [] for slot in MEAL_SLOTS})
    day_log.setdefault("exercises", [])
    for slot in MEAL_SLOTS:
        day_log["meals"].setdefault(slot, [])
    return day_log


def touch_log(day_log: Dict[str, Any]) -> None:
    day_log["updated_at"] = datetime.now().isoformat(timespec="seconds")


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


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("items", [])
    payload.setdefault("meal_total", {})
    payload.setdefault("confidence", 0.45)
    payload.setdefault("notes", "Estimated using common food references.")
    payload["meal_total"].setdefault("calories_kcal", 0.0)
    payload["meal_total"].setdefault("protein_g", 0.0)
    payload["meal_total"].setdefault("carbs_g", 0.0)
    payload["meal_total"].setdefault("fat_g", 0.0)

    if payload["items"] and not payload["meal_total"]["calories_kcal"]:
        payload["meal_total"] = {
            "calories_kcal": round(sum(float(item.get("calories_kcal", 0)) for item in payload["items"]), 1),
            "protein_g": round(sum(float(item.get("protein_g", 0)) for item in payload["items"]), 1),
            "carbs_g": round(sum(float(item.get("carbs_g", 0)) for item in payload["items"]), 1),
            "fat_g": round(sum(float(item.get("fat_g", 0)) for item in payload["items"]), 1),
        }
    return payload


def candidate_models() -> List[str]:
    raw = [
        TEXT_MODEL,
        "gemini-2.0-flash",
        "models/gemini-2.0-flash",
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
    ]
    unique: List[str] = []
    for model_name in raw:
        if model_name and model_name not in unique:
            unique.append(model_name)
    return unique


def try_ai_estimate(user_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if not AI_ENABLED:
        return None, "No Google API key found. Using local calorie estimates."

    last_error = "Gemini estimate unavailable."
    for model_name in candidate_models():
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([CALORIE_PROMPT, f"Meal text: {user_text}"])
            payload = normalize_payload(_safe_json_loads(getattr(response, "text", "") or ""))
            if payload["items"] or payload["meal_total"]["calories_kcal"]:
                payload["source"] = "Gemini"
                payload["model_used"] = model_name
                return payload, ""
            last_error = f"{model_name} returned an empty response."
        except Exception as exc:
            last_error = f"{model_name}: {exc}"
    return None, last_error


def split_meal_text(user_text: str) -> List[str]:
    parts = re.split(r"\s*(?:,|\+| and )\s*", user_text.strip(), flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def parse_quantity(text: str) -> float:
    lowered = text.lower()
    fraction_match = re.search(r"(\d+)\s*/\s*(\d+)", lowered)
    if fraction_match:
        denominator = float(fraction_match.group(2))
        if denominator:
            return round(float(fraction_match.group(1)) / denominator, 2)

    number_match = re.search(r"(\d+(?:\.\d+)?)", lowered)
    if number_match:
        return float(number_match.group(1))

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return 1.0


def find_food_match(item_text: str) -> Optional[Dict[str, Any]]:
    lowered = item_text.lower()
    best_match: Optional[Dict[str, Any]] = None
    best_alias_length = -1
    for food in FOOD_DB:
        for alias in food["aliases"]:
            if alias in lowered and len(alias) > best_alias_length:
                best_match = food
                best_alias_length = len(alias)
    return best_match


def estimate_meal_locally(user_text: str, reason: str) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    unmatched: List[str] = []

    for part in split_meal_text(user_text):
        quantity = parse_quantity(part)
        food = find_food_match(part)

        if food:
            items.append(
                {
                    "name": food["name"],
                    "quantity_text": part,
                    "calories_kcal": round(food["calories"] * quantity, 1),
                    "protein_g": round(food["protein"] * quantity, 1),
                    "carbs_g": round(food["carbs"] * quantity, 1),
                    "fat_g": round(food["fat"] * quantity, 1),
                }
            )
        else:
            unmatched.append(part)
            items.append(
                {
                    "name": part.title(),
                    "quantity_text": part,
                    "calories_kcal": round(120 * quantity, 1),
                    "protein_g": round(4 * quantity, 1),
                    "carbs_g": round(14 * quantity, 1),
                    "fat_g": round(4 * quantity, 1),
                }
            )

    payload = normalize_payload({"items": items})
    if unmatched:
        payload["notes"] = (
            "Local estimate used. Some foods were approximated: "
            + ", ".join(unmatched[:3])
            + "."
        )
    else:
        payload["notes"] = "Local estimate used from common food references."
    payload["confidence"] = 0.6 if not unmatched else 0.42
    payload["source"] = "Local fallback"
    payload["model_used"] = "offline-estimator"
    payload["estimation_error"] = reason
    return payload


def estimate_meal_from_text(user_text: str) -> Dict[str, Any]:
    ai_payload, ai_error = try_ai_estimate(user_text)
    if ai_payload:
        if ai_error:
            ai_payload["notes"] = f"{ai_payload.get('notes', '')} Fallback note: {ai_error}".strip()
        return ai_payload
    return estimate_meal_locally(user_text, ai_error)


def day_totals(day_log: Dict[str, Any]) -> Dict[str, float]:
    calories = protein = carbs = fat = exercise_burn = 0.0
    for slot in MEAL_SLOTS:
        for entry in day_log["meals"][slot]:
            meal_total = entry.get("meal_total", {})
            calories += float(meal_total.get("calories_kcal", 0))
            protein += float(meal_total.get("protein_g", 0))
            carbs += float(meal_total.get("carbs_g", 0))
            fat += float(meal_total.get("fat_g", 0))

    for exercise in day_log["exercises"]:
        exercise_burn += float(exercise.get("calories_burned", 0))

    return {
        "calories_in": round(calories, 1),
        "protein_g": round(protein, 1),
        "carbs_g": round(carbs, 1),
        "fat_g": round(fat, 1),
        "calories_out": round(exercise_burn, 1),
        "net_calories": round(calories - exercise_burn, 1),
    }


def meal_slot_totals(day_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for slot in MEAL_SLOTS:
        slot_total = {"calories_kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        for entry in day_log["meals"][slot]:
            meal_total = entry.get("meal_total", {})
            slot_total["calories_kcal"] += float(meal_total.get("calories_kcal", 0))
            slot_total["protein_g"] += float(meal_total.get("protein_g", 0))
            slot_total["carbs_g"] += float(meal_total.get("carbs_g", 0))
            slot_total["fat_g"] += float(meal_total.get("fat_g", 0))
        rows.append(
            {
                "Meal": slot,
                "Entries": len(day_log["meals"][slot]),
                "Calories": round(slot_total["calories_kcal"], 1),
                "Protein (g)": round(slot_total["protein_g"], 1),
                "Carbs (g)": round(slot_total["carbs_g"], 1),
                "Fat (g)": round(slot_total["fat_g"], 1),
            }
        )
    return rows


def history_rows(store: Dict[str, Any], days: int = 7) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    today = date.today()
    for offset in range(days):
        day_key = (today - timedelta(days=offset)).isoformat()
        day_log = store["days"].get(day_key, default_day_log())
        totals = day_totals(day_log)
        rows.append(
            {
                "Date": day_key,
                "Calories In": totals["calories_in"],
                "Calories Out": totals["calories_out"],
                "Net Calories": totals["net_calories"],
                "Protein (g)": totals["protein_g"],
                "Water (ml)": day_log["water_ml"],
                "Steps": day_log["steps"],
                "Sleep (h)": day_log["sleep_hours"],
                "Weight (kg)": day_log["weight_kg"],
            }
        )
    return rows


def make_export_payload(day_key: str, day_log: Dict[str, Any]) -> Dict[str, Any]:
    return {"date": day_key, "summary": day_totals(day_log), "log": day_log}


def latest_known_weight(store: Dict[str, Any], day_key: str) -> float:
    current_weight = store["days"].get(day_key, {}).get("weight_kg")
    if current_weight is not None:
        return float(current_weight)

    dated_weights: List[Tuple[str, float]] = []
    for existing_day, entry in store["days"].items():
        weight = entry.get("weight_kg")
        if weight is not None and existing_day <= day_key:
            dated_weights.append((existing_day, float(weight)))
    if dated_weights:
        dated_weights.sort(key=lambda item: item[0], reverse=True)
        return dated_weights[0][1]
    return 60.0


store = get_store()
profile = store["profile"]

st.title("Calorie Lens")
st.caption(
    "A personal HealthifyMe-style tracker for meals, water, exercise, sleep, steps, weight, and daily calorie control."
)

with st.sidebar:
    st.header("Profile & Goals")
    with st.form("profile_form"):
        profile_name = st.text_input("Profile name", value=profile["name"])
        calorie_goal_input = st.number_input(
            "Daily calorie goal (kcal)", min_value=800, max_value=6000, value=int(profile["calorie_goal"]), step=50
        )
        protein_goal_input = st.number_input(
            "Daily protein goal (g)", min_value=20, max_value=300, value=int(profile["protein_goal"]), step=5
        )
        water_goal_input = st.number_input(
            "Water goal (ml)", min_value=500, max_value=8000, value=int(profile["water_goal"]), step=100
        )
        step_goal_input = st.number_input(
            "Step goal", min_value=1000, max_value=50000, value=int(profile["step_goal"]), step=500
        )
        save_profile = st.form_submit_button("Save goals", use_container_width=True)
        if save_profile:
            profile["name"] = profile_name
            profile["calorie_goal"] = int(calorie_goal_input)
            profile["protein_goal"] = int(protein_goal_input)
            profile["water_goal"] = int(water_goal_input)
            profile["step_goal"] = int(step_goal_input)
            save_store(store)
            st.success("Goals saved.")
    st.markdown("---")
    if AI_ENABLED:
        st.success("Gemini calorie estimation is enabled.")
        st.caption(f"Primary model preference: `{TEXT_MODEL}`")
    else:
        st.warning("No Google API key found. The app will use built-in calorie estimates.")
    st.caption(f"Logs are saved locally in `{DATA_FILE.relative_to(APP_DIR)}`")

selected_date = st.date_input("Tracking date", value=date.today())
day_key = selected_date.isoformat()
log = ensure_day(store, day_key)

summary = day_totals(log)
meal_completion = sum(1 for slot in MEAL_SLOTS if log["meals"][slot])

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Calories In", f"{summary['calories_in']:.0f} kcal", delta=f"{summary['calories_in'] - profile['calorie_goal']:.0f} vs goal")
m2.metric("Protein", f"{summary['protein_g']:.0f} g", delta=f"{summary['protein_g'] - profile['protein_goal']:.0f} vs goal")
m3.metric("Calories Out", f"{summary['calories_out']:.0f} kcal")
m4.metric("Water", f"{log['water_ml']} ml", delta=f"{log['water_ml'] - profile['water_goal']:.0f} vs goal")
m5.metric("Steps", f"{log['steps']}", delta=f"{log['steps'] - profile['step_goal']:.0f} vs goal")
m6.metric("Meals Logged", f"{meal_completion}/4", delta=f"{4 - meal_completion} left")

st.progress(min(summary["calories_in"] / max(profile["calorie_goal"], 1), 1.0), text="Calorie goal progress")
st.progress(min(log["water_ml"] / max(profile["water_goal"], 1), 1.0), text="Water goal progress")

dashboard_tab, meals_tab, exercise_tab, history_tab = st.tabs(["Dashboard", "Meals", "Exercise", "History & Export"])

with dashboard_tab:
    st.subheader("Daily Fitness Check-In")
    with st.form("daily_checkin_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            water_input = st.number_input(
                "Water intake (ml)", min_value=0, max_value=10000, step=100, value=int(log["water_ml"])
            )
        with c2:
            sleep_input = st.number_input(
                "Sleep (hours)", min_value=0.0, max_value=24.0, step=0.5, value=float(log["sleep_hours"])
            )
        with c3:
            steps_input = st.number_input("Steps", min_value=0, max_value=100000, step=100, value=int(log["steps"]))
        with c4:
            weight_input = st.number_input(
                "Weight (kg)",
                min_value=20.0,
                max_value=300.0,
                step=0.1,
                value=latest_known_weight(store, day_key),
            )

        notes_input = st.text_area(
            "Daily notes",
            value=log.get("notes", ""),
            placeholder="Energy levels, cravings, mood, digestion, training quality, anything useful for your own tracking.",
            height=100,
        )
        save_daily_checkin = st.form_submit_button("Save daily check-in", use_container_width=True)
        if save_daily_checkin:
            log["water_ml"] = int(water_input)
            log["sleep_hours"] = float(sleep_input)
            log["steps"] = int(steps_input)
            log["weight_kg"] = float(weight_input)
            log["notes"] = notes_input
            touch_log(log)
            save_store(store)
            st.success("Daily check-in saved.")

    qw1, qw2, qw3, qw4 = st.columns(4)
    if qw1.button("+250 ml water", use_container_width=True):
        log["water_ml"] += 250
        touch_log(log)
        save_store(store)
        st.rerun()
    if qw2.button("+500 ml water", use_container_width=True):
        log["water_ml"] += 500
        touch_log(log)
        save_store(store)
        st.rerun()
    if qw3.button("+1000 steps", use_container_width=True):
        log["steps"] += 1000
        touch_log(log)
        save_store(store)
        st.rerun()
    if qw4.button("Reset selected day", use_container_width=True):
        store["days"][day_key] = default_day_log()
        save_store(store)
        st.rerun()

    st.markdown("### Meal breakdown")
    st.table(meal_slot_totals(log))

with meals_tab:
    st.subheader("Log a Meal in Plain English")
    st.caption("Example: `2 rotis, 1 cup dal, salad` or `1 banana and 1 scoop whey`")

    meal_slot = st.selectbox("Meal type", MEAL_SLOTS)
    meal_text = st.text_area(
        "What did you eat? Include quantity.",
        placeholder="Example: 2 egg omelette with 2 slices toast and 1 banana",
        height=110,
    )

    if st.button("Estimate and add meal", type="primary", use_container_width=True):
        if not meal_text.strip():
            st.warning("Please enter your meal first.")
        else:
            with st.spinner("Estimating calories and macros..."):
                result = estimate_meal_from_text(meal_text.strip())
            log["meals"][meal_slot].append(
                {
                    "logged_at": datetime.now().strftime("%H:%M"),
                    "input_text": meal_text.strip(),
                    **result,
                }
            )
            touch_log(log)
            save_store(store)
            st.success(f"Added to {meal_slot}.")
            st.rerun()

    st.markdown("### Today's meals")
    for slot in MEAL_SLOTS:
        entries = log["meals"][slot]
        with st.expander(f"{slot} ({len(entries)} entries)", expanded=(slot == "Breakfast")):
            if not entries:
                st.caption("No entries yet.")
                continue
            for idx, entry in enumerate(entries):
                meal_total = entry.get("meal_total", {})
                col_a, col_b = st.columns([7, 1])
                with col_a:
                    st.markdown(
                        f"**{idx + 1}. {entry.get('logged_at', '--')}**  "
                        f"{meal_total.get('calories_kcal', 0):.0f} kcal | "
                        f"P {meal_total.get('protein_g', 0):.1f}g | "
                        f"C {meal_total.get('carbs_g', 0):.1f}g | "
                        f"F {meal_total.get('fat_g', 0):.1f}g"
                    )
                    st.caption(entry.get("input_text", ""))
                    st.caption(
                        f"Estimated via {entry.get('source', 'Unknown')} | "
                        f"confidence {float(entry.get('confidence', 0)):.2f}"
                    )
                    if entry.get("notes"):
                        st.caption(entry["notes"])
                    if entry.get("items"):
                        st.table(entry["items"])
                with col_b:
                    if st.button("Delete", key=f"delete-meal-{slot}-{idx}", use_container_width=True):
                        log["meals"][slot].pop(idx)
                        touch_log(log)
                        save_store(store)
                        st.rerun()

with exercise_tab:
    st.subheader("Exercise & Activity")
    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        exercise_name = st.text_input("Exercise name", placeholder="Brisk walking")
    with ex2:
        duration_min = st.number_input("Duration (minutes)", min_value=0, max_value=600, value=30, step=5)
    with ex3:
        calories_burned = st.number_input("Calories burned (kcal)", min_value=0, max_value=3000, value=150, step=10)
    with ex4:
        intensity = st.selectbox("Intensity", ["Low", "Moderate", "High"])

    exercise_note = st.text_input("Exercise note", placeholder="Felt easy, outdoor walk, upper body session, etc.")

    if st.button("Add exercise", use_container_width=True):
        if exercise_name.strip():
            log["exercises"].append(
                {
                    "name": exercise_name.strip(),
                    "duration_min": duration_min,
                    "calories_burned": calories_burned,
                    "intensity": intensity,
                    "notes": exercise_note.strip(),
                    "logged_at": datetime.now().strftime("%H:%M"),
                }
            )
            touch_log(log)
            save_store(store)
            st.success("Exercise added.")
            st.rerun()
        else:
            st.warning("Please provide an exercise name.")

    st.markdown("### Today's exercise log")
    if log["exercises"]:
        for idx, entry in enumerate(log["exercises"]):
            col_a, col_b = st.columns([7, 1])
            with col_a:
                st.markdown(
                    f"**{entry.get('name', '')}** | {entry.get('duration_min', 0)} min | "
                    f"{entry.get('calories_burned', 0)} kcal | {entry.get('intensity', 'Moderate')}"
                )
                if entry.get("notes"):
                    st.caption(entry["notes"])
                st.caption(f"Logged at {entry.get('logged_at', '--')}")
            with col_b:
                if st.button("Delete", key=f"delete-ex-{idx}", use_container_width=True):
                    log["exercises"].pop(idx)
                    touch_log(log)
                    save_store(store)
                    st.rerun()
    else:
        st.caption("No exercise added yet.")

with history_tab:
    st.subheader("Recent history")
    st.table(history_rows(store, days=7))

    st.markdown("### Export current day")
    payload = make_export_payload(day_key, log)
    st.json(payload)

    st.download_button(
        "Download day JSON",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name=f"fitness_log_{day_key}.json",
        mime="application/json",
    )

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(
        ["date", "meal_slot", "logged_at", "input_text", "source", "calories_kcal", "protein_g", "carbs_g", "fat_g"]
    )
    for slot in MEAL_SLOTS:
        for entry in log["meals"][slot]:
            meal_total = entry.get("meal_total", {})
            writer.writerow(
                [
                    day_key,
                    slot,
                    entry.get("logged_at", ""),
                    entry.get("input_text", ""),
                    entry.get("source", ""),
                    meal_total.get("calories_kcal", 0),
                    meal_total.get("protein_g", 0),
                    meal_total.get("carbs_g", 0),
                    meal_total.get("fat_g", 0),
                ]
            )

    st.download_button(
        "Download meals CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name=f"meals_{day_key}.csv",
        mime="text/csv",
    )

st.info(
    "Nutrition values are estimates. If Gemini is unavailable or the configured model is missing, the app now falls back to a built-in estimator instead of crashing."
)
