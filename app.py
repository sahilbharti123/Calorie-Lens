import csv
import hashlib
import io
import json
import os
import re
import secrets
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError


load_dotenv()
st.set_page_config(page_title="Calorie Lens", page_icon="🥗", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_FILE = DATA_DIR / "fitness_logs.json"
DEVICE_COOKIE_NAME = "calorie_lens_device"
DEVICE_STORAGE_KEY = "calorie_lens_device_token"
MEAL_SLOTS = ["Breakfast", "Lunch", "Evening Snack", "Dinner"]
DEFAULT_PROFILE = {
    "display_name": "",
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
    {"name": "Roti", "aliases": ["roti", "chapati", "phulka"], "calories": 120, "protein": 3.5, "carbs": 18, "fat": 3},
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


def normalize_username(username: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", username.strip().lower())


def hash_password(password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return derived.hex(), salt.hex()


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    candidate_hash, _ = hash_password(password, salt_hex)
    return secrets.compare_digest(candidate_hash, password_hash)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def default_day_log() -> Dict[str, Any]:
    return {
        "water_ml": 0,
        "sleep_hours": 0.0,
        "steps": 0,
        "weight_kg": None,
        "notes": "",
        "mood": "Steady",
        "energy": "Balanced",
        "meals": {slot: [] for slot in MEAL_SLOTS},
        "exercises": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def default_user(username: str) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE.copy()
    profile["display_name"] = username
    return {
        "username": username,
        "password_hash": "",
        "salt": "",
        "remember_tokens": [],
        "profile": profile,
        "days": {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_store() -> Dict[str, Any]:
    if DATA_FILE.exists():
        try:
            raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}
    else:
        raw = {}

    if "users" in raw:
        raw.setdefault("legacy_data", None)
        return raw

    legacy_data = None
    if raw.get("profile") or raw.get("days"):
        legacy_data = {"profile": raw.get("profile", DEFAULT_PROFILE.copy()), "days": raw.get("days", {})}

    return {"users": {}, "legacy_data": legacy_data}


def save_store(store: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def get_store() -> Dict[str, Any]:
    if "fitness_store" not in st.session_state:
        st.session_state.fitness_store = load_store()
    store = st.session_state.fitness_store
    store.setdefault("users", {})
    store.setdefault("legacy_data", None)
    return store


def ensure_user(store: Dict[str, Any], username: str) -> Dict[str, Any]:
    if username not in store["users"]:
        store["users"][username] = default_user(username)
    user = store["users"][username]
    user.setdefault("remember_tokens", [])
    user.setdefault("profile", DEFAULT_PROFILE.copy())
    user.setdefault("days", {})
    user["profile"].setdefault("display_name", username)
    for key, value in DEFAULT_PROFILE.items():
        user["profile"].setdefault(key, value)
    return user


def ensure_day(user: Dict[str, Any], day_key: str) -> Dict[str, Any]:
    if day_key not in user["days"]:
        user["days"][day_key] = default_day_log()
    day_log = user["days"][day_key]
    day_log.setdefault("notes", "")
    day_log.setdefault("mood", "Steady")
    day_log.setdefault("energy", "Balanced")
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


def set_cookie(name: str, value: str, days: int = 365) -> None:
    components.html(
        f"""
        <script>
        const tokenValue = {json.dumps(value)};
        const maxAge = {days} * 24 * 60 * 60;
        const secureFlag = window.location.protocol === "https:" ? "; Secure" : "";
        document.cookie = {json.dumps(name)} + "=" + tokenValue + "; max-age=" + maxAge + "; path=/; SameSite=Lax" + secureFlag;
        window.localStorage.setItem({json.dumps(DEVICE_STORAGE_KEY)}, tokenValue);
        </script>
        """,
        height=0,
    )


def clear_cookie(name: str) -> None:
    components.html(
        f"""
        <script>
        const secureFlag = window.location.protocol === "https:" ? "; Secure" : "";
        document.cookie = {json.dumps(name)} + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax" + secureFlag;
        window.localStorage.removeItem({json.dumps(DEVICE_STORAGE_KEY)});
        </script>
        """,
        height=0,
    )


def bootstrap_remembered_login() -> None:
    components.html(
        f"""
        <script>
        const cookieName = {json.dumps(DEVICE_COOKIE_NAME)};
        const storageKey = {json.dumps(DEVICE_STORAGE_KEY)};
        const hasCookie = document.cookie.split("; ").some((item) => item.startsWith(cookieName + "="));
        const storedToken = window.localStorage.getItem(storageKey);
        if (!hasCookie && storedToken) {{
          const secureFlag = window.location.protocol === "https:" ? "; Secure" : "";
          document.cookie = cookieName + "=" + storedToken + "; max-age=" + (365 * 24 * 60 * 60) + "; path=/; SameSite=Lax" + secureFlag;
          window.setTimeout(() => window.location.reload(), 50);
        }}
        </script>
        """,
        height=0,
    )


def authenticate_from_cookie(store: Dict[str, Any]) -> None:
    if st.session_state.get("auth_username") in store["users"]:
        return

    device_token = st.context.cookies.get(DEVICE_COOKIE_NAME)
    if not device_token:
        return

    device_token_hash = hash_token(device_token)
    for username, user in store["users"].items():
        for token_record in user.get("remember_tokens", []):
            if secrets.compare_digest(token_record.get("token_hash", ""), device_token_hash):
                st.session_state.auth_username = username
                st.session_state.auth_token_hash = device_token_hash
                token_record["last_used_at"] = datetime.now().isoformat(timespec="seconds")
                save_store(store)
                return


def login_user(store: Dict[str, Any], username: str, remember_device: bool) -> None:
    st.session_state.auth_username = username
    user = ensure_user(store, username)
    if remember_device:
        token = secrets.token_urlsafe(32)
        token_hash = hash_token(token)
        user["remember_tokens"] = [
            record for record in user.get("remember_tokens", []) if record.get("token_hash") != token_hash
        ]
        user["remember_tokens"].append(
            {
                "token_hash": token_hash,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "last_used_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        st.session_state.auth_token_hash = token_hash
        st.session_state.pending_device_cookie = token
    else:
        st.session_state.pop("auth_token_hash", None)
        st.session_state.clear_device_cookie = True
    save_store(store)


def logout_user(store: Dict[str, Any]) -> None:
    username = st.session_state.get("auth_username")
    token_hash = st.session_state.get("auth_token_hash")
    if username and token_hash and username in store["users"]:
        user = store["users"][username]
        user["remember_tokens"] = [
            record for record in user.get("remember_tokens", []) if record.get("token_hash") != token_hash
        ]
        save_store(store)
    st.session_state.clear_device_cookie = True
    st.session_state.pop("auth_username", None)
    st.session_state.pop("auth_token_hash", None)


def render_auth_screen(store: Dict[str, Any]) -> None:
    st.title("Calorie Lens")
    st.caption("Create separate accounts for different people, and optionally stay logged in on this device.")

    user_count = len(store["users"])
    if user_count == 0:
        st.info("Create the first account to start tracking meals, water, exercise, sleep, and weight.")

    login_tab, create_tab = st.tabs(["Login", "Create account"])

    with login_tab:
        with st.form("login_form"):
            login_username = normalize_username(st.text_input("Username"))
            login_password = st.text_input("Password", type="password")
            login_remember = st.checkbox("Stay logged in on this device", value=True)
            login_submit = st.form_submit_button("Login", use_container_width=True)
            if login_submit:
                user = store["users"].get(login_username)
                if not user:
                    st.error("That account does not exist.")
                elif not verify_password(login_password, user.get("password_hash", ""), user.get("salt", "")):
                    st.error("Incorrect password.")
                else:
                    login_user(store, login_username, login_remember)
                    st.rerun()

    with create_tab:
        with st.form("create_account_form"):
            signup_username_raw = st.text_input("Username", help="Use lowercase letters, numbers, or underscores.")
            signup_password = st.text_input("Password", type="password")
            signup_password_confirm = st.text_input("Confirm password", type="password")
            signup_remember = st.checkbox("Stay logged in on this device", value=True)
            import_legacy = False
            if store.get("legacy_data") and user_count == 0:
                import_legacy = st.checkbox("Import existing tracker data into this first account", value=True)
            signup_submit = st.form_submit_button("Create account", use_container_width=True)

            if signup_submit:
                signup_username = normalize_username(signup_username_raw)
                if not signup_username:
                    st.error("Please choose a valid username.")
                elif signup_username in store["users"]:
                    st.error("That username already exists.")
                elif len(signup_password) < 6:
                    st.error("Use a password with at least 6 characters.")
                elif signup_password != signup_password_confirm:
                    st.error("Passwords do not match.")
                else:
                    password_hash, salt_hex = hash_password(signup_password)
                    user = default_user(signup_username)
                    user["password_hash"] = password_hash
                    user["salt"] = salt_hex
                    if import_legacy and store.get("legacy_data"):
                        legacy_data = store["legacy_data"]
                        user["profile"] = legacy_data.get("profile", user["profile"])
                        user["profile"].setdefault("display_name", signup_username)
                        user["days"] = legacy_data.get("days", {})
                        store["legacy_data"] = None
                    store["users"][signup_username] = user
                    login_user(store, signup_username, signup_remember)
                    st.rerun()


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
        payload["notes"] = "Local estimate used. Some foods were approximated: " + ", ".join(unmatched[:3]) + "."
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


def history_rows(user: Dict[str, Any], days: int = 7) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    today = date.today()
    for offset in range(days):
        day_key = (today - timedelta(days=offset)).isoformat()
        day_log = user["days"].get(day_key, default_day_log())
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


def latest_known_weight(user: Dict[str, Any], day_key: str) -> float:
    current_weight = user["days"].get(day_key, {}).get("weight_kg")
    if current_weight is not None:
        return float(current_weight)

    dated_weights: List[Tuple[str, float]] = []
    for existing_day, entry in user["days"].items():
        weight = entry.get("weight_kg")
        if weight is not None and existing_day <= day_key:
            dated_weights.append((existing_day, float(weight)))
    if dated_weights:
        dated_weights.sort(key=lambda item: item[0], reverse=True)
        return dated_weights[0][1]
    return 60.0


def ratio(current: float, goal: float) -> float:
    if goal <= 0:
        return 0.0
    return max(0.0, min(current / goal, 1.0))


def whole_ratio(current: float, goal: float) -> int:
    return int(round(ratio(current, goal) * 100))


def completion_streak(user: Dict[str, Any]) -> int:
    streak = 0
    cursor = date.today()
    while True:
        day_key = cursor.isoformat()
        day_log = user["days"].get(day_key)
        if not day_log:
            break
        totals = day_totals(day_log)
        completed = 0
        if totals["calories_in"] >= 0.6 * float(user["profile"]["calorie_goal"]):
            completed += 1
        if day_log.get("water_ml", 0) >= 0.8 * float(user["profile"]["water_goal"]):
            completed += 1
        if day_log.get("steps", 0) >= 0.8 * float(user["profile"]["step_goal"]):
            completed += 1
        if day_log.get("sleep_hours", 0) >= 6:
            completed += 1
        if completed < 2:
            break
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def readiness_score(day_log: Dict[str, Any], profile: Dict[str, Any], totals: Dict[str, float]) -> int:
    score = 0
    score += whole_ratio(day_log.get("water_ml", 0), float(profile["water_goal"])) * 0.25
    score += whole_ratio(day_log.get("steps", 0), float(profile["step_goal"])) * 0.2
    score += whole_ratio(totals["protein_g"], float(profile["protein_goal"])) * 0.25
    sleep_component = min(max((float(day_log.get("sleep_hours", 0)) / 8.0) * 100, 0), 100)
    score += sleep_component * 0.2
    meal_component = min(sum(1 for slot in MEAL_SLOTS if day_log["meals"][slot]) * 25, 100)
    score += meal_component * 0.1
    return int(round(min(score, 100)))


def consistency_score(day_log: Dict[str, Any], profile: Dict[str, Any], totals: Dict[str, float]) -> int:
    calorie_balance = 100 - min(abs(totals["calories_in"] - float(profile["calorie_goal"])) / max(float(profile["calorie_goal"]), 1) * 100, 100)
    water_balance = whole_ratio(day_log.get("water_ml", 0), float(profile["water_goal"]))
    protein_balance = whole_ratio(totals["protein_g"], float(profile["protein_goal"]))
    steps_balance = whole_ratio(day_log.get("steps", 0), float(profile["step_goal"]))
    return int(round((calorie_balance * 0.35) + (water_balance * 0.2) + (protein_balance * 0.25) + (steps_balance * 0.2)))


def focus_message(day_log: Dict[str, Any], profile: Dict[str, Any], totals: Dict[str, float]) -> Tuple[str, str]:
    gaps = [
        ("Hydration needs attention", day_log.get("water_ml", 0) < 0.6 * float(profile["water_goal"]), "Push another 500-750 ml over the next few hours."),
        ("Protein is lagging", totals["protein_g"] < 0.7 * float(profile["protein_goal"]), "Aim for a protein-forward meal or shake in your next slot."),
        ("Recovery is low", float(day_log.get("sleep_hours", 0)) < 6.5, "Keep training light and prioritize an earlier bedtime tonight."),
        ("Movement can be stronger", day_log.get("steps", 0) < 0.6 * float(profile["step_goal"]), "Add a 10-15 minute walk after your next meal."),
    ]
    for title, condition, note in gaps:
        if condition:
            return title, note
    return "You are in a strong groove", "Stay steady, keep portions consistent, and finish the day with a calm dinner."


def coach_insight(day_log: Dict[str, Any], profile: Dict[str, Any], totals: Dict[str, float]) -> str:
    readiness = readiness_score(day_log, profile, totals)
    meal_count = sum(1 for slot in MEAL_SLOTS if day_log["meals"][slot])
    if readiness >= 80:
        return "You are showing excellent consistency today. Keep dinner lighter than lunch and protect your sleep to turn this into a strong full-day win."
    if totals["protein_g"] < 0.6 * float(profile["protein_goal"]):
        return "Your calorie intake may be moving, but your protein is still behind. Anchor the next meal around eggs, paneer, chicken, curd, or whey."
    if day_log.get("water_ml", 0) < 0.5 * float(profile["water_goal"]):
        return "Hydration is the easiest lever to improve today. Sip steadily instead of catching up all at once, and your hunger control should improve too."
    if meal_count <= 1 and datetime.now().hour >= 14:
        return "Your meal rhythm is sparse right now. Even a simple balanced meal will help energy, focus, and evening craving control."
    if float(day_log.get("sleep_hours", 0)) < 6:
        return "Low sleep changes hunger and recovery more than people think. Treat today like a maintenance day and avoid chasing intensity."
    return "This looks balanced. Keep momentum by finishing your final meal on time and avoiding random snack drift late in the evening."


def nutrient_split(totals: Dict[str, float]) -> List[Dict[str, Any]]:
    return [
        {"label": "Protein", "value": totals["protein_g"], "unit": "g", "tone": "var(--accent-forest)"},
        {"label": "Carbs", "value": totals["carbs_g"], "unit": "g", "tone": "var(--accent-gold)"},
        {"label": "Fat", "value": totals["fat_g"], "unit": "g", "tone": "var(--accent-coral)"},
    ]


def render_premium_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

        :root {
          --bg-main: #f4f7f6;
          --bg-card: rgba(255, 255, 255, 0.96);
          --bg-strong: #ffffff;
          --ink: #1f2937;
          --muted: #5f6c7b;
          --stroke: rgba(31, 41, 55, 0.10);
          --accent-forest: #2f6f5e;
          --accent-gold: #b88a44;
          --accent-coral: #bf6f5d;
          --accent-mint: #e7f0ec;
          --shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
          --radius-xl: 28px;
          --radius-lg: 20px;
        }

        .stApp {
          background:
            radial-gradient(circle at top left, rgba(47, 111, 94, 0.06), transparent 24%),
            radial-gradient(circle at top right, rgba(184, 138, 68, 0.06), transparent 22%),
            linear-gradient(180deg, #f5f7f8 0%, #f7faf9 58%, #eff4f2 100%);
          color: var(--ink);
          font-family: "DM Sans", sans-serif;
        }

        h1, h2, h3, .premium-title {
          font-family: "Fraunces", serif !important;
          letter-spacing: -0.03em;
          color: var(--ink);
        }

        p, label, .stCaption, .stMarkdown, .stText {
          color: var(--ink) !important;
        }

        [data-testid="stHeader"] {
          background: transparent;
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #eef4f1 0%, #f5f8f7 100%);
          border-right: 1px solid var(--stroke);
        }

        [data-testid="stSidebar"] * {
          color: var(--ink) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
        [data-testid="stSidebar"] .stCaption {
          color: var(--ink) !important;
          opacity: 1 !important;
        }

        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stDateInput label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
          color: var(--ink) !important;
        }

        [data-testid="stMetric"] {
          background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,252,251,0.98));
          border: 1px solid rgba(59, 72, 63, 0.07);
          border-radius: 22px;
          padding: 18px 18px 16px 18px;
          box-shadow: var(--shadow);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
          color: var(--ink) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
          gap: 10px;
          background: rgba(255,255,255,0.76);
          padding: 8px;
          border-radius: 999px;
          border: 1px solid var(--stroke);
        }

        .stTabs [data-baseweb="tab"] {
          border-radius: 999px;
          padding: 10px 18px;
          font-weight: 600;
          color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
          background: linear-gradient(135deg, #e9efe9, #f3f6f1) !important;
          color: var(--ink) !important;
          border: 1px solid rgba(103, 134, 114, 0.20);
        }

        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
          border-radius: 999px;
          border: 1px solid rgba(103, 134, 114, 0.16);
          background: linear-gradient(135deg, var(--accent-forest), #3e8571);
          color: #ffffff !important;
          font-weight: 600;
          padding: 0.65rem 1.1rem;
          box-shadow: 0 10px 24px rgba(57, 67, 61, 0.08);
        }

        .stButton > button[kind="secondary"] {
          background: rgba(255,255,255,0.82);
          color: var(--ink) !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
          border-radius: 18px !important;
          background: rgba(255,255,255,0.98) !important;
          color: var(--ink) !important;
          border: 1px solid rgba(59, 72, 63, 0.09) !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stDateInput input,
        .stDateInput input::placeholder,
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder,
        .stNumberInput input::placeholder,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input {
          -webkit-text-fill-color: var(--ink) !important;
        }

        .stDateInput [data-baseweb="input"],
        .stDateInput [data-baseweb="base-input"],
        .stTextInput [data-baseweb="input"],
        .stTextArea [data-baseweb="base-input"],
        .stNumberInput [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
          background: #ffffff !important;
          color: var(--ink) !important;
          border-radius: 16px !important;
        }

        .stDateInput [data-baseweb="input"] input,
        .stDateInput [data-baseweb="base-input"] input,
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="base-input"] input {
          color: var(--ink) !important;
          -webkit-text-fill-color: var(--ink) !important;
          opacity: 1 !important;
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder,
        .stNumberInput input::placeholder,
        .stDateInput input::placeholder {
          color: var(--muted) !important;
          opacity: 1 !important;
        }

        .stTextArea label,
        .stTextInput label,
        .stNumberInput label,
        .stDateInput label,
        .stSelectbox label,
        .stMultiSelect label {
          color: var(--ink) !important;
          font-weight: 600 !important;
        }

        .stDateInput button,
        .stSelectbox svg,
        .stNumberInput button svg,
        .stDateInput svg,
        [data-testid="stSidebar"] svg {
          color: var(--ink) !important;
          fill: var(--ink) !important;
        }

        textarea, input, [role="combobox"] {
          color: var(--ink) !important;
        }

        [data-testid="stSidebar"] [role="combobox"],
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
          color: var(--ink) !important;
          -webkit-text-fill-color: var(--ink) !important;
        }

        .stNumberInput button,
        .stDateInput button,
        [data-baseweb="input"] button,
        [data-baseweb="base-input"] button {
          background: #ffffff !important;
          color: var(--ink) !important;
          border: 1px solid rgba(59, 72, 63, 0.10) !important;
          border-radius: 12px !important;
        }

        .stNumberInput button span,
        .stDateInput button span,
        [data-baseweb="input"] button span,
        [data-baseweb="base-input"] button span {
          color: var(--ink) !important;
        }

        .account-badge {
          background: rgba(255,255,255,0.92);
          border: 1px solid rgba(31, 41, 55, 0.08);
          border-radius: 16px;
          padding: 12px 14px;
          margin-bottom: 12px;
          color: var(--ink) !important;
          box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }

        .account-badge .label {
          display: block;
          color: var(--muted) !important;
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-weight: 700;
          margin-bottom: 4px;
        }

        .account-badge .value {
          color: var(--ink) !important;
          font-weight: 700;
          font-size: 0.98rem;
        }

        .premium-hero {
          background:
            linear-gradient(135deg, rgba(255,255,255,0.98), rgba(247,250,249,0.98)),
            radial-gradient(circle at top right, rgba(184, 138, 68, 0.08), transparent 28%);
          border-radius: 30px;
          padding: 28px 30px;
          color: var(--ink);
          border: 1px solid rgba(59, 72, 63, 0.08);
          box-shadow: var(--shadow);
          margin-bottom: 16px;
        }

        .premium-kicker {
          text-transform: uppercase;
          letter-spacing: 0.14em;
          font-size: 0.72rem;
          opacity: 0.88;
          font-weight: 600;
          color: var(--muted);
        }

        .premium-hero h1 {
          color: var(--ink);
          margin: 0.35rem 0 0.4rem 0;
          font-size: 2.05rem;
          font-weight: 600;
          line-height: 1.15;
        }

        .premium-hero p {
          color: var(--muted);
          margin: 0;
          max-width: 700px;
          font-size: 1rem;
          line-height: 1.6;
        }

        .glass-card {
          background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,252,251,0.96));
          border: 1px solid rgba(59, 72, 63, 0.07);
          border-radius: 26px;
          padding: 20px 22px;
          box-shadow: var(--shadow);
          margin-bottom: 16px;
        }

        .coach-card {
          background: linear-gradient(180deg, #fffdfa, #faf7f2);
          border: 1px solid rgba(197, 162, 106, 0.18);
          border-radius: 24px;
          padding: 20px 22px;
          box-shadow: var(--shadow);
        }

        .mini-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px;
          margin-top: 14px;
        }

        .mini-stat {
          background: rgba(255,255,255,0.74);
          border: 1px solid rgba(59, 72, 63, 0.05);
          border-radius: 18px;
          padding: 14px 16px;
        }

        .mini-label {
          color: var(--muted);
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-weight: 700;
        }

        .mini-value {
          color: var(--ink);
          font-size: 1.45rem;
          font-weight: 600;
          margin-top: 6px;
        }

        .pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 10px;
        }

        .pill {
          padding: 8px 12px;
          border-radius: 999px;
          background: rgba(233, 240, 234, 0.92);
          color: var(--accent-forest);
          font-weight: 600;
          font-size: 0.85rem;
        }

        .progress-shell {
          margin-top: 10px;
          background: rgba(31, 41, 55, 0.08);
          border-radius: 999px;
          height: 10px;
          overflow: hidden;
        }

        .progress-bar {
          height: 10px;
          border-radius: 999px;
          background: linear-gradient(90deg, #d4b78c, #88a192);
        }

        .nutrient-row {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
        }

        .nutrient-card {
          border-radius: 18px;
          padding: 16px;
          background: rgba(255,255,255,0.72);
          border: 1px solid rgba(31, 51, 37, 0.06);
        }

        .table-wrap {
          background: rgba(255,255,255,0.76);
          border-radius: 22px;
          padding: 10px;
          border: 1px solid rgba(59, 72, 63, 0.07);
        }

        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stFormSubmitButton > button {
          background: linear-gradient(135deg, var(--accent-forest), #3e8571);
          color: #ffffff !important;
        }

        [data-testid="stSidebar"] input {
          color: var(--ink) !important;
          -webkit-text-fill-color: var(--ink) !important;
        }

        .stSelectbox label, .stNumberInput label, .stTextArea label, .stTextInput label, .stDateInput label {
          color: var(--ink) !important;
        }

        .stSlider label,
        .stSelectSlider label {
          color: var(--ink) !important;
        }

        .stAlert {
          color: var(--ink) !important;
        }

        @media (max-width: 768px) {
          .premium-hero { padding: 22px 20px; }
          .premium-hero h1 { font-size: 1.75rem; }
          .nutrient-row { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


store = get_store()
bootstrap_remembered_login()
if st.session_state.pop("clear_device_cookie", False):
    clear_cookie(DEVICE_COOKIE_NAME)
else:
    authenticate_from_cookie(store)

auth_username = st.session_state.get("auth_username")

if auth_username not in store["users"]:
    render_auth_screen(store)
    st.stop()
else:
    username = auth_username
    user = ensure_user(store, username)
    profile = user["profile"]
    render_premium_theme()

    pending_device_cookie = st.session_state.pop("pending_device_cookie", None)
    if pending_device_cookie:
        set_cookie(DEVICE_COOKIE_NAME, pending_device_cookie)

    with st.sidebar:
        st.header("Account")
        st.markdown(
            f"""
            <div class="account-badge">
              <span class="label">Logged In As</span>
              <span class="value">{username}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Log out", use_container_width=True):
            logout_user(store)
            st.rerun()

        st.markdown("---")
        st.header("Profile & Goals")
        with st.form("profile_form"):
            display_name_input = st.text_input("Display name", value=profile.get("display_name", username))
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
                profile["display_name"] = display_name_input.strip() or username
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
        st.caption(f"User data is saved in `{DATA_FILE.relative_to(APP_DIR)}`")

    selected_date = st.date_input("Tracking date", value=date.today())
    day_key = selected_date.isoformat()
    log = ensure_day(user, day_key)

    summary = day_totals(log)
    meal_completion = sum(1 for slot in MEAL_SLOTS if log["meals"][slot])
    streak = completion_streak(user)
    readiness = readiness_score(log, profile, summary)
    consistency = consistency_score(log, profile, summary)
    focus_title, focus_note = focus_message(log, profile, summary)
    coach_text = coach_insight(log, profile, summary)
    exercise_sessions = len(log["exercises"])
    total_duration = sum(int(ex.get("duration_min", 0)) for ex in log["exercises"])
    avg_duration = round(total_duration / exercise_sessions, 1) if exercise_sessions else 0.0
    calories_goal_delta = summary["calories_in"] - float(profile["calorie_goal"])

    st.markdown(
        f"""
        <section class="premium-hero">
          <div class="premium-kicker">Daily Coaching Studio</div>
          <h1>Build a calm, steady, healthy day.</h1>
          <p>
            A softer health dashboard that keeps your food rhythm, hydration, recovery,
            and movement in one clear place so progress feels grounded and easy to follow.
          </p>
          <div class="mini-grid">
            <div class="mini-stat">
              <div class="mini-label">Momentum Streak</div>
              <div class="mini-value">{streak} day{"s" if streak != 1 else ""}</div>
            </div>
            <div class="mini-stat">
              <div class="mini-label">Readiness</div>
              <div class="mini-value">{readiness}/100</div>
            </div>
            <div class="mini-stat">
              <div class="mini-label">Consistency</div>
              <div class="mini-value">{consistency}/100</div>
            </div>
            <div class="mini-stat">
              <div class="mini-label">Today's Focus</div>
              <div class="mini-value" style="font-size:1.05rem; line-height:1.35;">{focus_title}</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.25, 1])
    with top_left:
        st.markdown(
            f"""
            <div class="coach-card">
              <div class="premium-kicker" style="color: var(--accent-gold);">Coach Insight</div>
              <h3 style="margin: 0.45rem 0 0.65rem 0;">{focus_title}</h3>
              <p style="margin:0; color: var(--muted);">{coach_text}</p>
              <div class="pill-row">
                <div class="pill">Mood: {log.get("mood", "Steady")}</div>
                <div class="pill">Energy: {log.get("energy", "Balanced")}</div>
                <div class="pill">Net: {summary["net_calories"]:.0f} kcal</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="premium-kicker">Rhythm Overview</div>
              <div class="mini-grid">
                <div class="mini-stat">
                  <div class="mini-label">Meals Logged</div>
                  <div class="mini-value">{meal_completion}/4</div>
                </div>
                <div class="mini-stat">
                  <div class="mini-label">Exercise</div>
                  <div class="mini-value">{exercise_sessions}</div>
                </div>
                <div class="mini-stat">
                  <div class="mini-label">Average Session</div>
                  <div class="mini-value">{avg_duration:.0f} min</div>
                </div>
                <div class="mini-stat">
                  <div class="mini-label">Calorie Gap</div>
                  <div class="mini-value">{calories_goal_delta:+.0f}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Calories In", f"{summary['calories_in']:.0f} kcal", delta=f"{summary['calories_in'] - profile['calorie_goal']:.0f} vs goal")
    m2.metric("Protein", f"{summary['protein_g']:.0f} g", delta=f"{summary['protein_g'] - profile['protein_goal']:.0f} vs goal")
    m3.metric("Calories Out", f"{summary['calories_out']:.0f} kcal")
    m4.metric("Water", f"{log['water_ml']} ml", delta=f"{log['water_ml'] - profile['water_goal']:.0f} vs goal")
    m5.metric("Steps", f"{log['steps']}", delta=f"{log['steps'] - profile['step_goal']:.0f} vs goal")
    m6.metric("Meals Logged", f"{meal_completion}/4", delta=f"{4 - meal_completion} left")

    st.progress(min(summary["calories_in"] / max(profile["calorie_goal"], 1), 1.0), text="Calorie goal progress")
    st.progress(min(log["water_ml"] / max(profile["water_goal"], 1), 1.0), text="Water goal progress")

    dashboard_tab, meals_tab, exercise_tab, history_tab = st.tabs(["Command Center", "Meals", "Movement", "History & Export"])

    with dashboard_tab:
        left_col, right_col = st.columns([1.08, 0.92])

        with left_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Daily Check-In")
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
                        value=latest_known_weight(user, day_key),
                    )

                c5, c6 = st.columns(2)
                with c5:
                    mood_input = st.select_slider(
                        "Mood",
                        options=["Low", "Steady", "Good", "Great"],
                        value=log.get("mood", "Steady"),
                    )
                with c6:
                    energy_input = st.select_slider(
                        "Energy",
                        options=["Flat", "Balanced", "Sharp", "High"],
                        value=log.get("energy", "Balanced"),
                    )

                notes_input = st.text_area(
                    "Coach notes",
                    value=log.get("notes", ""),
                    placeholder="Write anything useful: hunger, cravings, digestion, training quality, schedule pressure, mood shifts.",
                    height=120,
                )
                save_daily_checkin = st.form_submit_button("Save daily check-in", use_container_width=True)
                if save_daily_checkin:
                    log["water_ml"] = int(water_input)
                    log["sleep_hours"] = float(sleep_input)
                    log["steps"] = int(steps_input)
                    log["weight_kg"] = float(weight_input)
                    log["mood"] = mood_input
                    log["energy"] = energy_input
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
                user["days"][day_key] = default_day_log()
                save_store(store)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Meal Rhythm")
            st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
            st.table(meal_slot_totals(log))
            st.markdown("</div></div>", unsafe_allow_html=True)

        with right_col:
            st.markdown(
                f"""
                <div class="glass-card">
                  <div class="premium-kicker">Performance Signals</div>
                  <h3 style="margin: 0.45rem 0 0.4rem 0;">Small steady choices matter most.</h3>
                  <p style="margin:0; color: var(--muted);">{focus_note}</p>
                  <div style="margin-top:16px;">
                    <div class="mini-label">Hydration</div>
                    <div class="progress-shell"><div class="progress-bar" style="width:{whole_ratio(log['water_ml'], float(profile['water_goal']))}%;"></div></div>
                    <div class="mini-label" style="margin-top:12px;">Protein</div>
                    <div class="progress-shell"><div class="progress-bar" style="width:{whole_ratio(summary['protein_g'], float(profile['protein_goal']))}%;"></div></div>
                    <div class="mini-label" style="margin-top:12px;">Steps</div>
                    <div class="progress-shell"><div class="progress-bar" style="width:{whole_ratio(log['steps'], float(profile['step_goal']))}%;"></div></div>
                    <div class="mini-label" style="margin-top:12px;">Sleep</div>
                    <div class="progress-shell"><div class="progress-bar" style="width:{min(int((float(log['sleep_hours']) / 8.0) * 100), 100)}%;"></div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="premium-kicker">Macro Balance</div>', unsafe_allow_html=True)
            n1, n2, n3 = st.columns(3)
            nutrient_items = nutrient_split(summary)
            n1.metric(nutrient_items[0]["label"], f"{nutrient_items[0]['value']:.0f}{nutrient_items[0]['unit']}")
            n2.metric(nutrient_items[1]["label"], f"{nutrient_items[1]['value']:.0f}{nutrient_items[1]['unit']}")
            n3.metric(nutrient_items[2]["label"], f"{nutrient_items[2]['value']:.0f}{nutrient_items[2]['unit']}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="coach-card">
                  <div class="premium-kicker" style="color: var(--accent-coral);">Coaching Cue</div>
                  <h3 style="margin: 0.45rem 0 0.6rem 0;">Finish today with balance.</h3>
                  <p style="margin:0; color: var(--muted);">{coach_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with meals_tab:
        st.markdown(
            """
            <div class="glass-card">
              <div class="premium-kicker">Food Log</div>
              <h3 style="margin: 0.45rem 0 0.45rem 0;">Write meals the way you actually think.</h3>
              <p style="margin:0; color: var(--muted);">Use natural language, rough portions, or quick templates. The goal is frictionless consistency, not perfect nutrition math.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.setdefault("meal_text_input", "")
        st.session_state.setdefault("meal_text_next_value", None)

        pending_meal_text = st.session_state.pop("meal_text_next_value", None)
        if pending_meal_text is not None:
            st.session_state["meal_text_input"] = pending_meal_text

        meal_flash_message = st.session_state.pop("meal_flash_message", None)
        if meal_flash_message:
            st.success(meal_flash_message)

        template_cols = st.columns(4)
        templates = [
            ("Indian Balanced", "2 rotis, 1 cup dal, salad, curd"),
            ("High Protein", "3 eggs, 1 bowl curd, 1 scoop whey"),
            ("Light Breakfast", "1 banana, oats with milk, 5 almonds"),
            ("Post Workout", "grilled chicken, rice, curd"),
        ]
        for idx, (label, value) in enumerate(templates):
            if template_cols[idx].button(label, use_container_width=True, key=f"meal-template-{idx}"):
                st.session_state["meal_text_next_value"] = value
                st.rerun()

        meal_slot = st.selectbox("Meal type", MEAL_SLOTS)
        meal_text = st.text_area(
            "What did you eat? Include quantity.",
            placeholder="Example: 2 egg omelette with 2 slices toast and 1 banana",
            height=110,
            key="meal_text_input",
        )

        if st.button("Estimate and add meal", type="primary", use_container_width=True):
            meal_text_clean = meal_text.strip()
            if not meal_text_clean:
                st.warning("Please enter your meal first.")
            else:
                with st.spinner("Estimating calories and macros..."):
                    result = estimate_meal_from_text(meal_text_clean)
                log["meals"][meal_slot].append(
                    {
                        "logged_at": datetime.now().strftime("%H:%M"),
                        "input_text": meal_text_clean,
                        **result,
                    }
                )
                touch_log(log)
                save_store(store)
                st.session_state["meal_text_next_value"] = ""
                st.session_state["meal_flash_message"] = f"Added to {meal_slot}."
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
                            f"Estimated via {entry.get('source', 'Unknown')} | confidence {float(entry.get('confidence', 0)):.2f}"
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
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="premium-kicker">Movement Log</div>
              <h3 style="margin: 0.45rem 0 0.45rem 0;">Train for momentum, not punishment.</h3>
              <p style="margin:0; color: var(--muted);">You have logged {exercise_sessions} session{"s" if exercise_sessions != 1 else ""} today with {summary['calories_out']:.0f} kcal burned. Keep intensity aligned with recovery.</p>
              <div class="mini-grid">
                <div class="mini-stat"><div class="mini-label">Sessions</div><div class="mini-value">{exercise_sessions}</div></div>
                <div class="mini-stat"><div class="mini-label">Burn</div><div class="mini-value">{summary['calories_out']:.0f}</div></div>
                <div class="mini-stat"><div class="mini-label">Time</div><div class="mini-value">{total_duration}</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        st.dataframe(history_rows(user, days=7), use_container_width=True, hide_index=True)

        st.markdown("### Export current day")
        payload = make_export_payload(day_key, log)
        st.json(payload)

        st.download_button(
            "Download day JSON",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name=f"{username}_fitness_log_{day_key}.json",
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
            file_name=f"{username}_meals_{day_key}.csv",
            mime="text/csv",
        )

    st.info(
        "Each user now has a separate account and private tracking data. Passwords are hashed, and you can stay logged in on the current device."
    )