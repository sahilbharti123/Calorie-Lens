"""Small AI API used by the native Calorie Lens apps.

Run locally with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from account_store import (
    AIBudgetExceededError,
    AccountExistsError,
    AccountStore,
    InvalidCredentialsError,
    SyncConflictError,
)
from calorie_engine import estimate_command

try:
    from google import genai
    from google.genai import types as genai_types
except ModuleNotFoundError:
    genai = None
    genai_types = None


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.1-flash-lite")
CLIENT = genai.Client(api_key=GOOGLE_API_KEY) if genai and GOOGLE_API_KEY else None
STORE = AccountStore()
AI_DAILY_LIMIT = max(1, int(os.getenv("CALORIE_LENS_AI_DAILY_LIMIT", "8")))
AI_KIND_LIMITS = {
    "text": max(1, int(os.getenv("CALORIE_LENS_AI_TEXT_DAILY_LIMIT", "4"))),
    "audio": max(1, int(os.getenv("CALORIE_LENS_AI_AUDIO_DAILY_LIMIT", "4"))),
    "coach": max(1, int(os.getenv("CALORIE_LENS_AI_COACH_DAILY_LIMIT", "2"))),
}

app = FastAPI(title="Calorie Lens API", version="3.0.0")
origins = [
    origin.strip()
    for origin in os.getenv("CALORIE_LENS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


class SignupRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=10, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class RecoveryRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    recovery_code: str = Field(min_length=15, max_length=100)
    new_password: str = Field(min_length=10, max_length=128)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=10, max_length=128)


class SyncRequest(BaseModel):
    payload: dict[str, Any]
    base_version: Optional[int] = Field(default=None, ge=0)


class CoachRequest(BaseModel):
    message: str = Field(min_length=1, max_length=800)
    today: Optional[dict[str, Any]] = None
    memory: Optional[dict[str, Any]] = None
    recent_messages: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


class TextCommand(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    preferred_slot: Optional[str] = None
    weight_kg: Optional[float] = Field(default=None, ge=20, le=400)
    bowl_ml: Optional[float] = Field(default=None, ge=50, le=1000)
    cup_ml: float = Field(default=200, ge=50, le=1000)
    previous_transcript: Optional[str] = Field(default=None, max_length=600)
    clarification_question: Optional[str] = Field(default=None, max_length=300)


COMMAND_SCHEMA = {
    "type": "object",
    "required": ["transcript", "intents"],
    "properties": {
        "transcript": {"type": "string"},
        "profileWeightKg": {"type": "number"},
        "bowlMl": {"type": "number"},
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "action"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["meal", "water", "workout", "steps", "sleep", "weight"],
                    },
                    "action": {"type": "string", "enum": ["add", "set"]},
                    "slot": {
                        "type": "string",
                        "enum": ["breakfast", "lunch", "snack", "dinner"],
                    },
                    "description": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "amount", "unit"],
                            "properties": {
                                "name": {"type": "string"},
                                "amount": {"type": "number"},
                                "unit": {
                                    "type": "string",
                                    "enum": [
                                        "g",
                                        "kg",
                                        "ml",
                                        "l",
                                        "piece",
                                        "bowl",
                                        "cup",
                                        "tbsp",
                                        "tsp",
                                        "scoop",
                                        "serving",
                                        "unknown",
                                    ],
                                },
                                "preparation": {"type": "string"},
                                "oilTsp": {"type": "number"},
                                "labelCalories": {"type": "number"},
                                "labelProtein": {"type": "number"},
                                "labelCarbs": {"type": "number"},
                                "labelFat": {"type": "number"},
                            },
                        },
                    },
                    "amount": {"type": "number"},
                    "name": {"type": "string"},
                    "durationMin": {"type": "number"},
                    "intensity": {
                        "type": "string",
                        "enum": ["light", "moderate", "hard"],
                    },
                    "speedKmh": {"type": "number"},
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You are the private nutrition and fitness logging engine for Calorie Lens.
Turn the user's short typed or spoken update into structured facts. A deterministic
reference engine calculates calories after you respond.

Rules:
- The user often speaks English, Hindi, or Hinglish.
- Never estimate calories, nutrients, grams, portion size, or workout burn.
- For every meal item, extract the food name, numeric amount, unit, preparation,
  and oil in the user's portion when stated.
- Preserve Indian food names such as rajma, dal, roti, poha, idli and dosa.
- If quantity was not stated, use amount 0 and unit "unknown". Never invent one.
- Convert explicit water units to ml. Treat a glass as 250 ml and bottle as 750 ml
  only when the user did not give its size.
- For workouts, extract the activity, duration, stated intensity, and speed when stated.
  Omit intensity when it was not stated. Do not infer body weight or calorie burn.
- When answering a body-weight clarification, set profileWeightKg to the stated kg.
- When answering a usual-bowl-size clarification, set bowlMl to the stated ml.
- When the user gives package or restaurant calories for the amount consumed,
  set labelCalories on that food. Extract labelProtein, labelCarbs, and labelFat
  only when the user states them. Never invent missing label macros.
- Use action "set" for steps, sleep, and weight unless the user clearly says to add.
- Meal slots are breakfast, lunch, snack, and dinner.
- If no slot is stated, respect the preferred slot when given; otherwise use local time:
  before 11 breakfast, 11-16 lunch, 16-19 snack, after 19 dinner.
- Keep transcript faithful, including any clarification answer.
- Do not give medical advice.
"""

COACH_PROMPT = """You are Calorie Lens, a supportive personal fitness coach.
Use the user's saved preferences, limitations, goals, and recent tracking data.
Be concise, practical, and non-judgmental. Never diagnose, prescribe, or claim
medical certainty. Encourage professional care for symptoms, eating-disorder
concerns, injuries, pregnancy, or other high-risk situations. Do not invent
logged facts. Calorie and exercise values are estimates."""


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _optional_user(
    authorization: Optional[str] = Header(default=None),
) -> Optional[dict[str, Any]]:
    token = _bearer_token(authorization)
    return STORE.user_for_token(token) if token else None


def _current_user(
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    user = _optional_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user


def _required_token(authorization: Optional[str]) -> str:
    token = _bearer_token(authorization)
    if not token or not STORE.user_for_token(token):
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return token


def _session_payload(
    user: dict[str, Any],
    *,
    recovery_code: Optional[str] = None,
) -> dict[str, Any]:
    token, expires_at = STORE.create_session(user["id"])
    payload: dict[str, Any] = {
        "token": token,
        "expiresAt": expires_at,
        "user": user,
    }
    if recovery_code:
        payload["recoveryCode"] = recovery_code
    return payload


def _vault_memory(user: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not user:
        return {}
    try:
        vault = STORE.read_vault(user["id"])["payload"]
    except (KeyError, ValueError):
        return {}
    memory = vault.get("coachMemory")
    return memory if isinstance(memory, dict) else {}


def _cache_key(kind: str, user_id: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "kind": kind,
            "model": MODEL_NAME,
            "user": user_id,
            "payload": payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_json(value: Any, max_chars: int) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )[:max_chars]


def _charge_ai(user_id: str, kind: str) -> dict[str, Any]:
    try:
        return STORE.consume_ai_request(
            user_id,
            kind,
            AI_DAILY_LIMIT,
            AI_KIND_LIMITS,
        )
    except AIBudgetExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "message": str(exc),
                "usage": exc.usage,
            },
        ) from exc


def _cached_result(user_id: str, cache_key: str) -> Optional[dict[str, Any]]:
    cached = STORE.read_ai_cache(user_id, cache_key)
    if not cached:
        return None
    return {
        **cached,
        "cost": {"cached": True, "model": MODEL_NAME},
    }


def _save_cached_result(
    user_id: str,
    cache_key: str,
    result: dict[str, Any],
    *,
    days: int,
) -> dict[str, Any]:
    STORE.write_ai_cache(user_id, cache_key, result, timedelta(days=days))
    return {
        **result,
        "cost": {"cached": False, "model": MODEL_NAME},
    }


@app.post("/v1/auth/signup", status_code=201)
def signup(request: SignupRequest) -> dict[str, Any]:
    try:
        user, recovery_code = STORE.create_user(
            request.email, request.password, request.display_name
        )
    except AccountExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _session_payload(user, recovery_code=recovery_code)


@app.post("/v1/auth/login")
def login(request: LoginRequest) -> dict[str, Any]:
    try:
        user = STORE.authenticate(request.email, request.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return _session_payload(user)


@app.post("/v1/auth/recover")
def recover_account(request: RecoveryRequest) -> dict[str, Any]:
    try:
        user_id, recovery_code = STORE.recover(
            request.email, request.recovery_code, request.new_password
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {
        "ok": True,
        "userId": user_id,
        "recoveryCode": recovery_code,
    }


@app.get("/v1/auth/me")
def auth_me(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    return {"user": user}


@app.post("/v1/auth/logout")
def logout(
    authorization: Optional[str] = Header(default=None),
) -> dict[str, bool]:
    token = _required_token(authorization)
    STORE.logout(token)
    return {"ok": True}


@app.post("/v1/auth/change-password")
def change_password(
    request: PasswordChangeRequest,
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, bool]:
    try:
        STORE.change_password(
            user["id"], request.current_password, request.new_password
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {"ok": True}


@app.delete("/v1/account")
def delete_account(
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, bool]:
    STORE.delete_user(user["id"])
    return {"ok": True}


@app.get("/v1/sync")
def read_sync(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    return STORE.read_vault(user["id"])


@app.put("/v1/sync")
def write_sync(
    request: SyncRequest,
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    try:
        return STORE.save_vault(
            user["id"], request.payload, request.base_version
        )
    except SyncConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "payload": exc.payload,
                "version": exc.version,
                "updatedAt": exc.updated_at,
            },
        ) from exc


@app.get("/v1/backup")
def export_backup(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    vault = STORE.read_vault(user["id"])
    return {
        "format": "calorie-lens-backup-v1",
        "exportedAt": datetime.now().astimezone().isoformat(),
        "payload": vault["payload"],
        "version": vault["version"],
    }


@app.post("/v1/backup/restore")
def restore_backup(
    request: SyncRequest,
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    return STORE.save_vault(user["id"], request.payload)


def _require_ai() -> None:
    if CLIENT is None or genai_types is None:
        raise HTTPException(
            status_code=503,
            detail="Set GOOGLE_API_KEY and install google-genai to enable AI and voice.",
        )


def _generate(
    typed_text: str,
    preferred_slot: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    audio_mime_type: str = "audio/mp4",
    profile: Optional[dict[str, Any]] = None,
    coach_memory: Optional[dict[str, Any]] = None,
    previous_transcript: Optional[str] = None,
    clarification_question: Optional[str] = None,
) -> dict[str, Any]:
    _require_ai()
    context = (
        f"{SYSTEM_PROMPT}\nCurrent local hour: {datetime.now().hour}."
        f"\nPreferred meal slot: {preferred_slot or 'none'}."
        f"\nPrevious update: {previous_transcript or 'none'}."
        f"\nQuestion awaiting an answer: {clarification_question or 'none'}."
        f"\nTyped update: {typed_text or 'none'}."
        f"\nSaved user preferences and limitations: "
        f"{json.dumps(coach_memory or {}, ensure_ascii=False)}."
        "\nWhen a previous update and question are present, merge the new answer into "
        "the original facts and return the complete combined intent."
    )
    contents: list[Any] = [context]
    if audio_bytes:
        contents.append(
            genai_types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type)
        )

    try:
        response = CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=COMMAND_SCHEMA,
                max_output_tokens=512,
            ),
        )
        payload = json.loads(getattr(response, "text", "") or "{}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}") from exc

    if not payload.get("intents"):
        raise HTTPException(status_code=422, detail="No fitness updates were found.")
    effective_profile = dict(profile or {})
    profile_updates: dict[str, float] = {}
    if payload.get("profileWeightKg"):
        weight_kg = float(payload["profileWeightKg"])
        if 20 <= weight_kg <= 400:
            effective_profile["weightKg"] = weight_kg
            profile_updates["weightKg"] = weight_kg
    if payload.get("bowlMl"):
        bowl_ml = float(payload["bowlMl"])
        if 50 <= bowl_ml <= 1000:
            effective_profile["bowlMl"] = bowl_ml
            profile_updates["bowlMl"] = bowl_ml
    result = estimate_command(payload, effective_profile)
    if profile_updates:
        result["profileUpdates"] = profile_updates
    result["source"] = "ai"
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ai_enabled": CLIENT is not None,
        "accounts_enabled": True,
        "encrypted_storage": True,
        "model": MODEL_NAME,
        "ai_daily_limit": AI_DAILY_LIMIT,
    }


@app.get("/v1/ai/usage")
def ai_usage(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    return STORE.ai_usage_summary(
        user["id"],
        AI_DAILY_LIMIT,
        AI_KIND_LIMITS,
    )


@app.post("/v1/parse-command")
def parse_command(
    command: TextCommand,
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    memory = _vault_memory(user)
    cache_key = _cache_key(
        "text",
        user["id"],
        {
            **command.model_dump(),
            "memory": memory,
        },
    )
    cached = _cached_result(user["id"], cache_key)
    if cached:
        return cached
    _charge_ai(user["id"], "text")
    result = _generate(
        command.text,
        command.preferred_slot,
        profile={
            "weightKg": command.weight_kg,
            "bowlMl": command.bowl_ml,
            "cupMl": command.cup_ml,
        },
        coach_memory=memory,
        previous_transcript=command.previous_transcript,
        clarification_question=command.clarification_question,
    )
    return _save_cached_result(user["id"], cache_key, result, days=30)


@app.post("/v1/parse-command/audio")
async def parse_audio_command(
    audio: UploadFile = File(...),
    preferred_slot: Optional[str] = Form(default=None),
    weight_kg: Optional[float] = Form(default=None),
    bowl_ml: Optional[float] = Form(default=None),
    cup_ml: float = Form(default=200),
    previous_transcript: Optional[str] = Form(default=None),
    clarification_question: Optional[str] = Form(default=None),
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="The recording was empty.")
    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Keep voice commands short (maximum recording size is 2 MB).",
        )
    memory = _vault_memory(user)
    cache_key = _cache_key(
        "audio",
        user["id"],
        {
            "audioSha256": hashlib.sha256(content).hexdigest(),
            "preferredSlot": preferred_slot,
            "weightKg": weight_kg,
            "bowlMl": bowl_ml,
            "cupMl": cup_ml,
            "previousTranscript": previous_transcript,
            "clarificationQuestion": clarification_question,
            "memory": memory,
        },
    )
    cached = _cached_result(user["id"], cache_key)
    if cached:
        return cached
    _charge_ai(user["id"], "audio")
    result = _generate(
        "",
        preferred_slot,
        audio_bytes=content,
        audio_mime_type=audio.content_type or "audio/mp4",
        profile={"weightKg": weight_kg, "bowlMl": bowl_ml, "cupMl": cup_ml},
        coach_memory=memory,
        previous_transcript=previous_transcript,
        clarification_question=clarification_question,
    )
    return _save_cached_result(user["id"], cache_key, result, days=30)


@app.post("/v1/coach")
def coach(
    request: CoachRequest,
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    _require_ai()
    if len(_compact_json(request.model_dump(), 12_001)) > 12_000:
        raise HTTPException(status_code=413, detail="The coach context is too large.")
    memory = {**_vault_memory(user), **(request.memory or {})}
    recent_messages = request.recent_messages[-6:]
    cache_key = _cache_key(
        "coach",
        user["id"],
        {
            **request.model_dump(),
            "memory": memory,
            "recent_messages": recent_messages,
        },
    )
    cached = _cached_result(user["id"], cache_key)
    if cached:
        return cached
    usage = _charge_ai(user["id"], "coach")
    context = (
        f"{COACH_PROMPT}\n"
        f"User: {user['displayName']}.\n"
        f"Saved long-term memory: {_compact_json(memory, 2_000)}.\n"
        f"Today's tracking snapshot: "
        f"{_compact_json(request.today or {}, 4_000)}.\n"
        f"Recent conversation: "
        f"{_compact_json(recent_messages, 3_000)}.\n"
        f"New message: {request.message}"
    )
    try:
        response = CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=[context],
            config=genai_types.GenerateContentConfig(max_output_tokens=180),
        )
        reply = (getattr(response, "text", "") or "").strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Coach request failed: {exc}") from exc
    if not reply:
        raise HTTPException(status_code=502, detail="The coach did not return a response.")
    return _save_cached_result(
        user["id"],
        cache_key,
        {"reply": reply, "usage": usage},
        days=1,
    )
