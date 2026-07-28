"""Small AI API used by the native Calorie Lens apps.

Run locally with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from calorie_engine import estimate_command

try:
    from google import genai
    from google.genai import types as genai_types
except ModuleNotFoundError:
    genai = None
    genai_types = None


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.6-flash")
CLIENT = genai.Client(api_key=GOOGLE_API_KEY) if genai and GOOGLE_API_KEY else None

app = FastAPI(title="Calorie Lens AI", version="2.0.0")
origins = [
    origin.strip()
    for origin in os.getenv("CALORIE_LENS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TextCommand(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    preferred_slot: Optional[str] = None
    weight_kg: Optional[float] = Field(default=None, ge=20, le=400)
    bowl_ml: Optional[float] = Field(default=None, ge=50, le=1000)
    cup_ml: float = Field(default=200, ge=50, le=1000)
    previous_transcript: Optional[str] = Field(default=None, max_length=2000)
    clarification_question: Optional[str] = Field(default=None, max_length=1000)


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
    return {"ok": True, "ai_enabled": CLIENT is not None, "model": MODEL_NAME}


@app.post("/v1/parse-command")
def parse_command(command: TextCommand) -> dict[str, Any]:
    return _generate(
        command.text,
        command.preferred_slot,
        profile={
            "weightKg": command.weight_kg,
            "bowlMl": command.bowl_ml,
            "cupMl": command.cup_ml,
        },
        previous_transcript=command.previous_transcript,
        clarification_question=command.clarification_question,
    )


@app.post("/v1/parse-command/audio")
async def parse_audio_command(
    audio: UploadFile = File(...),
    preferred_slot: Optional[str] = Form(default=None),
    weight_kg: Optional[float] = Form(default=None),
    bowl_ml: Optional[float] = Form(default=None),
    cup_ml: float = Form(default=200),
    previous_transcript: Optional[str] = Form(default=None),
    clarification_question: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="The recording was empty.")
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The recording is too large.")
    return _generate(
        "",
        preferred_slot,
        audio_bytes=content,
        audio_mime_type=audio.content_type or "audio/mp4",
        profile={"weightKg": weight_kg, "bowlMl": bowl_ml, "cupMl": cup_ml},
        previous_transcript=previous_transcript,
        clarification_question=clarification_question,
    )
