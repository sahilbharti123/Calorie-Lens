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

app = FastAPI(title="Calorie Lens AI", version="1.0.0")
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


COMMAND_SCHEMA = {
    "type": "object",
    "required": ["transcript", "confirmation", "operations"],
    "properties": {
        "transcript": {"type": "string"},
        "confirmation": {"type": "string"},
        "operations": {
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
                            "required": [
                                "name",
                                "quantity",
                                "calories",
                                "protein",
                                "carbs",
                                "fat",
                                "source",
                            ],
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": "string"},
                                "calories": {"type": "number"},
                                "protein": {"type": "number"},
                                "carbs": {"type": "number"},
                                "fat": {"type": "number"},
                                "source": {"type": "string", "enum": ["ai"]},
                            },
                        },
                    },
                    "amount": {"type": "number"},
                    "name": {"type": "string"},
                    "durationMin": {"type": "number"},
                    "calories": {"type": "number"},
                    "intensity": {
                        "type": "string",
                        "enum": ["light", "moderate", "hard"],
                    },
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You are the private nutrition and fitness logging engine for Calorie Lens.
Turn the user's short typed or spoken update into one or more operations.

Rules:
- The user often speaks English, Hindi, or Hinglish.
- For every meal operation, estimate each item's calories and protein/carbs/fat in grams.
- Be practical with Indian foods and respect explicit quantities.
- Treat 1 glass of water as 250 ml and 1 bottle as 750 ml unless specified.
- For workouts, estimate active calories conservatively and include duration and intensity.
- Use action "set" for steps, sleep, and weight unless the user clearly says to add.
- Meal slots are breakfast, lunch, snack, and dinner.
- If no slot is stated, respect the preferred slot when given; otherwise use local time:
  before 11 breakfast, 11-16 lunch, 16-19 snack, after 19 dinner.
- Keep transcript faithful. The confirmation should be a short review summary.
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
) -> dict[str, Any]:
    _require_ai()
    context = (
        f"{SYSTEM_PROMPT}\nCurrent local hour: {datetime.now().hour}."
        f"\nPreferred meal slot: {preferred_slot or 'none'}."
        f"\nTyped update: {typed_text or 'none'}."
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

    if not payload.get("operations"):
        raise HTTPException(status_code=422, detail="No fitness updates were found.")
    payload["source"] = "ai"
    return payload


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ai_enabled": CLIENT is not None, "model": MODEL_NAME}


@app.post("/v1/parse-command")
def parse_command(command: TextCommand) -> dict[str, Any]:
    return _generate(command.text, command.preferred_slot)


@app.post("/v1/parse-command/audio")
async def parse_audio_command(
    audio: UploadFile = File(...),
    preferred_slot: Optional[str] = Form(default=None),
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
    )
