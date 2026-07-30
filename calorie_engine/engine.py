"""Deterministic estimation from model-extracted food and workout facts."""

from __future__ import annotations

from typing import Any, Optional

from .catalog import FoodReference, find_food
from .exercise import active_kcal, find_activity, mets_for

USDA_LABEL = "USDA FoodData Central"
COMPENDIUM_LABEL = "2024 Adult Compendium of Physical Activities"


def _round_macro(value: float) -> float:
    return round(value, 1)


def _clarification(
    transcript: str,
    question: str,
    suggestions: list[str],
) -> dict[str, Any]:
    return {
        "transcript": transcript,
        "confirmation": "One detail will improve this estimate",
        "operations": [],
        "clarification": {
            "question": question,
            "suggestions": suggestions,
        },
    }


def _quantity_grams(
    item: dict[str, Any],
    reference: FoodReference,
    profile: dict[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[float], str, Optional[dict[str, Any]]]:
    amount = float(item.get("amount") or 0)
    unit = str(item.get("unit") or "unknown").casefold()
    if unit in {"slice", "slices"}:
        unit = "piece"
    elif unit in {"katori", "katoris"}:
        unit = "bowl"
    if amount <= 0:
        return 0, 0, 0, "", {
            "question": f"How much {item.get('name') or reference.name} did you have?",
            "suggestions": ["100 g", "1 piece", "1 bowl", "1 cup"],
        }

    if unit == "kg":
        grams = amount * 1000
        return grams, grams * 0.98, grams * 1.02, f"{amount:g} kg", None
    if unit in {"l", "liter", "liters", "litre", "litres"}:
        item = {**item, "amount": amount * 1000, "unit": "ml"}
        return _quantity_grams(item, reference, profile)
    if unit in {"g", "gram", "grams"}:
        return amount, amount * 0.98, amount * 1.02, f"{amount:g} g", None
    if unit in {"piece", "pieces", "item", "items"}:
        if reference.piece_grams is None:
            return 0, 0, 0, "", {
                "question": f"About how many grams was each {item.get('name') or reference.name}?",
                "suggestions": ["30 g", "50 g", "75 g", "100 g"],
            }
        grams = amount * reference.piece_grams
        variance = reference.piece_variance
        return (
            grams,
            grams * (1 - variance),
            grams * (1 + variance),
            f"{amount:g} × {reference.piece_grams:g} g standard piece",
            None,
        )

    if unit == "bowl":
        bowl_ml = float(profile.get("bowlMl") or 0)
        if bowl_ml <= 0:
            return 0, 0, 0, "", {
                "question": "About how large is your usual bowl?",
                "suggestions": ["150 ml", "200 ml", "250 ml", "300 ml"],
            }
        if reference.density_g_ml is None:
            return 0, 0, 0, "", {
                "question": f"About how many grams fit in that bowl of {item.get('name') or reference.name}?",
                "suggestions": ["100 g", "150 g", "200 g", "250 g"],
            }
        volume = amount * bowl_ml
        grams = volume * reference.density_g_ml
        variance = reference.density_variance
        return (
            grams,
            grams * (1 - variance),
            grams * (1 + variance),
            f"{amount:g} × {bowl_ml:g} ml bowl",
            None,
        )

    if unit in {"cup", "cups"}:
        cup_ml = float(profile.get("cupMl") or 200)
        if reference.density_g_ml is None:
            return 0, 0, 0, "", {
                "question": f"About how many grams were in the cup of {item.get('name') or reference.name}?",
                "suggestions": ["100 g", "150 g", "200 g", "250 g"],
            }
        volume = amount * cup_ml
        grams = volume * reference.density_g_ml
        variance = reference.density_variance
        return (
            grams,
            grams * (1 - variance),
            grams * (1 + variance),
            f"{amount:g} × {cup_ml:g} ml cup",
            None,
        )

    if unit in {"glass", "glasses"}:
        if reference.density_g_ml is None:
            return 0, 0, 0, "", {
                "question": f"About how many grams were in the glass of {item.get('name') or reference.name}?",
                "suggestions": ["100 g", "150 g", "200 g", "250 g"],
            }
        volume = amount * 250
        grams = volume * reference.density_g_ml
        variance = reference.density_variance
        return (
            grams,
            grams * (1 - variance),
            grams * (1 + variance),
            f"{amount:g} × 250 ml glass",
            None,
        )

    if unit in {"ml", "milliliter", "milliliters"}:
        if reference.density_g_ml is None:
            return 0, 0, 0, "", {
                "question": f"About how many grams was {amount:g} ml of {item.get('name') or reference.name}?",
                "suggestions": ["100 g", "150 g", "200 g", "250 g"],
            }
        grams = amount * reference.density_g_ml
        variance = reference.density_variance
        return (
            grams,
            grams * (1 - variance),
            grams * (1 + variance),
            f"{amount:g} ml",
            None,
        )

    if unit in {"tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons"}:
        if reference.density_g_ml is None:
            return 0, 0, 0, "", {
                "question": f"Can you give the grams for the {item.get('name') or reference.name}?",
                "suggestions": ["5 g", "10 g", "15 g", "20 g"],
            }
        ml_each = 15 if unit.startswith("tb") else 5
        grams = amount * ml_each * reference.density_g_ml
        return (
            grams,
            grams * (1 - reference.density_variance),
            grams * (1 + reference.density_variance),
            f"{amount:g} {unit}",
            None,
        )

    return 0, 0, 0, "", {
        "question": f"What was the quantity of {item.get('name') or reference.name}?",
        "suggestions": ["100 g", "1 piece", "1 bowl", "1 cup"],
    }


def _estimate_food_item(
    item: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    raw_name = str(item.get("name") or "").strip()
    label_calories = float(item.get("labelCalories") or 0)
    if label_calories > 0:
        amount = float(item.get("amount") or 1)
        unit = str(item.get("unit") or "serving")
        quantity = (
            f"{amount:g} {unit}"
            if unit != "unknown"
            else "amount described by user"
        )
        return {
            "name": raw_name or "Packaged food",
            "quantity": quantity,
            "calories": round(label_calories),
            "calorieLow": round(label_calories * 0.95),
            "calorieHigh": round(label_calories * 1.05),
            "protein": _round_macro(float(item.get("labelProtein") or 0)),
            "carbs": _round_macro(float(item.get("labelCarbs") or 0)),
            "fat": _round_macro(float(item.get("labelFat") or 0)),
            "source": "label",
            "sourceLabel": "Food label supplied by user",
            "sourceId": "declared serving",
            "confidence": "medium",
            "basis": f"{label_calories:g} kcal declared for the amount consumed",
            "assumptions": ["label rounding and serving accuracy still apply"],
        }, None
    reference = find_food(raw_name)
    if reference is None:
        return None, {
            "question": (
                f"I do not have a verified reference for “{raw_name or 'that food'}” yet. "
                "Give its label calories, or log the main parts with amounts."
            ),
            "suggestions": [
                "e.g. 350 kcal from the label",
                "e.g. 150 g rice and 1 bowl dal",
                "e.g. 2 rotis and 100 g paneer",
            ],
        }

    grams, grams_low, grams_high, quantity_label, missing = _quantity_grams(
        item, reference, profile
    )
    if missing:
        return None, missing
    assert grams is not None and grams_low is not None and grams_high is not None

    scale = grams / 100
    low_scale = grams_low / 100
    high_scale = grams_high / 100
    calories = reference.calories * scale
    calories_low = reference.calories * low_scale
    calories_high = reference.calories * high_scale
    protein = reference.protein * scale
    carbs = reference.carbs * scale
    fat = reference.fat * scale
    assumptions: list[str] = []

    normalized_name = raw_name.casefold()
    preparation = str(item.get("preparation") or "").casefold()
    oil_tsp = item.get("oilTsp")
    is_home_curry = reference.key in {"kidney_beans", "lentils", "chickpeas"} and (
        "curry" in preparation
        or "gravy" in preparation
        or (
            any(word in normalized_name for word in ("rajma", "dal", "chole", "chana"))
            and not any(style in preparation for style in ("plain", "boiled", "dry"))
        )
    )
    if oil_tsp is not None and float(oil_tsp) >= 0:
        oil_grams = float(oil_tsp) * 4.6
        oil_calories = oil_grams * 8.84
        calories += oil_calories
        calories_low += oil_calories * 0.95
        calories_high += oil_calories * 1.05
        fat += oil_grams
        assumptions.append(f"{float(oil_tsp):g} tsp oil in your portion")
    elif is_home_curry:
        midpoint_oil_kcal = 4.6 * 8.84
        calories += midpoint_oil_kcal
        calories_low += 0.5 * midpoint_oil_kcal
        calories_high += 2 * midpoint_oil_kcal
        fat += 4.6
        assumptions.append("home curry range assumes ½–2 tsp oil in this portion")

    exact_grams = str(item.get("unit") or "").casefold() in {"g", "gram", "grams", "kg"}
    confidence = "high" if exact_grams and not assumptions else "medium"
    if "bowl" in quantity_label or (assumptions and not exact_grams):
        confidence = "low"
    source_id = f"FDC {reference.fdc_id}"
    basis = (
        f"{quantity_label} · {reference.calories:g} kcal/100 g"
        + (f" · {assumptions[0]}" if assumptions else "")
    )
    return {
        "name": reference.name,
        "quantity": quantity_label,
        "calories": round(calories),
        "calorieLow": round(calories_low),
        "calorieHigh": round(calories_high),
        "protein": _round_macro(protein),
        "carbs": _round_macro(carbs),
        "fat": _round_macro(fat),
        "source": "usda",
        "sourceLabel": USDA_LABEL,
        "sourceId": source_id,
        "confidence": confidence,
        "basis": basis,
        "assumptions": assumptions,
    }, None


def _estimate_workout(
    intent: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    name = str(intent.get("name") or "").strip()
    activity = find_activity(name)
    if activity is None:
        return None, {
            "question": "What kind of exercise was it?",
            "suggestions": ["Walking", "Running", "Strength training", "Cycling"],
        }
    duration = float(intent.get("durationMin") or 0)
    if duration <= 0:
        return None, {
            "question": f"How many minutes did you do {activity.name.lower()}?",
            "suggestions": ["15 min", "30 min", "45 min", "60 min"],
        }
    weight = float(profile.get("weightKg") or 0)
    if weight <= 0:
        return None, {
            "question": "What is your current body weight? I need it to estimate active calories.",
            "suggestions": ["60 kg", "70 kg", "80 kg", "90 kg"],
        }
    raw_intensity = intent.get("intensity")
    speed = intent.get("speedKmh")
    if not raw_intensity and not speed:
        return None, {
            "question": f"How hard was the {activity.name.lower()}?",
            "suggestions": ["Light", "Moderate", "Hard", "Give speed"],
        }
    intensity = str(raw_intensity or "moderate").casefold()
    if intensity not in {"light", "moderate", "hard"}:
        intensity = "moderate"
    speed_kmh = float(speed) if speed else None
    met, met_low, met_high = mets_for(activity, intensity, speed_kmh)
    calories = active_kcal(met, weight, duration)
    calories_low = active_kcal(met_low, weight, duration)
    calories_high = active_kcal(met_high, weight, duration)
    basis = (
        f"{met:g} MET · {weight:g} kg · {duration:g} min · resting energy excluded"
    )
    if speed_kmh:
        basis = f"{speed_kmh:g} km/h · " + basis
    return {
        "type": "workout",
        "action": "add",
        "name": activity.name,
        "durationMin": round(duration),
        "calories": round(calories),
        "calorieLow": round(calories_low),
        "calorieHigh": round(calories_high),
        "intensity": intensity,
        "met": met,
        "sourceLabel": COMPENDIUM_LABEL,
        "sourceId": activity.source_page,
        "confidence": "medium" if speed_kmh else "low",
        "basis": basis,
    }, None


def estimate_command(
    semantic_payload: dict[str, Any],
    profile: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Convert extracted facts into reviewable, evidence-backed operations."""
    profile = dict(profile or {})
    transcript = str(semantic_payload.get("transcript") or "").strip()
    intents = semantic_payload.get("intents") or []
    for intent in intents:
        if intent.get("type") == "weight" and float(intent.get("amount") or 0) > 0:
            profile["weightKg"] = float(intent["amount"])
    operations: list[dict[str, Any]] = []

    for intent in intents:
        kind = intent.get("type")
        action = intent.get("action") or ("set" if kind in {"steps", "sleep", "weight"} else "add")
        if kind == "meal":
            estimated_items: list[dict[str, Any]] = []
            for item in intent.get("items") or []:
                estimated, missing = _estimate_food_item(item, profile)
                if missing:
                    return _clarification(
                        transcript,
                        missing["question"],
                        missing["suggestions"],
                    )
                assert estimated is not None
                estimated_items.append(estimated)
            if not estimated_items:
                return _clarification(
                    transcript,
                    "What food did you have?",
                    ["Say the food and grams", "Say the food and pieces", "Say the food and bowl size"],
                )
            operations.append(
                {
                    "type": "meal",
                    "action": "add",
                    "slot": intent.get("slot") or "snack",
                    "description": intent.get("description") or transcript,
                    "items": estimated_items,
                }
            )
        elif kind == "workout":
            estimated_workout, missing = _estimate_workout(intent, profile)
            if missing:
                return _clarification(
                    transcript,
                    missing["question"],
                    missing["suggestions"],
                )
            assert estimated_workout is not None
            operations.append(estimated_workout)
        elif kind == "water":
            operations.append(
                {
                    "type": "water",
                    "action": action,
                    "amount": max(0, round(float(intent.get("amount") or 0))),
                }
            )
        elif kind == "steps":
            operations.append(
                {
                    "type": "steps",
                    "action": action,
                    "amount": max(0, round(float(intent.get("amount") or 0))),
                }
            )
        elif kind == "sleep":
            operations.append(
                {
                    "type": "sleep",
                    "action": "set",
                    "amount": max(0, float(intent.get("amount") or 0)),
                }
            )
        elif kind == "weight":
            operations.append(
                {
                    "type": "weight",
                    "action": "set",
                    "amount": max(0, float(intent.get("amount") or 0)),
                }
            )

    if not operations:
        return _clarification(
            transcript,
            "What would you like me to log?",
            ["A meal", "Water", "A workout", "Weight"],
        )
    count = len(operations)
    return {
        "transcript": transcript,
        "confirmation": (
            "1 evidence-backed update ready"
            if count == 1
            else f"{count} evidence-backed updates ready"
        ),
        "operations": operations,
    }
