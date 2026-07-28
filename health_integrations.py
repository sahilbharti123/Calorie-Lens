"""Apple Health snapshot import helpers.

The Streamlit app cannot call HealthKit directly because HealthKit is an
Apple-platform framework. This module imports the XML snapshot that the Health
app exports and merges Apple Watch/iPhone measurements into the tracker.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple
from xml.etree import ElementTree as ET


STEP_TYPE = "HKQuantityTypeIdentifierStepCount"
WATER_TYPE = "HKQuantityTypeIdentifierDietaryWater"
WEIGHT_TYPE = "HKQuantityTypeIdentifierBodyMass"
SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _day(value: str) -> str:
    return value[:10] if value else ""


def _date_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        for date_format in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, date_format)
            except ValueError:
                continue
        return None


def _duration_hours(start: str, end: str) -> float:
    start_at = _date_time(start)
    end_at = _date_time(end)
    if not start_at or not end_at or end_at <= start_at:
        return 0.0
    return (end_at - start_at).total_seconds() / 3600


def _water_ml(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"l", "liter", "litre"}:
        return value * 1000
    if normalized in {"fl_oz_us", "floz", "oz"}:
        return value * 29.5735
    if normalized in {"cup_us", "cup"}:
        return value * 236.588
    return value


def _weight_kg(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"lb", "lbs"}:
        return value * 0.45359237
    if normalized in {"g", "gram", "grams"}:
        return value / 1000
    return value


def _workout_name(identifier: str) -> str:
    raw = identifier.replace("HKWorkoutActivityType", "")
    if not raw:
        return "Apple Watch workout"
    words = []
    token = ""
    for char in raw:
        if char.isupper() and token:
            words.append(token)
            token = char
        else:
            token += char
    if token:
        words.append(token)
    return " ".join(words).replace("High Intensity Interval Training", "HIIT")


def _empty_day() -> Dict[str, Any]:
    return {
        "steps": 0,
        "_step_sources": {},
        "water_ml": 0,
        "sleep_hours": 0.0,
        "weight_kg": None,
        "exercises": [],
        "records": 0,
    }


def _xml_stream(file_bytes: bytes, filename: str) -> Tuple[io.BytesIO, str]:
    if filename.lower().endswith(".zip") or file_bytes[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith("export.xml") and "__macosx" not in name.lower()
            ]
            if not candidates:
                raise ValueError("The ZIP does not contain an Apple Health export.xml file.")
            selected = min(candidates, key=len)
            return io.BytesIO(archive.read(selected)), selected
    return io.BytesIO(file_bytes), filename


def parse_apple_health_export(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Parse a Health export ZIP/XML into tracker-friendly daily summaries."""

    stream, source_name = _xml_stream(file_bytes, filename)
    days: Dict[str, Dict[str, Any]] = {}
    record_count = 0
    workout_count = 0

    for _, element in ET.iterparse(stream, events=("end",)):
        if element.tag == "Record":
            health_type = element.attrib.get("type", "")
            day_key = _day(element.attrib.get("startDate", ""))
            if not day_key:
                element.clear()
                continue

            if health_type not in {STEP_TYPE, WATER_TYPE, WEIGHT_TYPE, SLEEP_TYPE}:
                element.clear()
                continue

            day_data = days.setdefault(day_key, _empty_day())
            value = _number(element.attrib.get("value"))
            unit = element.attrib.get("unit", "")

            if health_type == STEP_TYPE:
                source = element.attrib.get("sourceName", "Unknown source")
                source_totals = day_data["_step_sources"]
                source_totals[source] = source_totals.get(source, 0.0) + value
            elif health_type == WATER_TYPE:
                day_data["water_ml"] += int(round(_water_ml(value, unit)))
            elif health_type == WEIGHT_TYPE:
                day_data["weight_kg"] = round(_weight_kg(value, unit), 2)
            elif health_type == SLEEP_TYPE:
                sleep_value = element.attrib.get("value", "").lower()
                if "asleep" in sleep_value and "inbed" not in sleep_value:
                    day_data["sleep_hours"] += _duration_hours(
                        element.attrib.get("startDate", ""),
                        element.attrib.get("endDate", ""),
                    )

            day_data["records"] += 1
            record_count += 1

        elif element.tag == "Workout":
            day_key = _day(element.attrib.get("startDate", ""))
            if not day_key:
                element.clear()
                continue

            duration = _number(element.attrib.get("duration"))
            duration_unit = element.attrib.get("durationUnit", "min").lower()
            if duration_unit.startswith("h"):
                duration *= 60
            elif duration_unit.startswith("s"):
                duration /= 60

            energy = _number(element.attrib.get("totalEnergyBurned"))
            energy_unit = element.attrib.get("totalEnergyBurnedUnit", "kcal").lower()
            if energy_unit == "kj":
                energy /= 4.184

            workout_type = element.attrib.get("workoutActivityType", "")
            start_at = element.attrib.get("startDate", "")
            source_id = hashlib.sha256(
                f"{workout_type}|{start_at}|{duration:.2f}".encode("utf-8")
            ).hexdigest()[:20]
            day_data = days.setdefault(day_key, _empty_day())
            day_data["exercises"].append(
                {
                    "name": _workout_name(workout_type),
                    "duration_min": int(round(duration)),
                    "calories_burned": int(round(energy)),
                    "intensity": "Moderate",
                    "notes": f"Imported from {element.attrib.get('sourceName', 'Apple Health')}",
                    "logged_at": start_at[11:16] if len(start_at) >= 16 else "--",
                    "source": "Apple Health",
                    "source_id": source_id,
                }
            )
            day_data["records"] += 1
            record_count += 1
            workout_count += 1

        element.clear()

    for day_data in days.values():
        source_step_totals = day_data.pop("_step_sources", {})
        if source_step_totals:
            # Raw Health exports can contain overlapping phone and Watch samples.
            # The largest single-source total is a safer snapshot than summing
            # every source and double-counting the same walk.
            day_data["steps"] = int(round(max(source_step_totals.values())))
        day_data["sleep_hours"] = min(round(day_data["sleep_hours"], 2), 24.0)

    if not days:
        raise ValueError(
            "No supported steps, water, sleep, weight, or workout records were found."
        )

    return {
        "days": days,
        "source_name": source_name,
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "record_count": record_count,
        "workout_count": workout_count,
    }


def merge_apple_health_snapshot(
    user: Dict[str, Any],
    snapshot: Dict[str, Any],
    imported_filename: str,
) -> Dict[str, int]:
    """Merge a parsed snapshot, avoiding duplicate imports and workouts."""

    file_hash = snapshot["file_hash"]
    imports = user.setdefault("health_imports", [])
    if any(entry.get("file_hash") == file_hash for entry in imports):
        raise ValueError("This Apple Health export has already been imported.")

    changed_days = 0
    added_workouts = 0
    for day_key, incoming in snapshot["days"].items():
        day_log = user["days"].setdefault(day_key, {})
        day_log.setdefault("water_ml", 0)
        day_log.setdefault("sleep_hours", 0.0)
        day_log.setdefault("steps", 0)
        day_log.setdefault("weight_kg", None)
        day_log.setdefault("exercises", [])
        day_log.setdefault(
            "meals",
            {
                "Breakfast": [],
                "Lunch": [],
                "Evening Snack": [],
                "Dinner": [],
            },
        )
        day_log.setdefault("notes", "")
        day_log.setdefault("mood", "Steady")
        day_log.setdefault("energy", "Balanced")
        day_log.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        # Snapshot values represent complete Apple totals. max() avoids doubling
        # manually logged values when overlapping exports are imported.
        day_log["steps"] = max(int(day_log["steps"]), int(incoming["steps"]))
        day_log["water_ml"] = max(int(day_log["water_ml"]), int(incoming["water_ml"]))
        day_log["sleep_hours"] = max(
            float(day_log["sleep_hours"]), float(incoming["sleep_hours"])
        )
        if incoming["weight_kg"] is not None:
            day_log["weight_kg"] = float(incoming["weight_kg"])

        existing_ids = {
            exercise.get("source_id")
            for exercise in day_log["exercises"]
            if exercise.get("source_id")
        }
        for exercise in incoming["exercises"]:
            if exercise["source_id"] not in existing_ids:
                day_log["exercises"].append(exercise)
                existing_ids.add(exercise["source_id"])
                added_workouts += 1

        day_log["updated_at"] = datetime.now().isoformat(timespec="seconds")
        changed_days += 1

    imports.append(
        {
            "file_hash": file_hash,
            "filename": imported_filename,
            "source_name": snapshot["source_name"],
            "record_count": snapshot["record_count"],
            "days": len(snapshot["days"]),
            "imported_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return {"changed_days": changed_days, "added_workouts": added_workouts}
