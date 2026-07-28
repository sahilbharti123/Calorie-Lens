"""2024 Adult Compendium based active-energy calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActivityReference:
    name: str
    aliases: tuple[str, ...]
    light: float
    moderate: float
    hard: float
    source_page: str


ACTIVITIES: tuple[ActivityReference, ...] = (
    ActivityReference("Walking", ("walk", "walking"), 2.8, 3.8, 4.8, "walking"),
    ActivityReference("Running", ("run", "running", "jog", "jogging"), 6.5, 8.5, 11.0, "running"),
    ActivityReference("Cycling", ("cycle", "cycling", "bike", "biking"), 4.3, 7.0, 9.0, "bicycling"),
    ActivityReference("Strength training", ("strength", "weights", "weight training", "gym", "lifting"), 3.5, 5.0, 6.0, "conditioning-exercise"),
    ActivityReference("Circuit training", ("circuit", "kettlebell"), 3.5, 5.0, 7.5, "conditioning-exercise"),
    ActivityReference("HIIT", ("hiit", "high intensity interval"), 7.0, 9.0, 11.0, "conditioning-exercise"),
    ActivityReference("Yoga", ("yoga", "vinyasa", "hatha"), 2.3, 2.7, 4.0, "conditioning-exercise"),
    ActivityReference("Calisthenics", ("calisthenics", "bodyweight"), 2.8, 3.8, 7.5, "conditioning-exercise"),
)


def find_activity(name: str) -> Optional[ActivityReference]:
    normalized = name.casefold()
    return next(
        (
            activity
            for activity in ACTIVITIES
            if any(alias in normalized for alias in activity.aliases)
        ),
        None,
    )


def mets_for(
    activity: ActivityReference,
    intensity: str,
    speed_kmh: Optional[float] = None,
) -> tuple[float, float, float]:
    normalized = intensity if intensity in {"light", "moderate", "hard"} else "moderate"
    if activity.name == "Walking" and speed_kmh:
        if speed_kmh < 4:
            return 2.8, 2.3, 3.0
        if speed_kmh < 5.6:
            return 3.8, 3.5, 4.3
        if speed_kmh < 6.4:
            return 4.8, 4.3, 5.5
        return 5.5, 4.8, 6.8
    if activity.name == "Running" and speed_kmh:
        if speed_kmh < 7.5:
            return 6.5, 6.0, 7.5
        if speed_kmh < 9:
            return 8.5, 7.5, 9.3
        if speed_kmh < 10.5:
            return 9.3, 8.5, 10.5
        if speed_kmh < 12:
            return 11.0, 10.0, 11.8
        return 12.0, 11.0, 14.8

    selected = getattr(activity, normalized)
    ordered = sorted((activity.light, activity.moderate, activity.hard))
    index = ordered.index(selected)
    low = ordered[index - 1] if index > 0 else max(1.1, selected * 0.9)
    high = ordered[index + 1] if index < len(ordered) - 1 else selected * 1.1
    return selected, low, high


def active_kcal(met: float, weight_kg: float, minutes: float) -> float:
    """Net active energy, subtracting the 1 MET resting component."""
    return max(0.0, met - 1.0) * 3.5 * weight_kg / 200 * minutes
