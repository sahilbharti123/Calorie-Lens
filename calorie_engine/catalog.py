"""Small, reviewed reference catalog used when the API is offline.

Nutrients are expressed per 100 g. FoodData Central records are CC0 and each
entry keeps its FDC identifier so a user can inspect the exact source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FoodReference:
    key: str
    name: str
    aliases: tuple[str, ...]
    calories: float
    protein: float
    carbs: float
    fat: float
    fdc_id: int
    source_description: str
    piece_grams: Optional[float] = None
    piece_variance: float = 0.12
    density_g_ml: Optional[float] = None
    density_variance: float = 0.12


FOODS: tuple[FoodReference, ...] = (
    FoodReference(
        "kidney_beans",
        "Cooked kidney beans",
        ("rajma", "kidney bean", "red bean"),
        127,
        8.67,
        22.8,
        0.5,
        175194,
        "Beans, kidney, red, mature seeds, cooked, boiled, without salt",
        density_g_ml=0.75,
        density_variance=0.18,
    ),
    FoodReference(
        "lentils",
        "Cooked lentils",
        ("dal", "dhal", "lentil"),
        116,
        9.02,
        20.1,
        0.38,
        172421,
        "Lentils, mature seeds, cooked, boiled, without salt",
        density_g_ml=0.75,
        density_variance=0.18,
    ),
    FoodReference(
        "white_rice",
        "Cooked white rice",
        ("white rice", "plain rice", "cooked rice", "rice"),
        130,
        2.69,
        28.2,
        0.28,
        168878,
        "Rice, white, long-grain, regular, enriched, cooked",
        density_g_ml=0.79,
        density_variance=0.12,
    ),
    FoodReference(
        "roti",
        "Whole-wheat roti",
        ("chapati", "chappati", "phulka", "roti"),
        299,
        7.85,
        46.1,
        9.2,
        174075,
        "Bread, chapati or roti, whole wheat, commercially prepared, frozen",
        piece_grams=40,
        piece_variance=0.2,
    ),
    FoodReference(
        "chicken_breast",
        "Roasted chicken breast",
        ("chicken breast", "grilled chicken", "roasted chicken"),
        165,
        31,
        0,
        3.57,
        171477,
        "Chicken, broilers or fryers, breast, meat only, cooked, roasted",
    ),
    FoodReference(
        "boiled_egg",
        "Boiled egg",
        ("hard boiled egg", "boiled egg", "egg"),
        155,
        12.6,
        1.12,
        10.6,
        173424,
        "Egg, whole, cooked, hard-boiled",
        piece_grams=50,
        piece_variance=0.12,
    ),
    FoodReference(
        "whole_milk",
        "Whole milk",
        ("whole milk", "full fat milk", "milk"),
        60,
        3.27,
        4.63,
        3.2,
        746782,
        "Milk, whole, 3.25% milkfat, with added vitamin D",
        density_g_ml=1.03,
        density_variance=0.03,
    ),
    FoodReference(
        "plain_yogurt",
        "Plain whole-milk yogurt",
        ("plain yogurt", "yogurt", "curd", "dahi"),
        61,
        3.47,
        4.66,
        3.25,
        171284,
        "Yogurt, plain, whole milk",
        density_g_ml=1.03,
        density_variance=0.06,
    ),
    FoodReference(
        "banana",
        "Banana",
        ("banana", "kela"),
        89,
        1.09,
        22.8,
        0.33,
        173944,
        "Bananas, raw",
        piece_grams=118,
        piece_variance=0.18,
    ),
    FoodReference(
        "whole_wheat_bread",
        "Whole-wheat bread",
        ("whole wheat bread", "brown bread", "bread", "toast"),
        252,
        12.4,
        42.7,
        3.5,
        172688,
        "Bread, whole-wheat, commercially prepared",
        piece_grams=28,
        piece_variance=0.18,
    ),
    FoodReference(
        "peanut_butter",
        "Smooth peanut butter",
        ("peanut butter",),
        598,
        22.2,
        22.3,
        51.4,
        172470,
        "Peanut butter, smooth style, without salt",
        density_g_ml=1.07,
        density_variance=0.08,
    ),
    FoodReference(
        "idli",
        "Idli",
        ("idli", "idlis"),
        128,
        6.36,
        24.98,
        0.35,
        2708346,
        "Idli",
        piece_grams=50,
        piece_variance=0.22,
    ),
    FoodReference(
        "plain_dosa",
        "Plain dosa",
        ("plain dosa", "dosa"),
        210,
        5.7,
        37.04,
        4.05,
        2708347,
        "Dosa, plain",
        piece_grams=100,
        piece_variance=0.3,
    ),
)


def find_food(name: str) -> Optional[FoodReference]:
    normalized = name.casefold().strip()
    candidates = [
        food
        for food in FOODS
        if food.key == normalized
        or food.name.casefold() == normalized
        or any(alias in normalized or normalized in alias for alias in food.aliases)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda food: max(
            [len(alias) for alias in food.aliases if alias in normalized or normalized in alias]
            or [0]
        ),
    )
