# Calorie Lens estimation method

Calorie Lens separates language understanding from measurement:

1. Gemini transcribes speech and extracts facts such as food, amount, unit,
   preparation, activity, duration, speed, and intensity.
2. A deterministic engine maps those facts to reviewed references.
3. The app shows the midpoint, plausible low–high range, assumptions,
   confidence, and source before anything is saved.

The model is explicitly prohibited from generating calories or macros.

## Food estimates

The bundled catalog uses per-100-g records from
[USDA FoodData Central](https://fdc.nal.usda.gov/). Every catalog item stores
its FoodData Central ID. USDA publishes this data under CC0.

Examples:

| Food | Reference | Energy |
| --- | --- | ---: |
| Cooked red kidney beans | FDC 175194 | 127 kcal/100 g |
| Cooked lentils | FDC 172421 | 116 kcal/100 g |
| Cooked white rice | FDC 168878 | 130 kcal/100 g |
| Whole-wheat chapati/roti | FDC 174075 | 299 kcal/100 g |
| Roasted chicken breast | FDC 171477 | 165 kcal/100 g |
| Boiled egg | FDC 173424 | 155 kcal/100 g |
| Idli | FDC 2708346 | 128 kcal/100 g |
| Plain dosa | FDC 2708347 | 210 kcal/100 g |

Grams receive the narrowest portion range. Pieces use a documented standard
piece weight and a size range. Volume measures use the user's calibrated bowl
or the app's 200 ml cup plus a food-density range.

Home rajma and dal are not equivalent to plain boiled pulses. Unless the user
states the oil in their portion, the review includes a 0.5–2 tsp oil range.
Unknown foods do not receive a generic 300-kcal fallback; the app asks for a
label or ingredient description. A user-supplied package or restaurant value
is stored as a label-sourced estimate with a small rounding range instead of
being overwritten by the model.

## Exercise estimates

Activities map to the
[2024 Adult Compendium of Physical Activities](https://pacompendium.com/adult-compendium/).
The app calculates net active energy:

```text
active kcal = (MET − 1) × 3.5 × body weight kg ÷ 200 × minutes
```

Subtracting one MET avoids counting resting energy as exercise energy. The
range uses neighboring plausible MET values for the stated activity and
intensity. Speed narrows walking and running estimates.

Wearable active-energy values can still be synced through Apple Health or
Health Connect, but they remain estimates. They are preserved as device data
rather than presented as laboratory measurement.

## Confidence labels

- **High:** an exact gram weight with a reviewed food match and no recipe
  assumption.
- **Medium:** a standard piece, direct liquid volume, or more specific workout
  such as a known speed.
- **Low:** a bowl, a home recipe with unknown oil, or an activity whose effort
  varies substantially.

## Benchmarks and tests

Reviewed benchmark cases live in
[`tests/accuracy_benchmarks.json`](tests/accuracy_benchmarks.json). Run:

```bash
python -m unittest discover -s tests -v
```

The suite verifies linear gram scaling, bowl calibration, recipe uncertainty,
standard roti sizing, refusal to fabricate unknown foods, MET calculations,
and required workout body weight.

## Limits

This is an estimation system, not a calorimeter. Recipe ingredients, absorbed
oil, cooked yield, restaurant preparation, portion reporting, and individual
exercise efficiency can all change the true number. A range is more honest
than a precise-looking single value.

ICMR-NIN's Indian Food Composition Tables were reviewed as an Indian-food
reference, but their publication restricts electronic product reproduction
without prior permission. Those tables are not copied into this repository.
