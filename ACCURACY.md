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
| Boiled chickpeas (chana/chole) | FDC 173757 | 164 kcal/100 g |
| Cooked white rice | FDC 168878 | 130 kcal/100 g |
| Cooked brown rice | FDC 169704 | 123 kcal/100 g |
| Rolled oats, dry | FDC 173904 | 379 kcal/100 g |
| Whole-wheat chapati/roti | FDC 174075 | 299 kcal/100 g |
| Whole-wheat paratha | FDC 174076 | 326 kcal/100 g |
| Roasted chicken breast | FDC 171477 | 165 kcal/100 g |
| Boiled egg | FDC 173424 | 155 kcal/100 g |
| Egg white | FDC 172183 | 52 kcal/100 g |
| Boiled potato | FDC 170438 | 87 kcal/100 g |
| Idli | FDC 2708346 | 128 kcal/100 g |
| Plain dosa | FDC 2708347 | 210 kcal/100 g |
| Samosa (FNDDS recipe) | FDC 2344214 | 309 kcal/100 g |
| Biryani with meat (FNDDS recipe) | FDC 2341916 | 144 kcal/100 g |
| Chicken curry (FNDDS recipe) | FDC 2341861 | 82 kcal/100 g |

Paneer is mapped to USDA "Cheese, white, queso blanco" (FDC 172224,
310 kcal/100 g), the closest published analog to fresh paneer; the FNDDS
"Cheese, paneer" survey record was reviewed and rejected because its
carbohydrate value (22.5 g/100 g) is inconsistent with fresh acid-set
cheese composition. The catalog was last re-verified against the published
FoodData Central values in July 2026.

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

The intensity METs were re-verified against the published 2024 Adult
Compendium tables in July 2026: resistance training 3.5 / 5.0 / 6.0 (codes
02054 / 02052 / 02050), circuit training 3.5 / 5.0 / 7.5 (02034 / 02035 /
02040), HIIT 7.0–11.0 (02210 / 02214), yoga 2.3 / 2.7 / 4.0 (Hatha /
Vinyasa / Power), calisthenics 2.8 / 3.8 / 7.5 (02024 / 02022 / 02020), and
the walking speed bands match codes 17170–17220.

## Strength sessions (Train tab)

A finished strength workout logs its active energy with the same net-MET
method. The session MET is the set-weighted average of the involved
exercises' Compendium-family MET values, the shown range widens toward the
light (3.5) and vigorous ends, and the basis line always states the MET,
body weight, and duration used. Set-by-set energy is *not* claimed — rest
periods dominate gym sessions, which is exactly what the duration × MET
method represents.

Estimated 1RM in exercise records uses the Epley formula
(weight × (1 + reps ÷ 30)) and is labeled as an estimate.

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
