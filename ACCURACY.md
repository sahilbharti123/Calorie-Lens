# Vigorly estimation method

Vigorly separates language understanding from measurement:

1. The operating system transcribes speech on the device — Apple's Speech
   framework on iOS, the platform recogniser on Android. No audio is uploaded,
   and typed input skips this step entirely.
2. A deterministic parser extracts facts from that text: food, amount, unit,
   preparation, activity, duration, speed, and intensity. It is ordinary
   pattern matching against a reviewed catalog, not a model.
3. A deterministic engine maps those facts to reviewed references.
4. The app shows the midpoint, plausible low–high range, assumptions,
   confidence, and source before anything is saved.

No model generates calories or macros. In the shipped default configuration no
model is involved at any point: remote AI parsing is opt-in behind
`EXPO_PUBLIC_ENABLE_AI_PARSING=1` and is off. When the parser cannot recognize
a food it asks for the label calories or the main parts with amounts, rather
than guessing.

Turning the flag on changes only which component reads the *language*. Even
then the model is prohibited from generating calories or macros — it returns
facts, and the same deterministic engine produces every number.

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

## Exercise guides, photos, and videos

Every built-in exercise ships with a researched how-to (setup, execution,
breathing, tempo, common mistakes, safety), real start/finish demonstration
photos, and a linked technique video:

- **Instructions** were written from reputable coaching sources (StrengthLog,
  NASM, ACE Fitness, Squat University, Renaissance Periodization, BarBend,
  Bret Contreras, and others; per-guide source lists are kept in the content
  pipeline), then graded by an adversarial reviewer against a strict rubric —
  equipment-specific setup, a concrete range-of-motion standard, correct
  breathing for the lift type, real prevalent mistakes with fixes — and
  revised until every guide passed at 10/10 (three review rounds).
- **Photos** are from the free-exercise-db project
  (https://github.com/yuhonas/free-exercise-db), released into the public
  domain (Unlicense), bundled in `mobile/assets/exercises/`.
- **Videos** are links to established channels (Renaissance Periodization,
  Squat University, ScottHermanFitness, BarBend, Jeff Nippard, PureGym, GCN,
  Planet Fitness, and similar). Every URL was verified against YouTube's
  oEmbed metadata — exact title and channel — and re-verified by the
  reviewer; wrong-variation videos (e.g. a barbell demo for a dumbbell
  movement) were rejected and replaced.
- **Movement-path figures** (the stylized skeletons) are a secondary aid and
  went through eleven rounds of pixel-measured visual review: 60 of 63
  templates at 10/10, with the remaining three (shrug, russian twist, walk)
  documented as inherent limits of a stylized side view.

## Confidence labels

- **High:** an exact gram weight with a reviewed food match and no recipe
  assumption.
- **Medium:** a standard piece, direct liquid volume, or more specific workout
  such as a known speed.
- **Low:** a bowl, a home recipe with unknown oil, or an activity whose effort
  varies substantially.

## Coach insights

The Coach tab shows the day's focus, the current plan, and a short list of
insights. All of it is computed on the device — the insights by
`mobile/src/lib/insights.ts`, the day's focus by
`mobile/src/lib/personalization.ts` — as arithmetic and fixed rules over the
user's own entries and stated profile. Nothing is sent anywhere, no model is
involved, and no service is called.

Each insight is a mean or a difference over a stated window, and it names the
number it came from so the reader can check it:

- **Protein and calorie adherence** — the average across the last seven days
  that actually have a logged meal, compared with the plan target. Days with no
  meals are excluded, because averaging them in would understate intake.
  Calorie drift is only flagged past 12% of target.
- **Training volume** — logged workout minutes over the last seven days against
  the planned weekly minutes, with the session count.
- **Hydration** — the average over days where water was actually tracked, shown
  only when it falls below three quarters of target.
- **Estimate quality** — the share of the last seven days' entries that carry a
  low confidence label, surfaced when it exceeds 40%. This is the app checking
  its own precision back to the user.
- **Weight trend** — the change across up to the last eight weigh-ins, with the
  reminder that only direction over weeks is signal.
- **Logging streak** — consecutive days with an entry.

These are descriptive statements about what was logged, not predictions and not
medical advice. They inherit every limitation of the underlying estimates: an
average of wide estimates is still a wide estimate, which is why the estimate
quality insight exists. The conversational AI coach is not part of this
release.

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
