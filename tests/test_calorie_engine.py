import unittest

from calorie_engine import estimate_command
from calorie_engine.exercise import active_kcal


def meal_payload(name: str, amount: float, unit: str, **extra):
    return {
        "transcript": f"{amount} {unit} {name}",
        "intents": [
            {
                "type": "meal",
                "action": "add",
                "slot": "lunch",
                "items": [{"name": name, "amount": amount, "unit": unit, **extra}],
            }
        ],
    }


class FoodEstimationTests(unittest.TestCase):
    def test_exact_grams_use_usda_per_100g(self):
        result = estimate_command(meal_payload("kidney beans", 100, "g"))
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["calories"], 127)
        self.assertEqual(item["protein"], 8.7)
        self.assertEqual(item["sourceId"], "FDC 175194")
        self.assertEqual(item["confidence"], "high")

    def test_amount_scales_linearly(self):
        result = estimate_command(meal_payload("kidney beans", 200, "g"))
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["calories"], 254)
        self.assertEqual(item["protein"], 17.3)

    def test_bowl_requires_calibration(self):
        result = estimate_command(meal_payload("rajma", 1, "bowl"))
        self.assertEqual(result["operations"], [])
        self.assertIn("usual bowl", result["clarification"]["question"])

    def test_rajma_bowl_exposes_recipe_range(self):
        result = estimate_command(
            meal_payload("rajma", 1, "bowl", preparation="curry"),
            {"bowlMl": 200},
        )
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["calories"], 231)
        self.assertLess(item["calorieLow"], item["calories"])
        self.assertGreater(item["calorieHigh"], item["calories"])
        self.assertEqual(item["confidence"], "low")
        self.assertIn("½–2 tsp oil", item["basis"])

    def test_two_roti_uses_standard_piece_with_range(self):
        result = estimate_command(meal_payload("roti", 2, "piece"))
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["calories"], 239)
        self.assertEqual(item["quantity"], "2 × 40 g standard piece")
        self.assertLess(item["calorieLow"], item["calories"])
        self.assertGreater(item["calorieHigh"], item["calories"])

    def test_unknown_food_never_gets_a_fabricated_fallback(self):
        result = estimate_command(meal_payload("mystery curry", 1, "bowl"), {"bowlMl": 200})
        self.assertEqual(result["operations"], [])
        self.assertIn("verified reference", result["clarification"]["question"])

    def test_user_supplied_label_resolves_unknown_food(self):
        payload = meal_payload("protein bar", 1, "piece", labelCalories=218, labelProtein=20)
        result = estimate_command(payload)
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["calories"], 218)
        self.assertEqual(item["protein"], 20)
        self.assertEqual(item["source"], "label")
        self.assertEqual(item["calorieLow"], 207)
        self.assertEqual(item["calorieHigh"], 229)

    def test_chickpeas_use_verified_usda_record(self):
        result = estimate_command(meal_payload("chana", 100, "g", preparation="boiled"))
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["calories"], 164)
        self.assertEqual(item["sourceId"], "FDC 173757")
        self.assertEqual(item["confidence"], "high")

    def test_chole_gets_home_curry_oil_range(self):
        result = estimate_command(meal_payload("chole", 100, "g"))
        item = result["operations"][0]["items"][0]
        self.assertIn("½–2 tsp oil", item["basis"])
        self.assertGreater(item["calories"], 164)

    def test_paneer_uses_queso_blanco_analog(self):
        result = estimate_command(meal_payload("paneer", 100, "g"))
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["calories"], 310)
        self.assertEqual(item["sourceId"], "FDC 172224")

    def test_glass_unit_uses_250_ml(self):
        result = estimate_command(meal_payload("milk", 1, "glass"))
        item = result["operations"][0]["items"][0]
        # 250 ml × 1.03 g/ml = 257.5 g of whole milk at 60 kcal/100 g.
        self.assertEqual(item["calories"], 154)
        self.assertIn("250 ml glass", item["quantity"])

    def test_slice_maps_to_piece(self):
        result = estimate_command(meal_payload("bread", 2, "slice"))
        item = result["operations"][0]["items"][0]
        self.assertEqual(item["quantity"], "2 × 28 g standard piece")

    def test_katori_maps_to_bowl(self):
        result = estimate_command(meal_payload("dal", 1, "katori"), {"bowlMl": 150})
        item = result["operations"][0]["items"][0]
        self.assertIn("150 ml bowl", item["quantity"])


class ExerciseEstimationTests(unittest.TestCase):
    def test_active_energy_formula_excludes_resting_met(self):
        self.assertAlmostEqual(active_kcal(4.8, 70, 30), 139.65, places=2)

    def test_walk_uses_weight_duration_and_compendium_met(self):
        payload = {
            "transcript": "30 minute hard walk",
            "intents": [
                {
                    "type": "workout",
                    "action": "add",
                    "name": "walking",
                    "durationMin": 30,
                    "intensity": "hard",
                }
            ],
        }
        result = estimate_command(payload, {"weightKg": 70})
        workout = result["operations"][0]
        self.assertEqual(workout["calories"], 140)
        self.assertEqual(workout["met"], 4.8)
        self.assertLess(workout["calorieLow"], workout["calories"])
        self.assertGreater(workout["calorieHigh"], workout["calories"])
        self.assertIn("resting energy excluded", workout["basis"])

    def test_workout_requires_current_weight(self):
        payload = {
            "transcript": "30 minute walk",
            "intents": [
                {
                    "type": "workout",
                    "action": "add",
                    "name": "walking",
                    "durationMin": 30,
                    "intensity": "moderate",
                }
            ],
        }
        result = estimate_command(payload)
        self.assertEqual(result["operations"], [])
        self.assertIn("body weight", result["clarification"]["question"])

    def test_workout_requires_intensity_or_speed(self):
        payload = {
            "transcript": "30 minute walk",
            "intents": [
                {
                    "type": "workout",
                    "action": "add",
                    "name": "walking",
                    "durationMin": 30,
                }
            ],
        }
        result = estimate_command(payload, {"weightKg": 70})
        self.assertEqual(result["operations"], [])
        self.assertIn("How hard", result["clarification"]["question"])

    def test_weight_answer_in_same_command_finishes_workout(self):
        payload = {
            "transcript": "30 minute walk, my weight is 70 kg",
            "intents": [
                {"type": "weight", "action": "set", "amount": 70},
                {
                    "type": "workout",
                    "action": "add",
                    "name": "walking",
                    "durationMin": 30,
                    "intensity": "moderate",
                },
            ],
        }
        result = estimate_command(payload)
        self.assertNotIn("clarification", result)
        self.assertEqual(result["operations"][0]["amount"], 70)
        self.assertEqual(result["operations"][1]["calories"], 103)


if __name__ == "__main__":
    unittest.main()
