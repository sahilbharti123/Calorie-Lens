import unittest

from health_integrations import (
    merge_apple_health_snapshot,
    parse_apple_health_export,
)


SAMPLE_EXPORT = b"""<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Sahil's Apple Watch"
    unit="count" creationDate="2026-07-28 08:00:00 +0530"
    startDate="2026-07-28 07:00:00 +0530" endDate="2026-07-28 08:00:00 +0530"
    value="1250"/>
  <Record type="HKQuantityTypeIdentifierDietaryWater" sourceName="Health"
    unit="mL" creationDate="2026-07-28 09:00:00 +0530"
    startDate="2026-07-28 09:00:00 +0530" endDate="2026-07-28 09:00:00 +0530"
    value="500"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"
    value="HKCategoryValueSleepAnalysisAsleepCore"
    startDate="2026-07-28 00:00:00 +0530" endDate="2026-07-28 06:30:00 +0530"/>
  <Workout workoutActivityType="HKWorkoutActivityTypeTraditionalStrengthTraining"
    duration="45" durationUnit="min" totalEnergyBurned="280"
    totalEnergyBurnedUnit="kcal" sourceName="Sahil's Apple Watch"
    startDate="2026-07-28 18:00:00 +0530" endDate="2026-07-28 18:45:00 +0530"/>
</HealthData>"""


class AppleHealthImportTests(unittest.TestCase):
    def test_parses_daily_metrics_and_workout(self):
        snapshot = parse_apple_health_export(SAMPLE_EXPORT, "export.xml")
        day = snapshot["days"]["2026-07-28"]

        self.assertEqual(day["steps"], 1250)
        self.assertEqual(day["water_ml"], 500)
        self.assertEqual(day["sleep_hours"], 6.5)
        self.assertEqual(day["exercises"][0]["duration_min"], 45)
        self.assertEqual(day["exercises"][0]["calories_burned"], 280)

    def test_merge_is_idempotent_by_file_hash(self):
        snapshot = parse_apple_health_export(SAMPLE_EXPORT, "export.xml")
        user = {"days": {}, "health_imports": []}

        result = merge_apple_health_snapshot(user, snapshot, "export.xml")
        self.assertEqual(result["changed_days"], 1)
        self.assertEqual(result["added_workouts"], 1)

        with self.assertRaisesRegex(ValueError, "already been imported"):
            merge_apple_health_snapshot(user, snapshot, "export.xml")


if __name__ == "__main__":
    unittest.main()
