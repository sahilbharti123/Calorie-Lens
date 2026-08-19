import type { AppData } from '@/src/types';

function record(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function finite(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value);
}

function stringArray(value: unknown) {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

/** Rejects damaged or unrelated JSON before it can replace live app state. */
export function isBackupPayload(value: unknown): value is AppData {
  if (!record(value) || !record(value.goals) || !record(value.profile)) return false;
  if (!record(value.plan) || !record(value.estimation) || !record(value.days)) return false;
  if (!record(value.coachMemory) || !record(value.training)) return false;

  const goals = value.goals;
  if (![
    'calories', 'protein', 'carbs', 'fat', 'waterMl', 'steps',
    'weeklyWorkoutMinutes', 'strengthDays',
  ].every((key) => finite(goals[key]))) return false;

  if (!stringArray(value.profile.allergies) || !stringArray(value.profile.injuries)) return false;
  if (!Array.isArray(value.weights) || !value.weights.every((point) => (
    record(point) && typeof point.date === 'string' && finite(point.kg)
  ))) return false;

  for (const [date, day] of Object.entries(value.days)) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !record(day)) return false;
    if (!Array.isArray(day.meals) || !Array.isArray(day.workouts)) return false;
    if (![day.waterMl, day.steps, day.activeCalories, day.sleepHours].every(finite)) return false;
  }

  for (const key of [
    'coachMessages', 'deletedMealIds', 'deletedWorkoutIds', 'deletedSavedMealIds',
    'savedMeals', 'learnedFoods', 'deletedLearnedFoodIds',
  ]) {
    if (!Array.isArray(value[key])) return false;
  }

  const training = value.training;
  if (![
    'routines', 'sessions', 'customExercises', 'deletedRoutineIds', 'deletedSessionIds',
  ].every((key) => Array.isArray(training[key]))) return false;
  if (training.deletedCustomExerciseIds != null && !Array.isArray(training.deletedCustomExerciseIds)) return false;
  if (!(training.activeSession == null || record(training.activeSession))) return false;
  if (!finite(training.defaultRestSec) || typeof training.rpeEnabled !== 'boolean') return false;
  return true;
}
