import type { DayLog, MealSlot } from '@/src/types';

export const slotLabels: Record<MealSlot, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  snack: 'Evening snack',
  dinner: 'Dinner',
};

export function dayTotals(day: DayLog) {
  return day.meals.reduce(
    (totals, meal) => ({
      calories: totals.calories + meal.calories,
      protein: totals.protein + meal.protein,
      carbs: totals.carbs + meal.carbs,
      fat: totals.fat + meal.fat,
    }),
    { calories: 0, protein: 0, carbs: 0, fat: 0 },
  );
}

export function workoutTotals(day: DayLog) {
  return day.workouts.reduce(
    (totals, workout) => ({
      minutes: totals.minutes + workout.durationMin,
      calories: totals.calories + workout.calories,
    }),
    { minutes: 0, calories: 0 },
  );
}
