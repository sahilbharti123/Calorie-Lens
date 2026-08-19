import type { MealItem } from '@/src/types';

/** Last-write-wins merge for correctable scalar metrics, including zero. */
export function metricByNewest<T>(
  localValue: T,
  localUpdatedAt: string | undefined,
  remoteValue: T,
  remoteUpdatedAt: string | undefined,
) {
  if (localUpdatedAt && remoteUpdatedAt) {
    return localUpdatedAt >= remoteUpdatedAt
      ? { value: localValue, updatedAt: localUpdatedAt }
      : { value: remoteValue, updatedAt: remoteUpdatedAt };
  }
  if (remoteUpdatedAt) return { value: remoteValue, updatedAt: remoteUpdatedAt };
  return { value: localValue, updatedAt: localUpdatedAt };
}

/**
 * Resolves a meal's single canonical day after independent devices edit or
 * move the same row. `dayKey` is written with corrections; legacy rows fall
 * back to the container date.
 */
export function resolveMealLocations(mealsByDate: Record<string, MealItem[]>) {
  const locations = new Map<string, { date: string; meal: MealItem }>();
  for (const [containerDate, meals] of Object.entries(mealsByDate)) {
    for (const meal of meals) {
      const previous = locations.get(meal.id);
      if (
        !previous
        || (meal.updatedAt ?? meal.loggedAt) >= (previous.meal.updatedAt ?? previous.meal.loggedAt)
      ) {
        locations.set(meal.id, { date: meal.dayKey ?? containerDate, meal });
      }
    }
  }
  return [...locations.values()];
}
