import type { AppData, LogOperation, MealItem, MealSlot, SavedMeal } from '../types';

type MealLogOperation = Extract<LogOperation, { type: 'meal' }>;

export type MealGroup = {
  key: string;
  date: string;
  slot: MealSlot;
  items: MealItem[];
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  latestAt: string;
};

/**
 * Builds recent meal-sized groups from the ledger without adding a new storage
 * format. A slot on a day is the stable unit users recognise as breakfast,
 * lunch, snack, or dinner. Identical groups are de-duplicated so the shortcut
 * stays useful instead of becoming a list of the same lunch.
 */
export function recentMealGroups(data: Pick<AppData, 'days'>, limit = 3): MealGroup[] {
  const groups = Object.entries(data.days)
    .flatMap(([date, day]) => {
      const slots = new Map<MealSlot, MealItem[]>();
      for (const meal of day.meals) {
        slots.set(meal.slot, [...(slots.get(meal.slot) ?? []), meal]);
      }
      return [...slots.entries()].map(([slot, items]) => toMealGroup(date, slot, items));
    })
    .filter((group) => group.items.length > 0)
    .sort((left, right) => right.latestAt.localeCompare(left.latestAt));

  const seen = new Set<string>();
  const unique: MealGroup[] = [];
  for (const group of groups) {
    const signature = mealSignature(group.items);
    if (seen.has(signature)) continue;
    seen.add(signature);
    unique.push(group);
    if (unique.length >= limit) break;
  }
  return unique;
}

export function savedMealToOperation(meal: SavedMeal, slot = meal.slot): MealLogOperation {
  return {
    type: 'meal',
    action: 'add',
    slot,
    description: meal.name,
    items: meal.items.map((item) => ({ ...item, assumptions: [...(item.assumptions ?? [])] })),
  };
}

export function mealGroupToOperation(group: MealGroup, slot = group.slot): MealLogOperation {
  return {
    type: 'meal',
    action: 'add',
    slot,
    description: group.name,
    items: group.items.map(({ id: _id, loggedAt: _loggedAt, slot: _slot, ...item }) => ({
      ...item,
      assumptions: [...(item.assumptions ?? [])],
    })),
  };
}

export function savedMealFromGroup(group: MealGroup, id: string, now = new Date().toISOString()): SavedMeal {
  return {
    id,
    name: group.name,
    slot: group.slot,
    items: mealGroupToOperation(group).items,
    createdAt: now,
    updatedAt: now,
  };
}

/** Merge cloud/device shortcuts without resurrecting deletions or older edits. */
export function mergeSavedMeals(
  local: SavedMeal[],
  remote: SavedMeal[],
  deletedIds: string[],
  limit = 50,
) {
  const deleted = new Set(deletedIds);
  const merged = new Map<string, SavedMeal>();
  for (const meal of [...remote, ...local]) {
    if (deleted.has(meal.id)) continue;
    const previous = merged.get(meal.id);
    if (!previous || meal.updatedAt >= previous.updatedAt) merged.set(meal.id, meal);
  }
  return [...merged.values()]
    .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
    .slice(0, limit);
}

function toMealGroup(date: string, slot: MealSlot, items: MealItem[]): MealGroup {
  const latestAt = items.reduce(
    (latest, item) => item.loggedAt > latest ? item.loggedAt : latest,
    `${date}T00:00:00.000Z`,
  );
  return {
    key: `${date}:${slot}`,
    date,
    slot,
    items,
    name: mealGroupName(items),
    calories: sum(items, 'calories'),
    protein: sum(items, 'protein'),
    carbs: sum(items, 'carbs'),
    fat: sum(items, 'fat'),
    latestAt,
  };
}

function mealGroupName(items: MealItem[]) {
  if (items.length === 1) return items[0].name;
  if (items.length === 2) return `${items[0].name} & ${items[1].name}`;
  return `${items[0].name} + ${items.length - 1} more`;
}

function mealSignature(items: MealItem[]) {
  return items
    .map((item) => `${item.name.trim().toLowerCase()}|${item.quantity.trim().toLowerCase()}`)
    .sort()
    .join('::');
}

function sum(items: MealItem[], key: 'calories' | 'protein' | 'carbs' | 'fat') {
  return Math.round(items.reduce((total, item) => total + item[key], 0) * 10) / 10;
}
