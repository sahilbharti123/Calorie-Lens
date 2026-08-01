import assert from 'node:assert/strict';
import test from 'node:test';

import {
  mealGroupToOperation,
  mergeSavedMeals,
  recentMealGroups,
  savedMealFromGroup,
  savedMealToOperation,
} from '../src/lib/meals.ts';

function item(id: string, name: string, quantity: string, slot: 'breakfast' | 'lunch', loggedAt: string) {
  return {
    id,
    name,
    quantity,
    calories: 100,
    protein: 10,
    carbs: 12,
    fat: 2,
    slot,
    loggedAt,
    source: 'usda' as const,
    assumptions: ['reviewed'],
  };
}

test('recent meal groups are newest first and de-duplicate repeated meals', () => {
  const data = {
    days: {
      '2026-07-30': {
        meals: [item('1', 'Dal', '1 bowl', 'lunch', '2026-07-30T12:00:00.000Z')],
      },
      '2026-07-31': {
        meals: [item('2', 'Dal', '1 bowl', 'lunch', '2026-07-31T12:00:00.000Z')],
      },
      '2026-08-01': {
        meals: [
          item('3', 'Eggs', '2 pieces', 'breakfast', '2026-08-01T08:00:00.000Z'),
          item('4', 'Toast', '2 slices', 'breakfast', '2026-08-01T08:01:00.000Z'),
        ],
      },
    },
  };

  const groups = recentMealGroups(data as never, 3);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].name, 'Eggs & Toast');
  assert.equal(groups[1].name, 'Dal');
});

test('repeating and saving preserve evidence but create add operations', () => {
  const group = recentMealGroups({
    days: {
      '2026-08-01': {
        meals: [item('3', 'Eggs', '2 pieces', 'breakfast', '2026-08-01T08:00:00.000Z')],
      },
    },
  } as never, 1)[0];

  const repeated = mealGroupToOperation(group, 'lunch');
  assert.equal(repeated.type, 'meal');
  assert.equal(repeated.slot, 'lunch');
  assert.deepEqual(repeated.items[0].assumptions, ['reviewed']);

  const saved = savedMealFromGroup(group, 'saved-1', '2026-08-01T09:00:00.000Z');
  const savedOperation = savedMealToOperation(saved);
  assert.equal(savedOperation.description, 'Eggs');
  assert.equal(savedOperation.items[0].source, 'usda');
});

test('saved meal sync keeps the newest edit and never resurrects a deletion', () => {
  const group = recentMealGroups({
    days: {
      '2026-08-01': {
        meals: [item('3', 'Eggs', '2 pieces', 'breakfast', '2026-08-01T08:00:00.000Z')],
      },
    },
  } as never, 1)[0];
  const oldMeal = savedMealFromGroup(group, 'saved-1', '2026-08-01T09:00:00.000Z');
  const newMeal = { ...oldMeal, name: 'Egg breakfast', updatedAt: '2026-08-01T10:00:00.000Z' };

  assert.equal(mergeSavedMeals([oldMeal], [newMeal], [])[0].name, 'Egg breakfast');
  assert.deepEqual(mergeSavedMeals([oldMeal], [newMeal], ['saved-1']), []);
});
