import assert from 'node:assert/strict';
import test from 'node:test';

import { isBackupPayload } from '@/src/lib/backup.ts';

function validPayload() {
  return {
    goals: { calories: 2200, protein: 120, carbs: 250, fat: 70, waterMl: 3000, steps: 8000, weeklyWorkoutMinutes: 150, strengthDays: 2 },
    profile: { allergies: [], injuries: [], updatedAt: '' },
    plan: { method: 'test', summary: 'test', warnings: [], updatedAt: '' },
    estimation: { cupMl: 200 },
    days: {},
    weights: [],
    coachMemory: { dietaryPreferences: [], injuries: [], workoutPreferences: [], coachingStyle: '', notes: '', updatedAt: '' },
    coachMessages: [],
    deletedMealIds: [], deletedWorkoutIds: [], deletedSavedMealIds: [], savedMeals: [],
    learnedFoods: [], deletedLearnedFoodIds: [],
    training: { routines: [], sessions: [], activeSession: null, customExercises: [], deletedRoutineIds: [], deletedSessionIds: [], defaultRestSec: 90, rpeEnabled: false, updatedAt: '' },
  };
}

test('a structurally valid exported vault is accepted', () => {
  assert.equal(isBackupPayload(validPayload()), true);
});

test('malformed nested data is rejected before replacing live state', () => {
  assert.equal(isBackupPayload({ ...validPayload(), days: { today: { meals: 'not-an-array' } } }), false);
  assert.equal(isBackupPayload({ ...validPayload(), training: { sessions: 'not-an-array' } }), false);
  assert.equal(isBackupPayload({ ...validPayload(), goals: { calories: 'many' } }), false);
});
