import assert from 'node:assert/strict';
import test from 'node:test';

import { calculatePersonalTargets, personalOfflineReply } from '@/src/lib/personalization.ts';

test('recommended targets use activity, pace, target weight and allow a zero-day recovery plan', () => {
  const base = {
    primaryGoal: 'lose-fat' as const,
    equationSex: 'female' as const,
    age: 32,
    heightCm: 165,
    weightKg: 82,
    targetWeightKg: 68,
    goalPace: 'steady' as const,
    activityLevel: 'active' as const,
    workoutPreference: 'gym' as const,
    trainingDays: 0,
    availableMinutes: 30,
    allergies: [],
    injuries: [],
    updatedAt: '',
  };
  const result = calculatePersonalTargets(base);
  assert.equal(result.goals.weeklyWorkoutMinutes, 0);
  assert.equal(result.goals.strengthDays, 0);
  assert.match(result.plan.summary, /toward 68 kg/);

  const sedentary = calculatePersonalTargets({ ...base, activityLevel: 'mostly-seated', trainingDays: 3 });
  assert.ok(result.plan.maintenanceCalories! > sedentary.plan.maintenanceCalories!);
});

test('offline coach respects diet and allergy constraints in protein suggestions', () => {
  const day = {
    date: '2026-08-19', meals: [], workouts: [], waterMl: 0, steps: 0,
    activeCalories: 0, sleepHours: 0,
  };
  const data = {
    profile: {
      primaryGoal: 'build-muscle', dietStyle: 'vegan', allergies: ['soy', 'dairy'],
      injuries: [], coachingTone: 'data-led', updatedAt: '',
    },
    goals: { calories: 2400, protein: 140, carbs: 280, fat: 70, waterMl: 2800, steps: 8000, weeklyWorkoutMinutes: 120, strengthDays: 3 },
    plan: { method: 'test', summary: 'test', warnings: [], updatedAt: '' },
    days: {}, training: { sessions: [], routines: [], customExercises: [], deletedCustomExerciseIds: [], deletedRoutineIds: [], deletedSessionIds: [], activeSession: null, defaultRestSec: 90, rpeEnabled: false, updatedAt: '' },
  };
  const reply = personalOfflineReply('What should I eat for protein?', data as never, day as never);
  assert.doesNotMatch(reply, /tofu|soy chunks|curd|paneer|whey/i);
  assert.match(reply, /lentils|beans|pea protein/i);
});
