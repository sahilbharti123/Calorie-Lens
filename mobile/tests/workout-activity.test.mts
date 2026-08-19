import assert from 'node:assert/strict';
import test from 'node:test';

import { classifyAppleWorkout } from '@/src/lib/workout-activity.ts';

test('a mixed weights and treadmill session launches as cross training', () => {
  assert.equal(classifyAppleWorkout(['Barbell back squat', 'Treadmill running']), 'crossTraining');
});

test('single-mode workouts keep their specific HealthKit activity', () => {
  assert.equal(classifyAppleWorkout(['Treadmill running']), 'running');
  assert.equal(classifyAppleWorkout(['Barbell bench press']), 'traditionalStrengthTraining');
  assert.equal(classifyAppleWorkout(['Yoga']), 'yoga');
});
