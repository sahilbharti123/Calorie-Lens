import assert from 'node:assert/strict';
import test from 'node:test';

import {
  EQUIPMENT_TYPES,
  EXERCISES,
  MUSCLE_GROUPS,
  findExercise,
  searchExercises,
} from '../src/lib/exercises.ts';

function normalized(value: string) {
  return value.toLowerCase().replace(/[’']/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
}

test('the built-in library is broad, valid, and collision-free', () => {
  assert.ok(EXERCISES.length >= 250, `expected at least 250 exercises, found ${EXERCISES.length}`);
  assert.equal(new Set(EXERCISES.map((exercise) => exercise.id)).size, EXERCISES.length, 'exercise IDs must be unique');
  assert.equal(
    new Set(EXERCISES.map((exercise) => normalized(exercise.name))).size,
    EXERCISES.length,
    'normalized exercise names must be unique',
  );

  for (const exercise of EXERCISES) {
    assert.equal(findExercise(exercise.id), exercise);
    assert.ok(exercise.instructions.length >= 2, `${exercise.name} needs useful instructions`);
    assert.ok(exercise.tips.length >= 1, `${exercise.name} needs at least one coaching tip`);
    assert.ok(exercise.met > 0, `${exercise.name} needs a positive MET estimate`);
  }
});

test('every visible muscle and equipment filter has exercises', () => {
  for (const muscle of MUSCLE_GROUPS) {
    assert.ok(searchExercises({ muscle }).length > 0, `${muscle} filter is empty`);
  }
  for (const equipment of EQUIPMENT_TYPES) {
    assert.ok(searchExercises({ equipment }).length > 0, `${equipment} filter is empty`);
  }
});

test('gym names, punctuation variants, muscles, and equipment are searchable', () => {
  for (const query of ['21s', "21's", 'twenty ones', 'assault bike', 'trx row', 'band glutes', 'cable rear delt', 'timed core']) {
    assert.ok(searchExercises({ query }).length > 0, `expected a result for “${query}”`);
  }
  assert.equal(searchExercises({ query: '21s' })[0]?.id, 'biceps-21s');
  assert.equal(searchExercises({ query: "21's" })[0]?.id, 'biceps-21s');
});
