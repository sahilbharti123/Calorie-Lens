import assert from 'node:assert/strict';
import test from 'node:test';

import { initialTraining, mergeTraining } from '@/src/lib/training.ts';
import {
  isWorkoutSnapshotNewer,
  revisionedWorkoutUpdate,
  watchTrainingPayload,
  workoutSessionFromWatch,
  type WatchWorkoutPlan,
} from '@/src/lib/watch-workout-contract.ts';
import type { TrainingData, WorkoutSession } from '@/src/types.ts';

const prior: WorkoutSession = {
  id: 'prior-session',
  name: 'Prior workout',
  startedAt: '2026-08-15T10:00:00.000Z',
  endedAt: '2026-08-15T11:00:00.000Z',
  exercises: [{
    id: 'prior-bench',
    exerciseId: 'bench-press',
    restSec: 90,
    sets: [{ id: 'prior-set', type: 'normal', weightKg: 82.5, reps: 6, completed: true }],
  }],
};

const active: WorkoutSession = {
  id: 'watch-session',
  routineId: 'push-day',
  name: 'Push day',
  startedAt: '2026-08-17T10:00:00.000Z',
  watchRevision: 4,
  watchUpdatedAt: '2026-08-17T10:04:00.000Z',
  exercises: [
    {
      id: 'session-bench',
      exerciseId: 'bench-press',
      note: 'Pause on the chest',
      restSec: 120,
      supersetId: 'push-pair',
      sets: [{ id: 'bench-set', type: 'normal', weightKg: 85, reps: 5, rpe: 9, completed: true }],
    },
    {
      id: 'session-plank',
      exerciseId: 'plank',
      restSec: 45,
      sets: [{ id: 'plank-set', type: 'normal', durationSec: 60, completed: false }],
    },
  ],
  liveMetrics: {
    heartRateBpm: 131,
    activeCalories: 147.4,
    source: 'Apple Watch',
    updatedAt: '2026-08-17T10:04:00.000Z',
  },
  activeSetTimer: {
    sessionExerciseId: 'session-plank',
    setId: 'plank-set',
    targetSec: 60,
    startedAt: '2026-08-17T10:04:00.000Z',
    endsAt: '2026-08-17T10:05:00.000Z',
  },
  activeRestTimer: { endsAt: 1_776_594_330_000, totalSec: 90 },
};

function fixture(): TrainingData {
  return {
    ...initialTraining,
    sessions: [prior],
    activeSession: active,
    routines: [{
      id: 'push-day',
      name: 'Push day',
      createdAt: '2026-08-01T00:00:00.000Z',
      updatedAt: '2026-08-16T00:00:00.000Z',
      exercises: [{
        id: 'routine-bench',
        exerciseId: 'bench-press',
        note: 'Pause on the chest',
        restSec: 120,
        sets: [{ id: 'routine-set', type: 'normal', reps: 6 }],
      }],
    }],
    updatedAt: '2026-08-17T10:04:00.000Z',
  };
}

test('the Watch receives routines, the exercise catalog, and previous performance', () => {
  const payload = watchTrainingPayload(fixture());
  assert.equal(payload.version, 2);
  assert.equal(payload.routines[0]?.exercises[0]?.name, 'Bench Press (Barbell)');
  assert.equal(payload.routines[0]?.exercises[0]?.sets[0]?.previousWeightKg, 82.5);
  assert.equal(payload.routines[0]?.exercises[0]?.sets[0]?.previousReps, 6);
  assert.ok(payload.catalog.some((exercise) => exercise.id === 'plank' && exercise.kind === 'duration'));
});

test('a complete Watch snapshot round-trips weights, reps, RPE, supersets, timers, notes, and metrics', () => {
  const payload = watchTrainingPayload(fixture()).activeWorkout;
  assert.ok(payload);
  const restored = workoutSessionFromWatch(payload);
  assert.equal(restored.watchRevision, 4);
  assert.equal(restored.exercises[0]?.sets[0]?.weightKg, 85);
  assert.equal(restored.exercises[0]?.sets[0]?.reps, 5);
  assert.equal(restored.exercises[0]?.sets[0]?.rpe, 9);
  assert.equal(restored.exercises[0]?.supersetId, 'push-pair');
  assert.equal(restored.exercises[0]?.note, 'Pause on the chest');
  assert.equal(restored.exercises[1]?.sets[0]?.durationSec, 60);
  assert.equal(restored.liveMetrics?.activeCalories, 147.4);
  assert.equal(restored.activeSetTimer?.setId, 'plank-set');
  assert.equal(restored.activeRestTimer?.totalSec, 90);
});

test('a Watch-started workout remains valid without the iPhone or a routine', () => {
  const watchOnly: WatchWorkoutPlan = {
    id: 'offline-watch-session',
    name: 'Workout',
    startedAt: '2026-08-17T12:00:00.000Z',
    updatedAt: '2026-08-17T12:05:00.000Z',
    revision: 8,
    exercises: [{
      id: 'watch-exercise',
      exerciseId: 'plank',
      name: 'Plank',
      kind: 'duration',
      restSec: 45,
      sets: [{ id: 'watch-set', index: 0, type: 'normal', durationSec: 47, completed: true }],
    }],
  };
  const restored = workoutSessionFromWatch(watchOnly);
  assert.equal(restored.id, 'offline-watch-session');
  assert.equal(restored.exercises[0]?.sets[0]?.durationSec, 47);
  assert.equal(restored.exercises[0]?.sets[0]?.completed, true);
});

test('phone-authored edits advance the shared revision and break ties by time', () => {
  const changed = revisionedWorkoutUpdate(
    active,
    { ...active, name: 'Push day — edited on phone' },
    '2026-08-17T10:05:00.000Z',
  );
  assert.equal(changed.watchRevision, 5);
  assert.equal(changed.watchUpdatedAt, '2026-08-17T10:05:00.000Z');
  assert.equal(isWorkoutSnapshotNewer(changed, active), true);
  assert.equal(isWorkoutSnapshotNewer(active, changed), false);

  const sameRevisionLater = { ...active, watchUpdatedAt: '2026-08-17T10:04:01.000Z' };
  assert.equal(isWorkoutSnapshotNewer(sameRevisionLater, active), true);
  assert.equal(isWorkoutSnapshotNewer(active, active), false);
});

test('the phone sends a dated tombstone after an active workout is cleared', () => {
  const training = fixture();
  training.activeWorkoutTombstone = {
    workoutId: training.activeSession!.id,
    clearedAt: '2026-08-17T11:00:00.000Z',
  };
  training.activeSession = null;
  training.updatedAt = '2026-08-17T11:00:00.000Z';
  const payload = watchTrainingPayload(training);
  assert.equal(payload.activeWorkout, undefined);
  assert.equal(payload.activeWorkoutClearedAt, '2026-08-17T11:00:00.000Z');
  assert.equal(payload.activeWorkoutClearedId, 'watch-session');
});

test('an unrelated routine edit never creates a workout tombstone', () => {
  const training = fixture();
  training.activeSession = null;
  training.updatedAt = '2026-08-18T11:00:00.000Z';
  const payload = watchTrainingPayload(training);
  assert.equal(payload.activeWorkoutClearedAt, undefined);
  assert.equal(payload.activeWorkoutClearedId, undefined);
});

test('a tombstone clears only its own workout during cloud merge', () => {
  const remote = fixture();
  const cleared = {
    ...fixture(),
    activeSession: null,
    activeWorkoutTombstone: {
      workoutId: 'watch-session',
      clearedAt: '2026-08-17T11:00:00.000Z',
    },
  };
  assert.equal(mergeTraining(cleared, remote).activeSession, null);

  cleared.activeWorkoutTombstone = {
    workoutId: 'some-old-workout',
    clearedAt: '2026-08-18T11:00:00.000Z',
  };
  assert.equal(mergeTraining(cleared, remote).activeSession?.id, 'watch-session');
});

test('custom exercise edits merge newest and deletions do not resurrect', () => {
  const local = fixture();
  const remote = fixture();
  remote.customExercises = [{
    id: 'custom-rehab',
    name: 'Old rehab hold',
    equipment: 'band',
    primaryMuscle: 'quads',
    kind: 'duration',
    createdAt: '2026-08-01T00:00:00.000Z',
    updatedAt: '2026-08-01T00:00:00.000Z',
  }];
  local.customExercises = [{
    ...remote.customExercises[0]!,
    name: 'Supported rehab hold',
    updatedAt: '2026-08-02T00:00:00.000Z',
  }];
  assert.equal(mergeTraining(local, remote).customExercises[0]?.name, 'Supported rehab hold');

  local.customExercises = [];
  local.deletedCustomExerciseIds = ['custom-rehab'];
  assert.equal(mergeTraining(local, remote).customExercises.length, 0);
});
