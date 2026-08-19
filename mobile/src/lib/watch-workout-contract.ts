import { EXERCISES } from '@/src/lib/exercises';
import { exerciseInfo, previousPerformance } from '@/src/lib/training';
import type {
  ActiveRestTimer,
  ActiveSetTimer,
  Routine,
  SessionExercise,
  SetType,
  TrainingData,
  WorkoutSession,
  WorkoutSet,
} from '@/src/types';

export type WatchWorkoutSet = {
  id: string;
  index: number;
  type: SetType;
  weightKg?: number;
  reps?: number;
  durationSec?: number;
  rpe?: number;
  previousWeightKg?: number;
  previousReps?: number;
  previousDurationSec?: number;
  completed: boolean;
};

export type WatchWorkoutExercise = {
  id: string;
  exerciseId: string;
  name: string;
  kind: 'weight-reps' | 'reps-only' | 'duration';
  note?: string;
  restSec: number;
  supersetId?: string;
  sets: WatchWorkoutSet[];
};

export type WatchWorkoutPlan = {
  id: string;
  name: string;
  routineId?: string;
  startedAt?: string;
  endedAt?: string;
  updatedAt: string;
  revision: number;
  exercises: WatchWorkoutExercise[];
  heartRateBpm?: number;
  activeCalories?: number;
  activeSetTimer?: ActiveSetTimer;
  activeRestTimer?: ActiveRestTimer;
};

export type WatchWorkoutLibrary = {
  version: 2;
  unit: 'kg';
  updatedAt: string;
  routines: WatchWorkoutPlan[];
  activeWorkout?: WatchWorkoutPlan;
  activeWorkoutClearedAt?: string;
  activeWorkoutClearedId?: string;
  defaultRestSec: number;
  catalog: {
    id: string;
    name: string;
    kind: 'weight-reps' | 'reps-only' | 'duration';
    primaryMuscle: string;
  }[];
};

function snapshotTime(value?: string) {
  const parsed = value ? Date.parse(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : 0;
}

/**
 * Resolves simultaneous phone/Watch edits with a monotonic revision first and
 * the persisted edit timestamp as the tie breaker. Equal snapshots are ignored
 * so echoing an application context cannot create a sync loop.
 */
export function isWorkoutSnapshotNewer(
  incoming: Pick<WorkoutSession, 'id' | 'startedAt' | 'watchRevision' | 'watchUpdatedAt'>,
  current?: Pick<WorkoutSession, 'id' | 'startedAt' | 'watchRevision' | 'watchUpdatedAt'> | null,
) {
  if (!current) return true;
  if (incoming.id !== current.id) {
    return snapshotTime(incoming.watchUpdatedAt ?? incoming.startedAt)
      > snapshotTime(current.watchUpdatedAt ?? current.startedAt);
  }
  const incomingRevision = incoming.watchRevision ?? 0;
  const currentRevision = current.watchRevision ?? 0;
  if (incomingRevision !== currentRevision) return incomingRevision > currentRevision;
  return snapshotTime(incoming.watchUpdatedAt) > snapshotTime(current.watchUpdatedAt);
}

/** Marks an iPhone-authored workout mutation as a new cross-device snapshot. */
export function revisionedWorkoutUpdate(
  current: WorkoutSession,
  updated: WorkoutSession,
  updatedAt = new Date().toISOString(),
): WorkoutSession {
  return {
    ...updated,
    watchRevision: Math.max(current.watchRevision ?? 0, updated.watchRevision ?? 0) + 1,
    watchUpdatedAt: updatedAt,
  };
}

function setPayload(
  set: Pick<WorkoutSet, 'id' | 'type' | 'weightKg' | 'reps' | 'durationSec' | 'rpe' | 'completed'>,
  index: number,
  previous?: WorkoutSet,
): WatchWorkoutSet {
  return {
    id: set.id,
    index,
    type: set.type,
    weightKg: set.weightKg,
    reps: set.reps,
    durationSec: set.durationSec,
    rpe: set.rpe,
    previousWeightKg: previous?.weightKg,
    previousReps: previous?.reps,
    previousDurationSec: previous?.durationSec,
    completed: set.completed,
  };
}

function exercisePayload(
  entry: SessionExercise,
  training: TrainingData,
): WatchWorkoutExercise {
  const info = exerciseInfo(training, entry.exerciseId);
  const previous = previousPerformance(training, entry.exerciseId);
  return {
    id: entry.id,
    exerciseId: entry.exerciseId,
    name: info.name,
    kind: info.kind,
    note: entry.note,
    restSec: entry.restSec,
    supersetId: entry.supersetId,
    sets: entry.sets.map((set, index) => setPayload(set, index, previous[index] ?? previous.at(-1))),
  };
}

function routinePayload(routine: Routine, training: TrainingData): WatchWorkoutPlan {
  return {
    id: routine.id,
    routineId: routine.id,
    name: routine.name.trim() || 'Workout',
    updatedAt: routine.updatedAt,
    revision: 0,
    exercises: routine.exercises.map((entry) => {
      const info = exerciseInfo(training, entry.exerciseId);
      const previous = previousPerformance(training, entry.exerciseId);
      return {
        id: entry.id,
        exerciseId: entry.exerciseId,
        name: info.name,
        kind: info.kind,
        note: entry.note,
        restSec: entry.restSec,
        supersetId: entry.supersetId,
        sets: entry.sets.map((set, index) => setPayload(
          { ...set, completed: false },
          index,
          previous[index] ?? previous.at(-1),
        )),
      };
    }),
  };
}

export function watchWorkoutPayload(session: WorkoutSession, training: TrainingData): WatchWorkoutPlan {
  return {
    id: session.id,
    routineId: session.routineId,
    name: session.name.trim() || 'Workout',
    startedAt: session.startedAt,
    endedAt: session.endedAt,
    updatedAt: session.watchUpdatedAt ?? session.startedAt,
    revision: session.watchRevision ?? 0,
    exercises: session.exercises.map((entry) => exercisePayload(entry, training)),
    heartRateBpm: session.liveMetrics?.heartRateBpm,
    activeCalories: session.liveMetrics?.activeCalories,
    activeSetTimer: session.activeSetTimer,
    activeRestTimer: session.activeRestTimer,
  };
}

export function watchTrainingPayload(training: TrainingData): WatchWorkoutLibrary {
  return {
    version: 2,
    unit: 'kg',
    updatedAt: training.updatedAt || new Date().toISOString(),
    routines: training.routines.map((routine) => routinePayload(routine, training)),
    activeWorkout: training.activeSession
      ? watchWorkoutPayload(training.activeSession, training)
      : undefined,
    activeWorkoutClearedAt: training.activeSession
      ? undefined
      : training.activeWorkoutTombstone?.clearedAt,
    activeWorkoutClearedId: training.activeSession
      ? undefined
      : training.activeWorkoutTombstone?.workoutId,
    defaultRestSec: training.defaultRestSec,
    catalog: [
      ...EXERCISES.map((exercise) => ({
        id: exercise.id,
        name: exercise.name,
        kind: exercise.kind,
        primaryMuscle: exercise.primaryMuscle,
      })),
      ...training.customExercises.map((exercise) => ({
        id: exercise.id,
        name: exercise.name,
        kind: exercise.kind,
        primaryMuscle: exercise.primaryMuscle,
      })),
    ],
  };
}

export function workoutSessionFromWatch(plan: WatchWorkoutPlan): WorkoutSession {
  return {
    id: plan.id,
    name: plan.name.trim() || 'Workout',
    routineId: plan.routineId,
    startedAt: plan.startedAt ?? new Date().toISOString(),
    endedAt: plan.endedAt,
    activeSetTimer: plan.activeSetTimer,
    activeRestTimer: plan.activeRestTimer,
    watchRevision: plan.revision,
    watchUpdatedAt: plan.updatedAt,
    exercises: plan.exercises.map((entry) => ({
      id: entry.id,
      exerciseId: entry.exerciseId,
      note: entry.note,
      restSec: entry.restSec,
      supersetId: entry.supersetId,
      sets: entry.sets.map((set) => ({
        id: set.id,
        type: set.type,
        weightKg: set.weightKg,
        reps: set.reps,
        durationSec: set.durationSec,
        rpe: set.rpe,
        completed: set.completed,
      })),
    })),
    liveMetrics: plan.heartRateBpm != null || plan.activeCalories != null
      ? {
          heartRateBpm: plan.heartRateBpm,
          activeCalories: plan.activeCalories,
          source: 'Apple Watch',
          updatedAt: plan.updatedAt,
        }
      : undefined,
  };
}
