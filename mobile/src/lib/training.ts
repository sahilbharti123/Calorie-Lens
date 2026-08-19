/**
 * Training domain logic: volume, estimated 1RM, personal records, previous
 * performance, weekly muscle-set counts, and transparent session energy.
 *
 * Energy uses the same net-MET method as the rest of the app:
 *   active kcal = (MET − 1) × 3.5 × body weight (kg) ÷ 200 × minutes
 * with resistance-training MET bounds from the 2024 Adult Compendium
 * (3.5 light · 5.0 moderate · 6.0 vigorous).
 */

import { findExercise, type Exercise, type MuscleGroup } from '@/src/lib/exercises';
import type {
  Routine,
  RoutineExercise,
  RoutineSetTemplate,
  SessionExercise,
  SetType,
  TrainingData,
  WorkoutSession,
  WorkoutSet,
} from '@/src/types';

export const REST_CHOICES = [0, 30, 45, 60, 90, 120, 150, 180, 240, 300];
export const RPE_CHOICES = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10];

export function newId(prefix: string) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export const initialTraining: TrainingData = {
  routines: [],
  sessions: [],
  activeSession: null,
  customExercises: [],
  deletedCustomExerciseIds: [],
  deletedRoutineIds: [],
  deletedSessionIds: [],
  defaultRestSec: 90,
  rpeEnabled: false,
  updatedAt: '',
};

export function normalizeTraining(saved?: Partial<TrainingData> | null): TrainingData {
  return {
    ...initialTraining,
    ...(saved ?? {}),
    routines: saved?.routines ?? [],
    sessions: saved?.sessions ?? [],
    activeSession: saved?.activeSession ?? null,
    customExercises: (saved?.customExercises ?? []).map((exercise) => ({
      ...exercise,
      updatedAt: exercise.updatedAt ?? exercise.createdAt,
    })),
    deletedCustomExerciseIds: saved?.deletedCustomExerciseIds ?? [],
    deletedRoutineIds: saved?.deletedRoutineIds ?? [],
    deletedSessionIds: saved?.deletedSessionIds ?? [],
    defaultRestSec: saved?.defaultRestSec ?? 90,
    rpeEnabled: saved?.rpeEnabled ?? false,
    activeWorkoutTombstone: saved?.activeWorkoutTombstone,
    updatedAt: saved?.updatedAt ?? '',
  };
}

export function mergeTraining(local: TrainingData, remote: TrainingData): TrainingData {
  const deletedRoutineIds = [...new Set([...local.deletedRoutineIds, ...remote.deletedRoutineIds])].slice(-500);
  const deletedSessionIds = [...new Set([...local.deletedSessionIds, ...remote.deletedSessionIds])].slice(-1000);
  const routines = new Map<string, Routine>();
  for (const routine of [...remote.routines, ...local.routines]) {
    const existing = routines.get(routine.id);
    if (!existing || routine.updatedAt >= existing.updatedAt) routines.set(routine.id, routine);
  }
  const sessions = new Map<string, WorkoutSession>();
  for (const session of [...remote.sessions, ...local.sessions]) sessions.set(session.id, session);
  const deletedCustomExerciseIds = [...new Set([
    ...local.deletedCustomExerciseIds,
    ...remote.deletedCustomExerciseIds,
  ])].slice(-500);
  const customs = new Map<string, TrainingData['customExercises'][number]>();
  for (const custom of [...remote.customExercises, ...local.customExercises]) {
    const existing = customs.get(custom.id);
    if (!existing || custom.updatedAt >= existing.updatedAt) customs.set(custom.id, custom);
  }
  const newestSettings = (local.updatedAt || '') >= (remote.updatedAt || '') ? local : remote;
  const tombstones = [local.activeWorkoutTombstone, remote.activeWorkoutTombstone]
    .filter((value): value is NonNullable<TrainingData['activeWorkoutTombstone']> => Boolean(value))
    .sort((a, b) => a.clearedAt.localeCompare(b.clearedAt));
  const activeWorkoutTombstone = tombstones.at(-1);
  const activeCandidates = [local.activeSession, remote.activeSession]
    .filter((value): value is WorkoutSession => Boolean(value))
    .filter((session) => !(
      activeWorkoutTombstone?.workoutId === session.id
      && activeWorkoutTombstone.clearedAt >= (session.watchUpdatedAt ?? session.startedAt)
    ))
    .sort((a, b) => {
      const revisionDifference = (a.watchRevision ?? 0) - (b.watchRevision ?? 0);
      return revisionDifference || (a.watchUpdatedAt ?? a.startedAt).localeCompare(
        b.watchUpdatedAt ?? b.startedAt,
      );
    });
  return {
    routines: [...routines.values()]
      .filter((routine) => !deletedRoutineIds.includes(routine.id))
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt)),
    sessions: [...sessions.values()]
      .filter((session) => !deletedSessionIds.includes(session.id))
      .sort((a, b) => a.startedAt.localeCompare(b.startedAt))
      .slice(-400),
    activeSession: activeCandidates.at(-1) ?? null,
    customExercises: [...customs.values()].filter((exercise) => !deletedCustomExerciseIds.includes(exercise.id)),
    deletedCustomExerciseIds,
    deletedRoutineIds,
    deletedSessionIds,
    defaultRestSec: newestSettings.defaultRestSec,
    rpeEnabled: newestSettings.rpeEnabled,
    activeWorkoutTombstone,
    updatedAt: [local.updatedAt, remote.updatedAt].sort().at(-1) ?? '',
  };
}

// ---------------------------------------------------------------------------
// Exercise lookups (built-in + custom)
// ---------------------------------------------------------------------------

export function exerciseInfo(training: TrainingData, exerciseId: string): Pick<
  Exercise,
  'id' | 'name' | 'primaryMuscle' | 'secondaryMuscles' | 'equipment' | 'kind' | 'template' | 'gear' | 'met'
> & { custom?: boolean; instructions?: string[]; tips?: string[] } {
  const builtIn = findExercise(exerciseId);
  if (builtIn) return builtIn;
  const custom = training.customExercises.find((candidate) => candidate.id === exerciseId);
  if (custom) {
    return {
      id: custom.id,
      name: custom.name,
      primaryMuscle: (custom.primaryMuscle as MuscleGroup) || 'full body',
      secondaryMuscles: [],
      equipment: (custom.equipment as Exercise['equipment']) || 'other',
      kind: custom.kind,
      template: custom.kind === 'duration' ? 'plank' : custom.kind === 'reps-only' ? 'pushup' : 'curl',
      gear: 'none',
      met: 3.5,
      custom: true,
    };
  }
  return {
    id: exerciseId,
    name: 'Unknown exercise',
    primaryMuscle: 'full body',
    secondaryMuscles: [],
    equipment: 'other',
    kind: 'weight-reps',
    template: 'curl',
    gear: 'none',
    met: 3.5,
  };
}

// ---------------------------------------------------------------------------
// Set math
// ---------------------------------------------------------------------------

export function setVolumeKg(set: WorkoutSet) {
  if (!set.weightKg || !set.reps) return 0;
  return set.weightKg * set.reps;
}

/** Epley estimate; standard practice in strength apps for <10-rep sets. */
export function estimated1Rm(weightKg: number, reps: number) {
  if (!weightKg || !reps) return 0;
  if (reps === 1) return weightKg;
  return weightKg * (1 + reps / 30);
}

export function isWorkingSet(set: WorkoutSet) {
  return set.completed && set.type !== 'warmup';
}

export function sessionTotals(session: WorkoutSession) {
  let volumeKg = 0;
  let sets = 0;
  let reps = 0;
  for (const exercise of session.exercises) {
    for (const set of exercise.sets) {
      if (!set.completed) continue;
      sets += 1;
      reps += set.reps ?? 0;
      volumeKg += setVolumeKg(set);
    }
  }
  return { volumeKg: Math.round(volumeKg), sets, reps };
}

export function nextSetType(current: SetType): SetType {
  const order: SetType[] = ['normal', 'warmup', 'failure', 'drop'];
  return order[(order.indexOf(current) + 1) % order.length];
}

export const SET_TYPE_LABEL: Record<SetType, string> = {
  normal: '',
  warmup: 'W',
  failure: 'F',
  drop: 'D',
};

// ---------------------------------------------------------------------------
// History, previous performance, records
// ---------------------------------------------------------------------------

export function completedSessions(training: TrainingData) {
  return training.sessions
    .filter((session) => Boolean(session.endedAt))
    .sort((a, b) => b.startedAt.localeCompare(a.startedAt));
}

export function sessionsWithExercise(training: TrainingData, exerciseId: string) {
  return completedSessions(training).filter((session) =>
    session.exercises.some((exercise) => exercise.exerciseId === exerciseId
      && exercise.sets.some((set) => set.completed)),
  );
}

/** Last completed sets for an exercise — powers the PREVIOUS column and prefill. */
export function previousPerformance(training: TrainingData, exerciseId: string): WorkoutSet[] {
  const [latest] = sessionsWithExercise(training, exerciseId);
  if (!latest) return [];
  const entry = latest.exercises.find((exercise) => exercise.exerciseId === exerciseId);
  return entry ? entry.sets.filter((set) => set.completed) : [];
}

export function formatSet(set: WorkoutSet, kind: 'weight-reps' | 'reps-only' | 'duration') {
  if (kind === 'duration') return set.durationSec ? `${formatDuration(set.durationSec)}` : '—';
  if (kind === 'reps-only') return set.reps ? `${set.reps} reps` : '—';
  if (set.weightKg == null && set.reps == null) return '—';
  return `${set.weightKg ?? 0} kg × ${set.reps ?? 0}`;
}

export function formatDuration(totalSec: number) {
  const minutes = Math.floor(totalSec / 60);
  const seconds = Math.round(totalSec % 60);
  if (!minutes) return `${seconds}s`;
  return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
}

export type ExerciseRecords = {
  heaviestKg: number;
  best1Rm: number;
  bestSetVolume: number;
  bestSessionVolume: number;
  bestReps: number;
  bestDurationSec: number;
  totalSessions: number;
};

export function exerciseRecords(
  training: TrainingData,
  exerciseId: string,
  options?: { excludeSessionId?: string },
): ExerciseRecords {
  const records: ExerciseRecords = {
    heaviestKg: 0,
    best1Rm: 0,
    bestSetVolume: 0,
    bestSessionVolume: 0,
    bestReps: 0,
    bestDurationSec: 0,
    totalSessions: 0,
  };
  for (const session of completedSessions(training)) {
    if (options?.excludeSessionId && session.id === options.excludeSessionId) continue;
    let sessionVolume = 0;
    let counted = false;
    for (const exercise of session.exercises) {
      if (exercise.exerciseId !== exerciseId) continue;
      for (const set of exercise.sets) {
        if (!isWorkingSet(set)) continue;
        counted = true;
        const volume = setVolumeKg(set);
        sessionVolume += volume;
        if ((set.weightKg ?? 0) > records.heaviestKg) records.heaviestKg = set.weightKg ?? 0;
        const oneRm = estimated1Rm(set.weightKg ?? 0, set.reps ?? 0);
        if (oneRm > records.best1Rm) records.best1Rm = oneRm;
        if (volume > records.bestSetVolume) records.bestSetVolume = volume;
        if ((set.reps ?? 0) > records.bestReps) records.bestReps = set.reps ?? 0;
        if ((set.durationSec ?? 0) > records.bestDurationSec) records.bestDurationSec = set.durationSec ?? 0;
      }
    }
    if (counted) {
      records.totalSessions += 1;
      if (sessionVolume > records.bestSessionVolume) records.bestSessionVolume = sessionVolume;
    }
  }
  return records;
}

/** PR labels earned by a set, judged against history *before* this session. */
export function detectSetRecords(
  set: WorkoutSet,
  kind: 'weight-reps' | 'reps-only' | 'duration',
  before: ExerciseRecords,
): string[] {
  if (set.type === 'warmup') return [];
  const flags: string[] = [];
  if (kind === 'weight-reps') {
    if ((set.weightKg ?? 0) > 0 && (set.weightKg ?? 0) > before.heaviestKg) flags.push('Heaviest weight');
    const oneRm = estimated1Rm(set.weightKg ?? 0, set.reps ?? 0);
    if (oneRm > 0 && oneRm > before.best1Rm) flags.push('Best est. 1RM');
    const volume = setVolumeKg(set);
    if (volume > 0 && volume > before.bestSetVolume) flags.push('Best set volume');
  } else if (kind === 'reps-only') {
    if ((set.reps ?? 0) > 0 && (set.reps ?? 0) > before.bestReps) flags.push('Most reps');
  } else if ((set.durationSec ?? 0) > 0 && (set.durationSec ?? 0) > before.bestDurationSec) {
    flags.push('Longest hold');
  }
  return flags;
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

export type TrendPoint = { date: string; value: number };
export type TrendMetric = 'heaviest' | 'oneRm' | 'setVolume' | 'sessionVolume' | 'reps' | 'duration';

export function exerciseTrend(
  training: TrainingData,
  exerciseId: string,
  metric: TrendMetric,
): TrendPoint[] {
  const points: TrendPoint[] = [];
  for (const session of sessionsWithExercise(training, exerciseId).slice(0, 60).reverse()) {
    let heaviest = 0;
    let oneRm = 0;
    let setVolume = 0;
    let sessionVolume = 0;
    let reps = 0;
    let duration = 0;
    for (const exercise of session.exercises) {
      if (exercise.exerciseId !== exerciseId) continue;
      for (const set of exercise.sets) {
        if (!isWorkingSet(set)) continue;
        heaviest = Math.max(heaviest, set.weightKg ?? 0);
        oneRm = Math.max(oneRm, estimated1Rm(set.weightKg ?? 0, set.reps ?? 0));
        setVolume = Math.max(setVolume, setVolumeKg(set));
        sessionVolume += setVolumeKg(set);
        reps = Math.max(reps, set.reps ?? 0);
        duration = Math.max(duration, set.durationSec ?? 0);
      }
    }
    const value = metric === 'heaviest' ? heaviest
      : metric === 'oneRm' ? Math.round(oneRm * 10) / 10
        : metric === 'setVolume' ? setVolume
          : metric === 'sessionVolume' ? sessionVolume
            : metric === 'reps' ? reps
              : duration;
    if (value > 0) points.push({ date: session.startedAt.slice(0, 10), value });
  }
  return points;
}

// ---------------------------------------------------------------------------
// Weekly muscle volume
// ---------------------------------------------------------------------------

export function weeklyMuscleSets(training: TrainingData, now = new Date()) {
  const sevenDaysAgo = new Date(now);
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
  const cutoff = sevenDaysAgo.toISOString();
  const counts = new Map<MuscleGroup, number>();
  for (const session of completedSessions(training)) {
    if (session.startedAt < cutoff) continue;
    for (const exercise of session.exercises) {
      const info = exerciseInfo(training, exercise.exerciseId);
      if (info.primaryMuscle === 'cardio') continue;
      const working = exercise.sets.filter(isWorkingSet).length;
      if (!working) continue;
      counts.set(info.primaryMuscle, (counts.get(info.primaryMuscle) ?? 0) + working);
    }
  }
  return [...counts.entries()]
    .map(([muscle, sets]) => ({ muscle, sets }))
    .sort((a, b) => b.sets - a.sets);
}

export function weeklyTrainingSummary(training: TrainingData, now = new Date()) {
  const sevenDaysAgo = new Date(now);
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
  const cutoff = sevenDaysAgo.toISOString();
  let workouts = 0;
  let minutes = 0;
  let volumeKg = 0;
  for (const session of completedSessions(training)) {
    if (session.startedAt < cutoff) continue;
    workouts += 1;
    minutes += session.durationMin ?? 0;
    volumeKg += session.totalVolumeKg ?? 0;
  }
  return { workouts, minutes, volumeKg: Math.round(volumeKg) };
}

// ---------------------------------------------------------------------------
// Session energy
// ---------------------------------------------------------------------------

export function sessionEnergy(
  session: WorkoutSession,
  training: TrainingData,
  bodyWeightKg: number | undefined,
  durationMin: number,
) {
  if (!bodyWeightKg || durationMin <= 0) return null;
  const setsByExercise = session.exercises
    .map((exercise) => ({
      met: exerciseInfo(training, exercise.exerciseId).met,
      sets: exercise.sets.filter((set) => set.completed).length,
    }))
    .filter((entry) => entry.sets > 0);
  const totalSets = setsByExercise.reduce((sum, entry) => sum + entry.sets, 0);
  if (!totalSets) return null;
  const met = setsByExercise.reduce((sum, entry) => sum + entry.met * entry.sets, 0) / totalSets;
  const metLow = Math.max(2.5, Math.min(met - 1.5, met * 0.7));
  const metHigh = met + 1;
  const active = (value: number) => Math.max(0, value - 1) * 3.5 * bodyWeightKg / 200 * durationMin;
  return {
    calories: Math.round(active(met)),
    calorieLow: Math.round(active(metLow)),
    calorieHigh: Math.round(active(metHigh)),
    met: Math.round(met * 10) / 10,
    basis: `${Math.round(met * 10) / 10} MET (set-weighted, 2024 Compendium) · ${bodyWeightKg} kg · ${Math.round(durationMin)} min · resting energy excluded`,
  };
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

export function makeRoutineSet(partial?: Partial<RoutineSetTemplate>): RoutineSetTemplate {
  return { id: newId('rset'), type: 'normal', ...partial };
}

export function makeRoutineExercise(
  exerciseId: string,
  defaultRestSec: number,
  sets = 3,
): RoutineExercise {
  return {
    id: newId('rex'),
    exerciseId,
    restSec: defaultRestSec,
    sets: Array.from({ length: sets }, () => makeRoutineSet()),
  };
}

export function sessionSetFromTemplate(template: RoutineSetTemplate): WorkoutSet {
  return {
    id: newId('set'),
    type: template.type,
    weightKg: template.weightKg,
    reps: template.reps ?? template.repsMax ?? template.repsMin,
    durationSec: template.durationSec,
    completed: false,
  };
}

export function sessionExerciseFromRoutine(entry: RoutineExercise): SessionExercise {
  return {
    id: newId('sex'),
    exerciseId: entry.exerciseId,
    note: entry.note,
    restSec: entry.restSec,
    supersetId: entry.supersetId,
    sets: entry.sets.length
      ? entry.sets.map(sessionSetFromTemplate)
      : [sessionSetFromTemplate(makeRoutineSet())],
  };
}

export function makeSessionExercise(exerciseId: string, defaultRestSec: number): SessionExercise {
  return {
    id: newId('sex'),
    exerciseId,
    restSec: defaultRestSec,
    sets: [{ id: newId('set'), type: 'normal', completed: false }],
  };
}

export function routineFromSession(session: WorkoutSession, name: string): Routine {
  const now = new Date().toISOString();
  return {
    id: newId('routine'),
    name,
    createdAt: now,
    updatedAt: now,
    exercises: session.exercises
      .filter((exercise) => exercise.sets.some((set) => set.completed))
      .map((exercise) => ({
        id: newId('rex'),
        exerciseId: exercise.exerciseId,
        note: exercise.note,
        restSec: exercise.restSec,
        supersetId: exercise.supersetId,
        sets: exercise.sets
          .filter((set) => set.completed)
          .map((set) => ({
            id: newId('rset'),
            type: set.type,
            weightKg: set.weightKg,
            reps: set.reps,
            durationSec: set.durationSec,
          })),
      })),
  };
}

export function templateTargetLabel(template: RoutineSetTemplate, kind: 'weight-reps' | 'reps-only' | 'duration') {
  if (kind === 'duration') return template.durationSec ? formatDuration(template.durationSec) : 'time';
  const reps = template.repsMin && template.repsMax
    ? `${template.repsMin}–${template.repsMax}`
    : template.reps ?? template.repsMax ?? template.repsMin;
  if (kind === 'reps-only') return reps ? `${reps} reps` : 'reps';
  const weight = template.weightKg ? `${template.weightKg} kg × ` : '';
  return reps ? `${weight}${reps}` : weight || 'work set';
}
