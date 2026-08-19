import React, { createContext, useCallback, useContext, useMemo, useRef } from 'react';

import { TEMPLATE_ROUTINE_SEEDS } from '@/src/lib/exercises';
import {
  isWorkoutSnapshotNewer,
  revisionedWorkoutUpdate,
} from '@/src/lib/watch-workout-contract';
import {
  detectSetRecords,
  exerciseInfo,
  exerciseRecords,
  makeRoutineExercise,
  makeRoutineSet,
  makeSessionExercise,
  newId,
  previousPerformance,
  routineFromSession,
  sessionEnergy,
  sessionExerciseFromRoutine,
  sessionTotals,
} from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import type {
  CustomExercise,
  Routine,
  SessionExercise,
  WorkoutSession,
  WorkoutSet,
} from '@/src/types';

type WorkoutContextValue = {
  /** Editable draft used by the routine editor screen. */
  draftRef: React.RefObject<Routine | null>;
  beginRoutineDraft: (routineId?: string) => Routine;
  saveRoutineDraft: (draft: Routine) => void;
  deleteRoutine: (routineId: string) => void;
  duplicateRoutine: (routineId: string) => void;
  importTemplateRoutine: (seedName: string) => void;
  createCustomExercise: (exercise: Pick<CustomExercise, 'name' | 'equipment' | 'primaryMuscle' | 'kind'>) => CustomExercise;
  updateCustomExercise: (id: string, patch: Pick<CustomExercise, 'name' | 'equipment' | 'primaryMuscle' | 'kind'>) => void;
  deleteCustomExercise: (id: string) => boolean;

  startEmptyWorkout: () => void;
  startRoutine: (routineId: string) => void;
  resumeOrStartFromRoutine: (routineId: string) => void;
  updateActiveSession: (recipe: (session: WorkoutSession) => WorkoutSession) => void;
  updateLiveMetricsFromWatch: (metrics: WorkoutSession['liveMetrics']) => void;
  upsertActiveWorkoutFromWatch: (session: WorkoutSession) => void;
  completeSet: (
    sessionExerciseId: string,
    setId: string,
    override?: Partial<WorkoutSet>,
  ) => string[];
  discardActiveWorkout: () => void;
  finishActiveWorkout: () => string | null;
  deleteSession: (sessionId: string) => void;
  saveSessionAsRoutine: (sessionId: string, name: string) => void;
  setDefaultRest: (seconds: number) => void;
  setRpeEnabled: (enabled: boolean) => void;
};

const WorkoutContext = createContext<WorkoutContextValue | null>(null);

export function WorkoutProvider({ children }: React.PropsWithChildren) {
  const { applyOperations, data, updateTraining } = useApp();
  const training = data.training;
  const draftRef = useRef<Routine | null>(null);

  const trainingRef = useRef(training);
  trainingRef.current = training;
  const weightsRef = useRef(data.weights);
  weightsRef.current = data.weights;

  const beginRoutineDraft = useCallback((routineId?: string) => {
    const existing = routineId
      ? trainingRef.current.routines.find((routine) => routine.id === routineId)
      : undefined;
    const now = new Date().toISOString();
    const draft: Routine = existing
      ? JSON.parse(JSON.stringify(existing)) as Routine
      : { id: newId('routine'), name: '', exercises: [], createdAt: now, updatedAt: now };
    draftRef.current = draft;
    return draft;
  }, []);

  const saveRoutineDraft = useCallback((draft: Routine) => {
    const cleaned: Routine = {
      ...draft,
      name: draft.name.trim() || 'My routine',
      updatedAt: new Date().toISOString(),
      exercises: draft.exercises.map((exercise) => ({
        ...exercise,
        sets: exercise.sets.length ? exercise.sets : [makeRoutineSet()],
      })),
    };
    updateTraining((current) => {
      const others = current.routines.filter((routine) => routine.id !== cleaned.id);
      return { ...current, routines: [...others, cleaned] };
    });
    draftRef.current = null;
  }, [updateTraining]);

  const deleteRoutine = useCallback((routineId: string) => {
    updateTraining((current) => ({
      ...current,
      routines: current.routines.filter((routine) => routine.id !== routineId),
      deletedRoutineIds: [...new Set([...current.deletedRoutineIds, routineId])].slice(-500),
    }));
  }, [updateTraining]);

  const duplicateRoutine = useCallback((routineId: string) => {
    updateTraining((current) => {
      const source = current.routines.find((routine) => routine.id === routineId);
      if (!source) return current;
      const now = new Date().toISOString();
      const copy: Routine = {
        ...JSON.parse(JSON.stringify(source)) as Routine,
        id: newId('routine'),
        name: `${source.name} (copy)`,
        createdAt: now,
        updatedAt: now,
        lastPerformedAt: undefined,
      };
      copy.exercises = copy.exercises.map((exercise) => ({
        ...exercise,
        id: newId('rex'),
        sets: exercise.sets.map((set) => ({ ...set, id: newId('rset') })),
      }));
      return { ...current, routines: [...current.routines, copy] };
    });
  }, [updateTraining]);

  const importTemplateRoutine = useCallback((seedName: string) => {
    const seed = TEMPLATE_ROUTINE_SEEDS.find((candidate) => candidate.name === seedName);
    if (!seed) return;
    updateTraining((current) => {
      const now = new Date().toISOString();
      const routine: Routine = {
        id: newId('routine'),
        name: seed.name,
        folder: seed.folder,
        createdAt: now,
        updatedAt: now,
        exercises: seed.items.map((item) => ({
          ...makeRoutineExercise(item.exerciseId, current.defaultRestSec, item.sets),
          sets: Array.from({ length: item.sets }, () => makeRoutineSet({
            repsMin: item.repsMin,
            repsMax: item.repsMax,
            durationSec: item.durationSec,
          })),
        })),
      };
      return { ...current, routines: [...current.routines, routine] };
    });
  }, [updateTraining]);

  const createCustomExercise = useCallback((exercise: Pick<CustomExercise, 'name' | 'equipment' | 'primaryMuscle' | 'kind'>) => {
    const now = new Date().toISOString();
    const created: CustomExercise = {
      ...exercise,
      id: newId('custom-exercise'),
      name: exercise.name.trim(),
      createdAt: now,
      updatedAt: now,
    };
    updateTraining((current) => ({
      ...current,
      customExercises: [...current.customExercises, created],
    }));
    return created;
  }, [updateTraining]);

  const updateCustomExercise = useCallback((id: string, patch: Pick<CustomExercise, 'name' | 'equipment' | 'primaryMuscle' | 'kind'>) => {
    updateTraining((current) => ({
      ...current,
      customExercises: current.customExercises.map((exercise) => exercise.id === id
        ? { ...exercise, ...patch, name: patch.name.trim(), updatedAt: new Date().toISOString() }
        : exercise),
    }));
  }, [updateTraining]);

  const deleteCustomExercise = useCallback((id: string) => {
    const current = trainingRef.current;
    const inUse = current.activeSession?.exercises.some((entry) => entry.exerciseId === id)
      || current.routines.some((routine) => routine.exercises.some((entry) => entry.exerciseId === id));
    if (inUse) return false;
    updateTraining((training) => ({
      ...training,
      customExercises: training.customExercises.filter((exercise) => exercise.id !== id),
      deletedCustomExerciseIds: [...new Set([...training.deletedCustomExerciseIds, id])].slice(-500),
    }));
    return true;
  }, [updateTraining]);

  // ------------------------------------------------------------- live session

  const startEmptyWorkout = useCallback(() => {
    const now = new Date().toISOString();
    updateTraining((current) => current.activeSession ? current : {
      ...current,
      activeSession: {
        id: newId('workout'),
        name: 'Workout',
        startedAt: now,
        exercises: [],
        watchRevision: 1,
        watchUpdatedAt: now,
      },
    });
  }, [updateTraining]);

  const startRoutine = useCallback((routineId: string) => {
    updateTraining((current) => {
      if (current.activeSession) return current;
      const routine = current.routines.find((candidate) => candidate.id === routineId);
      if (!routine) return current;
      const now = new Date().toISOString();
      return {
        ...current,
        activeSession: {
          id: newId('workout'),
          name: routine.name,
          routineId: routine.id,
          startedAt: now,
          exercises: routine.exercises.map(sessionExerciseFromRoutine),
          watchRevision: 1,
          watchUpdatedAt: now,
        },
      };
    });
  }, [updateTraining]);

  const resumeOrStartFromRoutine = useCallback((routineId: string) => {
    if (trainingRef.current.activeSession) return;
    startRoutine(routineId);
  }, [startRoutine]);

  const updateActiveSession = useCallback((recipe: (session: WorkoutSession) => WorkoutSession) => {
    updateTraining((current) => current.activeSession
      ? {
          ...current,
          activeSession: revisionedWorkoutUpdate(
            current.activeSession,
            recipe(current.activeSession),
          ),
        }
      : current);
  }, [updateTraining]);

  const updateLiveMetricsFromWatch = useCallback((metrics: WorkoutSession['liveMetrics']) => {
    updateTraining((current) => current.activeSession
      ? { ...current, activeSession: { ...current.activeSession, liveMetrics: metrics } }
      : current);
  }, [updateTraining]);

  /**
   * Applies a complete Watch snapshot atomically. The Watch sends snapshots,
   * rather than a stream of fragile field mutations, so an offline session can
   * be replayed after the phone reconnects without losing weight or rep edits.
   */
  const upsertActiveWorkoutFromWatch = useCallback((incoming: WorkoutSession) => {
    updateTraining((current) => {
      if (current.sessions.some((saved) => saved.id === incoming.id)) return current;
      const active = current.activeSession;
      if (!isWorkoutSnapshotNewer(incoming, active)) return current;

      const priorExercises = new Map((active?.exercises ?? []).map((entry) => [entry.id, entry]));
      const exercises = incoming.exercises.map((entry) => {
        const prior = priorExercises.get(entry.id);
        const priorSets = new Map((prior?.sets ?? []).map((set) => [set.id, set]));
        const info = exerciseInfo(current, entry.exerciseId);
        const records = exerciseRecords(current, entry.exerciseId);
        return {
          ...entry,
          sets: entry.sets.map((set) => {
            const previous = priorSets.get(set.id);
            if (!set.completed || previous?.completed) {
              return previous?.completed && set.completed
                ? { ...set, prFlags: previous.prFlags }
                : set;
            }
            const flags = detectSetRecords(set, info.kind, records);
            return { ...set, prFlags: flags.length ? flags : undefined };
          }),
        };
      });

      return {
        ...current,
        activeSession: {
          ...active,
          ...incoming,
          exercises,
          liveMetrics: incoming.liveMetrics ?? active?.liveMetrics,
        },
      };
    });
  }, [updateTraining]);

  /**
   * Marks a set complete, prefilling blank inputs from the previous
   * performance, and returns any PR labels earned (evaluated against history
   * from *before* this session, Hevy-style live detection).
   */
  const completeSet = useCallback((
    sessionExerciseId: string,
    setId: string,
    override: Partial<WorkoutSet> = {},
  ): string[] => {
    const current = trainingRef.current;
    const session = current.activeSession;
    if (!session) return [];
    const exerciseEntry = session.exercises.find((entry) => entry.id === sessionExerciseId);
    if (!exerciseEntry) return [];
    const set = exerciseEntry.sets.find((candidate) => candidate.id === setId);
    if (!set) return [];
    const info = exerciseInfo(current, exerciseEntry.exerciseId);
    const previous = previousPerformance(current, exerciseEntry.exerciseId);
    const index = exerciseEntry.sets.indexOf(set);
    const fallback = previous[index] ?? previous.at(-1);

    const completed: WorkoutSet = {
      ...set,
      ...override,
      weightKg: override.weightKg ?? set.weightKg ?? fallback?.weightKg,
      reps: override.reps ?? set.reps ?? fallback?.reps,
      durationSec: override.durationSec ?? set.durationSec ?? fallback?.durationSec,
      completed: true,
    };
    const before = exerciseRecords(current, exerciseEntry.exerciseId);
    const flags = detectSetRecords(completed, info.kind, before);
    completed.prFlags = flags.length ? flags : undefined;

    updateTraining((state) => {
      if (!state.activeSession) return state;
      return {
        ...state,
        activeSession: {
          ...state.activeSession,
          exercises: state.activeSession.exercises.map((entry) => entry.id !== sessionExerciseId
            ? entry
            : {
                ...entry,
                sets: entry.sets.map((candidate) => candidate.id === setId ? completed : candidate),
              }),
          watchRevision: (state.activeSession.watchRevision ?? 0) + 1,
          watchUpdatedAt: new Date().toISOString(),
        },
      };
    });
    return flags;
  }, [updateTraining]);

  const discardActiveWorkout = useCallback(() => {
    updateTraining((current) => ({
      ...current,
      activeWorkoutTombstone: current.activeSession
        ? { workoutId: current.activeSession.id, clearedAt: new Date().toISOString() }
        : current.activeWorkoutTombstone,
      activeSession: null,
    }));
  }, [updateTraining]);

  const finishActiveWorkout = useCallback((): string | null => {
    const current = trainingRef.current;
    const session = current.activeSession;
    if (!session) return null;
    const endedAt = new Date();
    const durationMin = Math.max(
      1,
      Math.round((endedAt.getTime() - new Date(session.startedAt).getTime()) / 60_000),
    );
    const exercises: SessionExercise[] = session.exercises
      .map((entry) => ({ ...entry, sets: entry.sets.filter((set) => set.completed) }))
      .filter((entry) => entry.sets.length > 0);
    if (!exercises.length) {
      updateTraining((state) => ({
        ...state,
        activeWorkoutTombstone: state.activeSession
          ? { workoutId: state.activeSession.id, clearedAt: endedAt.toISOString() }
          : state.activeWorkoutTombstone,
        activeSession: null,
      }));
      return null;
    }
    const totals = sessionTotals({ ...session, exercises });
    const records = exercises.reduce(
      (sum, entry) => sum + entry.sets.reduce((count, set) => count + (set.prFlags?.length ?? 0), 0),
      0,
    );
    const bodyWeight = weightsRef.current.at(-1)?.kg;
    const energy = sessionEnergy({ ...session, exercises }, current, bodyWeight, durationMin);
    const measuredCalories = session.liveMetrics?.activeCalories;
    const hasMeasuredCalories = measuredCalories != null && measuredCalories > 0;
    const finished: WorkoutSession = {
      ...session,
      // The live session holds the raw text of the name field, so this is the
      // one place the fallback has to hold: nothing reaches history unnamed.
      name: session.name.trim() || 'Workout',
      exercises,
      endedAt: endedAt.toISOString(),
      durationMin,
      totalVolumeKg: totals.volumeKg,
      totalSets: totals.sets,
      records,
      calories: hasMeasuredCalories ? Math.round(measuredCalories) : energy?.calories,
      calorieLow: hasMeasuredCalories ? undefined : energy?.calorieLow,
      calorieHigh: hasMeasuredCalories ? undefined : energy?.calorieHigh,
      calorieBasis: hasMeasuredCalories
        ? 'Measured live by Apple Watch during this workout.'
        : energy?.basis,
    };
    updateTraining((state) => ({
      ...state,
      activeWorkoutTombstone: {
        workoutId: session.id,
        clearedAt: finished.endedAt!,
      },
      activeSession: null,
      sessions: [...state.sessions, finished].slice(-400),
      routines: session.routineId
        ? state.routines.map((routine) => routine.id === session.routineId
          ? { ...routine, lastPerformedAt: finished.endedAt }
          : routine)
        : state.routines,
    }));
    if (hasMeasuredCalories) {
      applyOperations([{
        type: 'workout',
        action: 'add',
        name: finished.name,
        durationMin,
        calories: Math.round(measuredCalories),
        intensity: 'moderate',
        confidence: 'high',
        sourceLabel: 'Apple Watch',
        sourceId: session.id,
        basis: 'Active energy measured live during the HealthKit workout session.',
      }]);
    } else if (energy) {
      applyOperations([{
        type: 'workout',
        action: 'add',
        name: finished.name,
        durationMin,
        calories: energy.calories,
        calorieLow: energy.calorieLow,
        calorieHigh: energy.calorieHigh,
        intensity: 'moderate',
        met: energy.met,
        sourceLabel: '2024 Adult Compendium of Physical Activities',
        sourceId: 'conditioning-exercise',
        confidence: 'low',
        basis: energy.basis,
      }]);
    } else {
      applyOperations([{
        type: 'workout',
        action: 'add',
        name: finished.name,
        durationMin,
        calories: 0,
        intensity: 'moderate',
        confidence: 'low',
        basis: 'Add your body weight in Profile to estimate active energy.',
      }]);
    }
    return finished.id;
  }, [applyOperations, updateTraining]);

  const deleteSession = useCallback((sessionId: string) => {
    updateTraining((current) => ({
      ...current,
      sessions: current.sessions.filter((session) => session.id !== sessionId),
      deletedSessionIds: [...new Set([...current.deletedSessionIds, sessionId])].slice(-1000),
    }));
  }, [updateTraining]);

  const saveSessionAsRoutine = useCallback((sessionId: string, name: string) => {
    updateTraining((current) => {
      const session = current.sessions.find((candidate) => candidate.id === sessionId);
      if (!session) return current;
      return { ...current, routines: [...current.routines, routineFromSession(session, name)] };
    });
  }, [updateTraining]);

  const setDefaultRest = useCallback((seconds: number) => {
    updateTraining((current) => ({ ...current, defaultRestSec: seconds }));
  }, [updateTraining]);

  const setRpeEnabled = useCallback((enabled: boolean) => {
    updateTraining((current) => ({ ...current, rpeEnabled: enabled }));
  }, [updateTraining]);

  const value = useMemo<WorkoutContextValue>(() => ({
    draftRef,
    beginRoutineDraft,
    saveRoutineDraft,
    deleteRoutine,
    duplicateRoutine,
    importTemplateRoutine,
    createCustomExercise,
    updateCustomExercise,
    deleteCustomExercise,
    startEmptyWorkout,
    startRoutine,
    resumeOrStartFromRoutine,
    updateActiveSession,
    updateLiveMetricsFromWatch,
    upsertActiveWorkoutFromWatch,
    completeSet,
    discardActiveWorkout,
    finishActiveWorkout,
    deleteSession,
    saveSessionAsRoutine,
    setDefaultRest,
    setRpeEnabled,
  }), [
    beginRoutineDraft,
    saveRoutineDraft,
    deleteRoutine,
    duplicateRoutine,
    importTemplateRoutine,
    createCustomExercise,
    updateCustomExercise,
    deleteCustomExercise,
    startEmptyWorkout,
    startRoutine,
    resumeOrStartFromRoutine,
    updateActiveSession,
    updateLiveMetricsFromWatch,
    upsertActiveWorkoutFromWatch,
    completeSet,
    discardActiveWorkout,
    finishActiveWorkout,
    deleteSession,
    saveSessionAsRoutine,
    setDefaultRest,
    setRpeEnabled,
  ]);

  return <WorkoutContext.Provider value={value}>{children}</WorkoutContext.Provider>;
}

export function useWorkouts() {
  const value = useContext(WorkoutContext);
  if (!value) throw new Error('useWorkouts must be used inside WorkoutProvider');
  return value;
}

export function makeSessionExerciseEntries(exerciseIds: string[], defaultRestSec: number) {
  return exerciseIds.map((exerciseId) => makeSessionExercise(exerciseId, defaultRestSec));
}
