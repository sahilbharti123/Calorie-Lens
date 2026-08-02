import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { ActivityIndicator, AppState, View } from 'react-native';

import { dateKey } from '@/src/lib/date';
import { healthMetricHasSamples, healthSnapshotHasSamples } from '@/src/lib/health';
import { mergeSavedMeals, savedMealFromGroup, type MealGroup } from '@/src/lib/meals';
import { calculatePersonalTargets } from '@/src/lib/personalization';
import {
  readEncryptedJson,
  removeEncryptedJson,
  takeUnreadableRecords,
  writeEncryptedJson,
} from '@/src/lib/secure-storage';
import {
  CloudConflictError,
  readCloudSnapshot,
  writeCloudSnapshot,
} from '@/src/lib/supabase-sync';
import { initialTraining, mergeTraining, normalizeTraining } from '@/src/lib/training';
import { useAuth } from '@/src/store/auth-store';
import { palette } from '@/src/theme';
import type {
  AppData,
  CoachMemory,
  CoachMessage,
  DayLog,
  EstimationProfile,
  Goals,
  HealthSnapshot,
  LogOperation,
  MealItem,
  PersonalProfile,
  SyncState,
  TrainingData,
  WeightPoint,
  Workout,
} from '@/src/types';

/**
 * Persisted-storage keys keep the pre-rename `calorie-lens.` prefix on purpose.
 * They identify data already written to the device — renaming them would
 * orphan every existing vault, session and backup rather than migrate it.
 * The name a user sees is set in app.json; these are invisible.
 */
const LEGACY_STORAGE_KEY = 'calorie-lens.app-data.v1';
const STORAGE_PREFIX = 'calorie-lens.encrypted-app-data.v2';
/** Coalesces a burst of keystrokes into one encrypt + write. See flushLocalWrite. */
const LOCAL_WRITE_DEBOUNCE_MS = 300;

type StoredEnvelope = {
  data: AppData;
  cloudVersion: number;
};

const initialCoachMemory: CoachMemory = {
  dietaryPreferences: [],
  injuries: [],
  workoutPreferences: [],
  coachingStyle: 'supportive and concise',
  notes: '',
  updatedAt: '',
};

const initialProfile: PersonalProfile = {
  allergies: [],
  injuries: [],
  updatedAt: '',
};

export const initialData: AppData = {
  goals: {
    calories: 2200,
    protein: 120,
    carbs: 250,
    fat: 70,
    waterMl: 3000,
    steps: 8000,
    weeklyWorkoutMinutes: 150,
    strengthDays: 2,
  },
  profile: initialProfile,
  plan: {
    method: 'Complete onboarding to calculate a personal starting point',
    summary: 'General starter targets',
    warnings: [],
    updatedAt: '',
  },
  estimation: { cupMl: 200 },
  days: {},
  weights: [],
  coachMemory: initialCoachMemory,
  coachMessages: [],
  deletedMealIds: [],
  deletedWorkoutIds: [],
  deletedSavedMealIds: [],
  savedMeals: [],
  training: initialTraining,
};

export function normalizeData(saved?: Partial<AppData> | null): AppData {
  return {
    ...initialData,
    ...(saved ?? {}),
    goals: { ...initialData.goals, ...(saved?.goals ?? {}) },
    profile: {
      ...initialProfile,
      ...(saved?.profile ?? {}),
      allergies: saved?.profile?.allergies ?? [],
      injuries: saved?.profile?.injuries ?? [],
    },
    plan: { ...initialData.plan, ...(saved?.plan ?? {}) },
    estimation: { ...initialData.estimation, ...(saved?.estimation ?? {}) },
    days: saved?.days ?? {},
    weights: saved?.weights ?? [],
    coachMemory: { ...initialCoachMemory, ...(saved?.coachMemory ?? {}) },
    coachMessages: saved?.coachMessages ?? [],
    deletedMealIds: saved?.deletedMealIds ?? [],
    deletedWorkoutIds: saved?.deletedWorkoutIds ?? [],
    deletedSavedMealIds: saved?.deletedSavedMealIds ?? [],
    savedMeals: saved?.savedMeals ?? [],
    healthSync: saved?.healthSync ?? (saved?.lastHealthSync ? {
      status: 'current',
      lastAttemptAt: saved.lastHealthSync,
      lastSuccessAt: saved.lastHealthSync,
      message: 'Health data was updated on this device.',
    } : undefined),
    training: normalizeTraining(saved?.training),
  };
}

function emptyDay(date = dateKey()): DayLog {
  return {
    date,
    meals: [],
    workouts: [],
    waterMl: 0,
    steps: 0,
    activeCalories: 0,
    sleepHours: 0,
  };
}

function withCurrentWeight(current: AppData, weightKg: number): AppData {
  const updatedAt = new Date().toISOString();
  const weights = [
    ...current.weights.filter((point) => point.date !== dateKey()),
    { date: dateKey(), kg: weightKg },
  ].slice(-365);
  if (!current.profile.primaryGoal) return { ...current, weights };
  const profile = { ...current.profile, weightKg, updatedAt };
  if (current.plan.method === 'Manually adjusted in profile settings') {
    return { ...current, profile, weights };
  }
  const { goals, plan } = calculatePersonalTargets(profile);
  return {
    ...current,
    goals,
    profile,
    plan: { ...plan, updatedAt },
    weights,
  };
}

function id(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function storageKey(scope: string) {
  return `${STORAGE_PREFIX}.${scope}`;
}

function isEnvelope(value: unknown): value is StoredEnvelope {
  return Boolean(
    value
    && typeof value === 'object'
    && 'data' in value
    && 'cloudVersion' in value,
  );
}

function isPristine(data: AppData) {
  return !Object.keys(data.days).length
    && !data.weights.length
    && !data.coachMessages.length
    && JSON.stringify(data.goals) === JSON.stringify(initialData.goals)
    && !data.profile.primaryGoal
    && JSON.stringify(data.estimation) === JSON.stringify(initialData.estimation)
    && !data.coachMemory.notes
    && !data.coachMemory.dietaryPreferences.length
    && !data.coachMemory.injuries.length
    && !data.coachMemory.workoutPreferences.length
    && !data.deletedMealIds.length
    && !data.deletedWorkoutIds.length
    && !data.deletedSavedMealIds.length
    && !data.savedMeals.length
    && !data.training.routines.length
    && !data.training.sessions.length
    && !data.training.activeSession;
}

function byNewest<T extends { updatedAt: string }>(local: T, remote: T) {
  if (!local.updatedAt) return remote;
  if (!remote.updatedAt) return local;
  return local.updatedAt >= remote.updatedAt ? local : remote;
}

function unionById<T extends { id: string }>(left: T[], right: T[]) {
  const values = new Map<string, T>();
  for (const value of [...left, ...right]) values.set(value.id, value);
  return [...values.values()];
}

export function mergeAppData(localInput: Partial<AppData>, remoteInput: Partial<AppData>) {
  const local = normalizeData(localInput);
  const remote = normalizeData(remoteInput);
  if (isPristine(local)) return remote;
  if (isPristine(remote)) return local;
  const deletedMealIds = [...new Set([...local.deletedMealIds, ...remote.deletedMealIds])];
  const deletedWorkoutIds = [...new Set([
    ...local.deletedWorkoutIds,
    ...remote.deletedWorkoutIds,
  ])];
  const deletedSavedMealIds = [...new Set([
    ...local.deletedSavedMealIds,
    ...remote.deletedSavedMealIds,
  ])];
  const dates = new Set([...Object.keys(remote.days), ...Object.keys(local.days)]);
  const days: Record<string, DayLog> = {};
  for (const date of dates) {
    const localDay = local.days[date];
    const remoteDay = remote.days[date];
    if (!localDay) {
      days[date] = {
        ...remoteDay,
        meals: remoteDay.meals.filter((meal) => !deletedMealIds.includes(meal.id)),
        workouts: remoteDay.workouts.filter((workout) => !deletedWorkoutIds.includes(workout.id)),
      };
      continue;
    }
    if (!remoteDay) {
      days[date] = {
        ...localDay,
        meals: localDay.meals.filter((meal) => !deletedMealIds.includes(meal.id)),
        workouts: localDay.workouts.filter((workout) => !deletedWorkoutIds.includes(workout.id)),
      };
      continue;
    }
    days[date] = {
      ...remoteDay,
      ...localDay,
      meals: unionById(remoteDay.meals, localDay.meals)
        .filter((meal) => !deletedMealIds.includes(meal.id)),
      workouts: unionById(remoteDay.workouts, localDay.workouts)
        .filter((workout) => !deletedWorkoutIds.includes(workout.id)),
      // Water accumulates, so the larger figure is the more complete one.
      waterMl: Math.max(remoteDay.waterMl, localDay.waterMl),
      // Steps, active energy and sleep are *set* rather than added — by a health
      // sync or by the user correcting a mistyped figure. Taking the maximum
      // silently reverted every correction downwards, so the local value wins.
      steps: localDay.steps || remoteDay.steps,
      activeCalories: localDay.activeCalories || remoteDay.activeCalories,
      sleepHours: localDay.sleepHours || remoteDay.sleepHours,
    };
  }
  const weights = new Map<string, WeightPoint>();
  for (const point of [...remote.weights, ...local.weights]) weights.set(point.date, point);
  const messages = unionById(remote.coachMessages, local.coachMessages)
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt))
    .slice(-60);
  return normalizeData({
    ...remote,
    ...local,
    days,
    weights: [...weights.values()].sort((a, b) => a.date.localeCompare(b.date)).slice(-365),
    profile: byNewest(local.profile, remote.profile),
    plan: byNewest(local.plan, remote.plan),
    coachMemory: byNewest(local.coachMemory, remote.coachMemory),
    coachMessages: messages,
    deletedMealIds,
    deletedWorkoutIds,
    deletedSavedMealIds,
    savedMeals: mergeSavedMeals(local.savedMeals, remote.savedMeals, deletedSavedMealIds),
    training: mergeTraining(local.training, remote.training),
    lastHealthSync: [local.lastHealthSync, remote.lastHealthSync]
      .filter((value): value is string => Boolean(value))
      .sort()
      .at(-1),
    healthSync: [local.healthSync, remote.healthSync]
      .filter((value): value is NonNullable<AppData['healthSync']> => Boolean(value))
      .sort((a, b) => a.lastAttemptAt.localeCompare(b.lastAttemptAt))
      .at(-1),
  });
}

type AppContextValue = {
  data: AppData;
  today: DayLog;
  hydrated: boolean;
  /** Local data existed but this device could not decrypt it, so it was reset. */
  vaultReset: boolean;
  dismissVaultReset: () => void;
  syncState: SyncState;
  syncError: string;
  applyOperations: (operations: LogOperation[]) => void;
  addWater: (amount: number) => void;
  removeMeal: (id: string) => void;
  removeWorkout: (id: string) => void;
  saveMeal: (group: MealGroup) => void;
  removeSavedMeal: (id: string) => void;
  applyHealthSnapshot: (snapshot: HealthSnapshot) => void;
  reportHealthSyncError: (message: string) => void;
  updateGoals: (goals: Partial<Goals>) => void;
  savePersonalization: (profile: PersonalProfile, bowlMl?: number) => void;
  updateEstimationProfile: (profile: Partial<EstimationProfile>, weightKg?: number) => void;
  updateCoachMemory: (memory: Partial<CoachMemory>) => void;
  addCoachMessage: (message: Omit<CoachMessage, 'id' | 'createdAt'>) => CoachMessage;
  updateTraining: (recipe: (training: TrainingData) => TrainingData) => void;
  replaceData: (data: Partial<AppData>) => void;
  syncNow: () => Promise<void>;
  clearLocalData: () => Promise<void>;
};

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: React.PropsWithChildren) {
  const { session } = useAuth();
  const scope = session?.user.id ?? 'guest';
  const [data, setData] = useState<AppData>(initialData);
  const [hydrated, setHydrated] = useState(false);
  /** True when this launch found local data it could not decrypt and reset it. */
  const [vaultReset, setVaultReset] = useState(false);
  const [cloudVersion, setCloudVersion] = useState(0);
  const [syncState, setSyncState] = useState<SyncState>('offline');
  const [syncError, setSyncError] = useState('');
  const dataRef = useRef(data);
  const versionRef = useRef(cloudVersion);
  const syncingRef = useRef(false);
  /** Newest vault snapshot still owed to disk, or null when nothing is pending. */
  const pendingWriteRef = useRef<{ key: string; envelope: StoredEnvelope } | null>(null);

  useEffect(() => {
    dataRef.current = data;
  }, [data]);
  useEffect(() => {
    versionRef.current = cloudVersion;
  }, [cloudVersion]);

  useEffect(() => {
    let active = true;
    setHydrated(false);
    setSyncState(session ? 'syncing' : 'offline');
    async function load() {
      const key = storageKey(scope);
      let saved = await readEncryptedJson<StoredEnvelope | AppData>(key);
      if (!saved && !session) {
        saved = await readEncryptedJson<AppData>(LEGACY_STORAGE_KEY);
      }
      if (!saved && session) {
        saved = await readEncryptedJson<StoredEnvelope | AppData>(storageKey('guest'))
          ?? await readEncryptedJson<AppData>(LEGACY_STORAGE_KEY);
      }
      const localData = normalizeData(isEnvelope(saved) ? saved.data : saved);
      let nextData = localData;
      let nextVersion = isEnvelope(saved) ? saved.cloudVersion : 0;
      if (session) {
        try {
          const remote = await readCloudSnapshot(session.user.id);
          nextVersion = remote.version;
          const hasRemote = remote.payload && Object.keys(remote.payload).length > 0;
          nextData = hasRemote ? mergeAppData(localData, remote.payload) : localData;
          let uploaded;
          try {
            uploaded = await writeCloudSnapshot(nextData, nextVersion);
          } catch (error) {
            if (!(error instanceof CloudConflictError)) throw error;
            nextData = mergeAppData(nextData, error.snapshot.payload);
            uploaded = await writeCloudSnapshot(nextData, error.snapshot.version);
          }
          nextVersion = uploaded.version;
          setSyncState('synced');
        } catch (error) {
          setSyncState('error');
          setSyncError(error instanceof Error ? error.message : 'Cloud sync failed.');
        }
      }
      if (!active) return;
      setData(nextData);
      setCloudVersion(nextVersion);
      await writeEncryptedJson(key, { data: nextData, cloudVersion: nextVersion });
    }

    // Hydration must always finish. Anything thrown here — an unreadable local
    // vault, a storage failure — would otherwise leave the app on its loading
    // state forever, which is a worse outcome than starting empty.
    void load()
      .catch((error) => {
        if (!active) return;
        setSyncState('error');
        setSyncError(
          error instanceof Error ? error.message : 'Local data could not be opened.',
        );
      })
      .finally(() => {
        if (!active) return;
        setVaultReset(takeUnreadableRecords().length > 0);
        setHydrated(true);
      });
    return () => {
      active = false;
    };
    // `scope` deliberately, not `session`: supabase mints a new session object
    // on every token refresh, and depending on it re-ran hydration mid-use —
    // which flips `hydrated` back to false and unmounts the whole navigation
    // stack, dropping the user out of whatever screen they were on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope]);

  const dismissVaultReset = useCallback(() => setVaultReset(false), []);

  const syncNow = useCallback(async () => {
    if (!session || !hydrated || syncingRef.current) {
      if (!session) setSyncState('offline');
      return;
    }
    syncingRef.current = true;
    setSyncState('syncing');
    setSyncError('');
    try {
      let response;
      let nextData = dataRef.current;
      try {
        response = await writeCloudSnapshot(nextData, versionRef.current);
      } catch (error) {
        if (!(error instanceof CloudConflictError)) throw error;
        nextData = mergeAppData(nextData, error.snapshot.payload);
        setData(nextData);
        dataRef.current = nextData;
        response = await writeCloudSnapshot(nextData, error.snapshot.version);
      }
      setCloudVersion(response.version);
      versionRef.current = response.version;
      setSyncState('synced');
    } catch (error) {
      setSyncState('error');
      setSyncError(error instanceof Error ? error.message : 'Cloud sync failed.');
    } finally {
      syncingRef.current = false;
    }
  }, [session, hydrated]);

  /** Writes the newest pending snapshot, if any. Safe to call any number of times. */
  const flushLocalWrite = useCallback(() => {
    const pending = pendingWriteRef.current;
    if (!pending) return;
    pendingWriteRef.current = null;
    void writeEncryptedJson(pending.key, pending.envelope);
  }, []);

  /**
   * Local persistence, debounced. Every keystroke in a weight, reps or workout
   * name field replaces `data`, and an undebounced write re-encrypts the entire
   * vault (stringify → AES-GCM → hex → SQLite) on the JS thread each time.
   *
   * The debounce can delay a write but never drop one. `pendingWriteRef` always
   * holds the newest snapshot, and because `data` is one immutable object every
   * snapshot is a superset of the ones it replaced — skipping intermediate
   * snapshots loses nothing. It is flushed on all four exits:
   *   1. the debounce timer fires (the normal path);
   *   2. the storage key changes (sign in/out), before the ref is repointed;
   *   3. the provider unmounts (cleanup of the AppState effect below);
   *   4. the app leaves the foreground — AppState 'inactive'/'background' are
   *      delivered while JS is still running, before iOS/Android can kill the
   *      process, so swiping the app away mid-workout still lands the last set.
   * `clearLocalData` drops the pending snapshot so a stale write cannot
   * resurrect a vault the user just erased.
   */
  useEffect(() => {
    if (!hydrated) return;
    const key = storageKey(scope);
    if (pendingWriteRef.current && pendingWriteRef.current.key !== key) flushLocalWrite();
    pendingWriteRef.current = { key, envelope: { data, cloudVersion } };
    const timer = setTimeout(flushLocalWrite, LOCAL_WRITE_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [data, cloudVersion, hydrated, scope, flushLocalWrite]);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      if (state === 'background' || state === 'inactive') flushLocalWrite();
    });
    return () => {
      subscription.remove();
      flushLocalWrite();
    };
  }, [flushLocalWrite]);

  useEffect(() => {
    if (!hydrated) return;
    if (!session) return;
    const timer = setTimeout(() => void syncNow(), 1200);
    return () => clearTimeout(timer);
  }, [data, hydrated, session, syncNow]);

  const updateToday = useCallback((recipe: (day: DayLog) => DayLog) => {
    const todayKey = dateKey();
    setData((current) => ({
      ...current,
      days: {
        ...current.days,
        [todayKey]: recipe(current.days[todayKey] ?? emptyDay(todayKey)),
      },
    }));
  }, []);

  const applyOperations = useCallback((operations: LogOperation[]) => {
    const weightPoints: WeightPoint[] = [];
    const loggedAt = new Date().toISOString();
    updateToday((day) => {
      const next = { ...day, meals: [...day.meals], workouts: [...day.workouts] };
      for (const operation of operations) {
        if (operation.type === 'meal') {
          const items: MealItem[] = operation.items.map((item) => ({
            ...item,
            id: id('meal'),
            slot: operation.slot,
            loggedAt,
          }));
          next.meals.push(...items);
        } else if (operation.type === 'water') {
          next.waterMl = operation.action === 'set'
            ? Math.max(0, operation.amount)
            : Math.max(0, next.waterMl + operation.amount);
        } else if (operation.type === 'steps') {
          next.steps = operation.action === 'set'
            ? Math.max(0, operation.amount)
            : Math.max(0, next.steps + operation.amount);
        } else if (operation.type === 'sleep') {
          next.sleepHours = Math.max(0, operation.amount);
        } else if (operation.type === 'workout') {
          const workout: Workout = {
            id: id('workout'),
            name: operation.name,
            durationMin: operation.durationMin,
            calories: operation.calories,
            calorieLow: operation.calorieLow,
            calorieHigh: operation.calorieHigh,
            confidence: operation.confidence,
            basis: operation.basis,
            sourceLabel: operation.sourceLabel,
            sourceId: operation.sourceId,
            met: operation.met,
            intensity: operation.intensity,
            loggedAt: new Date().toISOString(),
          };
          next.workouts.push(workout);
        } else if (operation.type === 'weight') {
          weightPoints.push({ date: dateKey(), kg: operation.amount });
        }
      }
      return next;
    });
    if (weightPoints.length) {
      setData((current) => withCurrentWeight(current, weightPoints.at(-1)!.kg));
    }
  }, [updateToday]);

  const addWater = useCallback((amount: number) => {
    applyOperations([{ type: 'water', action: 'add', amount }]);
  }, [applyOperations]);

  const removeMeal = useCallback((mealId: string) => {
    updateToday((day) => ({ ...day, meals: day.meals.filter((meal) => meal.id !== mealId) }));
    setData((current) => ({
      ...current,
      deletedMealIds: [...new Set([...current.deletedMealIds, mealId])].slice(-1000),
    }));
  }, [updateToday]);

  const removeWorkout = useCallback((workoutId: string) => {
    updateToday((day) => ({
      ...day,
      workouts: day.workouts.filter((workout) => workout.id !== workoutId),
    }));
    setData((current) => ({
      ...current,
      deletedWorkoutIds: [...new Set([...current.deletedWorkoutIds, workoutId])].slice(-1000),
    }));
  }, [updateToday]);

  const saveMeal = useCallback((group: MealGroup) => {
    setData((current) => {
      const existing = current.savedMeals.find((meal) => (
        meal.slot === group.slot
        && JSON.stringify(meal.items.map((item) => [item.name, item.quantity]).sort())
          === JSON.stringify(group.items.map((item) => [item.name, item.quantity]).sort())
      ));
      const now = new Date().toISOString();
      if (existing) {
        return {
          ...current,
          savedMeals: current.savedMeals.map((meal) => (
            meal.id === existing.id ? { ...meal, updatedAt: now } : meal
          )),
        };
      }
      return {
        ...current,
        savedMeals: [savedMealFromGroup(group, id('saved-meal'), now), ...current.savedMeals]
          .slice(0, 50),
      };
    });
  }, []);

  const removeSavedMeal = useCallback((savedMealId: string) => {
    setData((current) => ({
      ...current,
      savedMeals: current.savedMeals.filter((meal) => meal.id !== savedMealId),
      deletedSavedMealIds: [...new Set([...current.deletedSavedMealIds, savedMealId])].slice(-500),
    }));
  }, []);

  const applyHealthSnapshot = useCallback((snapshot: HealthSnapshot) => {
    updateToday((day) => ({
      ...day,
      steps: healthMetricHasSamples(snapshot, 'steps') ? (snapshot.steps ?? day.steps) : day.steps,
      activeCalories: healthMetricHasSamples(snapshot, 'activeCalories')
        ? (snapshot.activeCalories ?? day.activeCalories)
        : day.activeCalories,
      sleepHours: healthMetricHasSamples(snapshot, 'sleep')
        ? (snapshot.sleepHours ?? day.sleepHours)
        : day.sleepHours,
    }));
    setData((current) => {
      const now = new Date().toISOString();
      const hasSamples = healthSnapshotHasSamples(snapshot);
      const synced = {
        ...current,
        lastHealthSync: now,
        healthSync: {
          source: snapshot.source,
          status: hasSamples ? 'current' as const : 'empty' as const,
          lastAttemptAt: now,
          lastSuccessAt: now,
          message: hasSamples
            ? `Updated from ${snapshot.source}.`
            : `Connected to ${snapshot.source}, but no shared samples were found.`,
        },
      };
      return healthMetricHasSamples(snapshot, 'weight') && snapshot.weightKg
        ? withCurrentWeight(synced, snapshot.weightKg)
        : synced;
    });
  }, [updateToday]);

  const reportHealthSyncError = useCallback((message: string) => {
    setData((current) => ({
      ...current,
      healthSync: {
        ...current.healthSync,
        status: 'error',
        lastAttemptAt: new Date().toISOString(),
        message,
      },
    }));
  }, []);

  const updateGoals = useCallback((goals: Partial<Goals>) => {
    setData((current) => {
      const nextGoals = { ...current.goals, ...goals };
      return {
        ...current,
        goals: nextGoals,
        plan: {
          ...current.plan,
          method: 'Manually adjusted in profile settings',
          summary: `Manual targets · ${nextGoals.calories} kcal · ${nextGoals.protein} g protein`,
          updatedAt: new Date().toISOString(),
        },
      };
    });
  }, []);

  const savePersonalization = useCallback((profile: PersonalProfile, bowlMl?: number) => {
    const updatedAt = new Date().toISOString();
    const nextProfile = { ...profile, updatedAt };
    const { goals, plan } = calculatePersonalTargets(nextProfile);
    setData((current) => ({
      ...current,
      goals,
      profile: nextProfile,
      plan: { ...plan, updatedAt },
      estimation: {
        ...current.estimation,
        bowlMl: bowlMl && bowlMl >= 50 ? Math.min(1000, bowlMl) : current.estimation.bowlMl,
        cupMl: current.estimation.cupMl || 200,
      },
      weights: nextProfile.weightKg
        ? [
            ...current.weights.filter((point) => point.date !== dateKey()),
            { date: dateKey(), kg: nextProfile.weightKg },
          ].slice(-365)
        : current.weights,
    }));
  }, []);

  const updateEstimationProfile = useCallback((profile: Partial<EstimationProfile>, weightKg?: number) => {
    setData((current) => {
      const calibrated = {
        ...current,
        estimation: { ...current.estimation, ...profile },
      };
      return weightKg ? withCurrentWeight(calibrated, weightKg) : calibrated;
    });
  }, []);

  const updateCoachMemory = useCallback((memory: Partial<CoachMemory>) => {
    setData((current) => ({
      ...current,
      coachMemory: {
        ...current.coachMemory,
        ...memory,
        updatedAt: new Date().toISOString(),
      },
    }));
  }, []);

  const updateTraining = useCallback((recipe: (training: TrainingData) => TrainingData) => {
    setData((current) => ({
      ...current,
      training: {
        ...recipe(current.training),
        updatedAt: new Date().toISOString(),
      },
    }));
  }, []);

  const addCoachMessage = useCallback((message: Omit<CoachMessage, 'id' | 'createdAt'>) => {
    const next: CoachMessage = {
      ...message,
      id: id('coach'),
      createdAt: new Date().toISOString(),
    };
    setData((current) => ({
      ...current,
      coachMessages: [...current.coachMessages, next].slice(-60),
    }));
    return next;
  }, []);

  const replaceData = useCallback((replacement: Partial<AppData>) => {
    setData(normalizeData(replacement));
  }, []);

  const clearLocalData = useCallback(async () => {
    pendingWriteRef.current = null;
    await removeEncryptedJson(storageKey(scope));
    setData(initialData);
    dataRef.current = initialData;
    if (!session) {
      setCloudVersion(0);
      versionRef.current = 0;
    }
  }, [scope, session]);

  const today = data.days[dateKey()] ?? emptyDay();
  const value = useMemo<AppContextValue>(() => ({
    data,
    today,
    hydrated,
    vaultReset,
    dismissVaultReset,
    syncState,
    syncError,
    applyOperations,
    addWater,
    removeMeal,
    removeWorkout,
    saveMeal,
    removeSavedMeal,
    applyHealthSnapshot,
    reportHealthSyncError,
    updateGoals,
    savePersonalization,
    updateEstimationProfile,
    updateCoachMemory,
    addCoachMessage,
    updateTraining,
    replaceData,
    syncNow,
    clearLocalData,
  }), [
    data,
    today,
    hydrated,
    vaultReset,
    dismissVaultReset,
    syncState,
    syncError,
    applyOperations,
    addWater,
    removeMeal,
    removeWorkout,
    saveMeal,
    removeSavedMeal,
    applyHealthSnapshot,
    reportHealthSyncError,
    updateGoals,
    savePersonalization,
    updateEstimationProfile,
    updateCoachMemory,
    addCoachMessage,
    updateTraining,
    replaceData,
    syncNow,
    clearLocalData,
  ]);

  return (
    <AppContext.Provider value={value}>
      {hydrated ? children : (
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.bg }}>
          <ActivityIndicator color={palette.lime} />
        </View>
      )}
    </AppContext.Provider>
  );
}

export function useApp() {
  const value = useContext(AppContext);
  if (!value) throw new Error('useApp must be used inside AppProvider');
  return value;
}
