import Storage from 'expo-sqlite/kv-store';
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { dateKey } from '@/src/lib/date';
import type { AppData, DayLog, Goals, HealthSnapshot, LogOperation, MealItem, WeightPoint, Workout } from '@/src/types';

const STORAGE_KEY = 'calorie-lens.app-data.v1';

const initialData: AppData = {
  goals: { calories: 2200, protein: 120, waterMl: 3000, steps: 8000 },
  days: {},
  weights: [],
};

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

function id(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

type AppContextValue = {
  data: AppData;
  today: DayLog;
  hydrated: boolean;
  applyOperations: (operations: LogOperation[]) => void;
  addWater: (amount: number) => void;
  removeMeal: (id: string) => void;
  removeWorkout: (id: string) => void;
  applyHealthSnapshot: (snapshot: HealthSnapshot) => void;
  updateGoals: (goals: Partial<Goals>) => void;
};

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: React.PropsWithChildren) {
  const [data, setData] = useState<AppData>(initialData);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    Storage.getItem(STORAGE_KEY)
      .then((saved) => {
        if (saved) setData(JSON.parse(saved));
      })
      .finally(() => setHydrated(true));
  }, []);

  useEffect(() => {
    if (hydrated) void Storage.setItem(STORAGE_KEY, JSON.stringify(data));
  }, [data, hydrated]);

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
    updateToday((day) => {
      const next = { ...day, meals: [...day.meals], workouts: [...day.workouts] };
      for (const operation of operations) {
        if (operation.type === 'meal') {
          const items: MealItem[] = operation.items.map((item) => ({
            ...item,
            id: id('meal'),
            slot: operation.slot,
            loggedAt: new Date().toISOString(),
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
      setData((current) => ({
        ...current,
        weights: [
          ...current.weights.filter((point) => !weightPoints.some((next) => next.date === point.date)),
          ...weightPoints,
        ].slice(-90),
      }));
    }
  }, [updateToday]);

  const addWater = useCallback((amount: number) => {
    applyOperations([{ type: 'water', action: 'add', amount }]);
  }, [applyOperations]);

  const removeMeal = useCallback((mealId: string) => {
    updateToday((day) => ({ ...day, meals: day.meals.filter((meal) => meal.id !== mealId) }));
  }, [updateToday]);

  const removeWorkout = useCallback((workoutId: string) => {
    updateToday((day) => ({ ...day, workouts: day.workouts.filter((workout) => workout.id !== workoutId) }));
  }, [updateToday]);

  const applyHealthSnapshot = useCallback((snapshot: HealthSnapshot) => {
    updateToday((day) => ({
      ...day,
      steps: snapshot.steps ?? day.steps,
      activeCalories: snapshot.activeCalories ?? day.activeCalories,
      sleepHours: snapshot.sleepHours ?? day.sleepHours,
    }));
    setData((current) => ({
      ...current,
      lastHealthSync: new Date().toISOString(),
      weights: snapshot.weightKg
        ? [
            ...current.weights.filter((point) => point.date !== dateKey()),
            { date: dateKey(), kg: snapshot.weightKg },
          ].slice(-90)
        : current.weights,
    }));
  }, [updateToday]);

  const updateGoals = useCallback((goals: Partial<Goals>) => {
    setData((current) => ({ ...current, goals: { ...current.goals, ...goals } }));
  }, []);

  const today = data.days[dateKey()] ?? emptyDay();
  const value = useMemo(() => ({
    data,
    today,
    hydrated,
    applyOperations,
    addWater,
    removeMeal,
    removeWorkout,
    applyHealthSnapshot,
    updateGoals,
  }), [data, today, hydrated, applyOperations, addWater, removeMeal, removeWorkout, applyHealthSnapshot, updateGoals]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const value = useContext(AppContext);
  if (!value) throw new Error('useApp must be used inside AppProvider');
  return value;
}
