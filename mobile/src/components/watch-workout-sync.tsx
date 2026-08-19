import { useRouter } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AppState, Platform } from 'react-native';

import {
  startWorkoutOnAppleWatch,
  recordAppleWatchSyncReceipt,
  subscribeToAppleWatchWorkout,
  syncTrainingToWatch,
  workoutSessionFromWatch,
  type LiveWorkoutEvent,
} from '@/src/lib/live-workout';
import { isWorkoutSnapshotNewer } from '@/src/lib/watch-workout-contract';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';

/**
 * Keeps Watch connectivity alive at the app-shell level. A Watch-started
 * workout must be accepted even when the phone is not on the workout screen.
 */
export function WatchWorkoutSync() {
  const router = useRouter();
  const { data } = useApp();
  const workouts = useWorkouts();
  const training = data.training;
  const trainingRef = useRef(training);
  trainingRef.current = training;
  const sessionRef = useRef(training.activeSession);
  sessionRef.current = training.activeSession;
  const handledEvents = useRef(new Set<string>());
  const autoLaunchRef = useRef<{ sessionId: string; at: number } | null>(null);
  const [finishRequest, setFinishRequest] = useState<string | null>(null);

  const refreshWatch = useCallback((launchActiveWorkout: boolean) => {
    if (Platform.OS !== 'ios') return;
    const latest = trainingRef.current;
    syncTrainingToWatch(latest);
    const active = latest.activeSession;
    if (!launchActiveWorkout || !active) return;

    const now = Date.now();
    const prior = autoLaunchRef.current;
    if (prior?.sessionId === active.id && now - prior.at < 15_000) return;
    autoLaunchRef.current = { sessionId: active.id, at: now };
    void startWorkoutOnAppleWatch(active, latest).catch(() => {
      if (autoLaunchRef.current?.sessionId === active.id) autoLaunchRef.current = null;
    });
  }, []);

  const syncKey = useMemo(() => JSON.stringify({
    routines: training.routines.map((routine) => [routine.id, routine.updatedAt]),
    active: training.activeSession ? {
      id: training.activeSession.id,
      name: training.activeSession.name,
      revision: training.activeSession.watchRevision,
      exercises: training.activeSession.exercises.map((entry) => ({
        id: entry.id,
        restSec: entry.restSec,
        sets: entry.sets.map((set) => [
          set.id,
          set.type,
          set.weightKg,
          set.reps,
          set.durationSec,
          set.completed,
        ]),
      })),
    } : null,
  }), [training.activeSession, training.routines]);

  useEffect(() => {
    if (Platform.OS === 'ios') syncTrainingToWatch(training);
  }, [syncKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // A normal app launch cannot force-open an arbitrary Watch app. Apple only
  // allows that handoff for an actual workout, so foregrounding always refreshes
  // state and automatically launches the Watch when a live session exists.
  useEffect(() => {
    refreshWatch(Boolean(training.activeSession));
  }, [refreshWatch, training.activeSession?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      if (state === 'active') refreshWatch(Boolean(trainingRef.current.activeSession));
    });
    return () => subscription.remove();
  }, [refreshWatch]);

  useEffect(() => {
    const subscription = subscribeToAppleWatchWorkout((event: LiveWorkoutEvent) => {
      if ('eventId' in event && event.eventId) {
        if (handledEvents.current.has(event.eventId)) return;
        handledEvents.current.add(event.eventId);
        if (handledEvents.current.size > 100) {
          handledEvents.current = new Set([...handledEvents.current].slice(-50));
        }
      }

      if (event.type === 'metrics') {
        workouts.updateLiveMetricsFromWatch({
          heartRateBpm: event.heartRateBpm,
          activeCalories: event.activeCalories,
          source: 'Apple Watch',
          updatedAt: event.updatedAt ?? new Date().toISOString(),
        });
        return;
      }

      if (event.type === 'syncReceipt') {
        recordAppleWatchSyncReceipt(event.updatedAt, event.reachable);
        return;
      }

      if (event.type === 'sessionStarted' || event.type === 'sessionUpdated' || event.type === 'sessionFinished') {
        const session = workoutSessionFromWatch(event.session);
        const current = sessionRef.current;
        if (!isWorkoutSnapshotNewer(session, current)) {
          if (current && isWorkoutSnapshotNewer(current, session)) {
            syncTrainingToWatch(trainingRef.current);
          }
          return;
        }
        workouts.upsertActiveWorkoutFromWatch(session);
        if (event.type === 'sessionStarted') {
          router.push('/workout-session');
        }
        if (event.type === 'sessionFinished') {
          setFinishRequest(session.id);
        }
        return;
      }

      // Accept set toggles from the earlier metrics-only Watch build during the
      // rollout. New builds always send the complete session snapshot above.
      if (event.type === 'set') {
        const current = sessionRef.current;
        const entry = current?.exercises.find((candidate) => candidate.id === event.sessionExerciseId);
        const set = entry?.sets.find((candidate) => candidate.id === event.setId);
        if (!entry || !set || set.completed === event.completed) return;
        if (event.completed) workouts.completeSet(entry.id, set.id);
        else workouts.updateActiveSession((session) => ({
          ...session,
          exercises: session.exercises.map((candidate) => candidate.id !== entry.id ? candidate : {
            ...candidate,
            sets: candidate.sets.map((candidateSet) => candidateSet.id === set.id
              ? { ...candidateSet, completed: false, prFlags: undefined }
              : candidateSet),
          }),
        }));
      }
    });
    return () => subscription?.remove();
  }, [router, workouts]);

  useEffect(() => {
    if (!finishRequest || training.activeSession?.id !== finishRequest) return;
    const finishedId = workouts.finishActiveWorkout();
    setFinishRequest(null);
    if (finishedId) {
      router.replace({ pathname: '/workout/[id]', params: { id: finishedId, celebrate: '1' } });
    }
  }, [finishRequest, router, training.activeSession?.id, workouts]);

  return null;
}
