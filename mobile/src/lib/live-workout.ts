import {
  NativeEventEmitter,
  NativeModules,
  Platform,
  type EmitterSubscription,
} from 'react-native';
import { useSyncExternalStore } from 'react';

import { exerciseInfo } from '@/src/lib/training';
import {
  watchTrainingPayload,
  type WatchWorkoutPlan,
} from '@/src/lib/watch-workout-contract';
import type { TrainingData, WorkoutSession } from '@/src/types';
import { classifyAppleWorkout } from '@/src/lib/workout-activity';

export {
  workoutSessionFromWatch,
  watchTrainingPayload,
  watchWorkoutPayload,
} from '@/src/lib/watch-workout-contract';
export type {
  WatchWorkoutExercise,
  WatchWorkoutLibrary,
  WatchWorkoutPlan,
  WatchWorkoutSet,
} from '@/src/lib/watch-workout-contract';

type WorkoutBridge = {
  syncWorkout(payload: Record<string, unknown>): void;
  sendCommand(command: 'pause' | 'resume' | 'end' | 'discard'): void;
  drainEvents(): Promise<LiveWorkoutEvent[]>;
};

const bridge = NativeModules.VigorlyWorkoutBridge as WorkoutBridge | undefined;

export type AppleWatchSyncStatus = {
  state: 'unavailable' | 'queued' | 'synced';
  queuedAt?: string;
  lastSyncedAt?: string;
  reachable?: boolean;
};

let watchSyncStatus: AppleWatchSyncStatus = {
  state: Platform.OS === 'ios' && bridge ? 'queued' : 'unavailable',
};
const watchSyncListeners = new Set<() => void>();

function publishWatchSyncStatus(next: AppleWatchSyncStatus) {
  watchSyncStatus = next;
  watchSyncListeners.forEach((listener) => listener());
}

/** Observable delivery state for UI; a Watch receipt is stronger than an iPhone queue write. */
export function useAppleWatchSyncStatus() {
  return useSyncExternalStore(
    (listener) => {
      watchSyncListeners.add(listener);
      return () => watchSyncListeners.delete(listener);
    },
    () => watchSyncStatus,
    () => watchSyncStatus,
  );
}

export function recordAppleWatchSyncReceipt(updatedAt: string, reachable?: boolean) {
  publishWatchSyncStatus({ state: 'synced', lastSyncedAt: updatedAt, reachable });
}

export type LiveWorkoutEvent =
  | {
      type: 'metrics';
      heartRateBpm?: number;
      activeCalories?: number;
      source?: string;
      updatedAt?: string;
    }
  | {
      type: 'sessionStarted' | 'sessionUpdated' | 'sessionFinished';
      eventId?: string;
      session: WatchWorkoutPlan;
    }
  // Kept for compatibility with build 10 while testers move to the full logger.
  | { type: 'set'; sessionExerciseId: string; setId: string; completed: boolean }
  | { type: 'state'; state: string | number; source?: string }
  | { type: 'syncReceipt'; updatedAt: string; reachable?: boolean }
  | { type: 'error'; message: string };

export function syncTrainingToWatch(training: TrainingData) {
  if (Platform.OS !== 'ios' || !bridge) return false;
  bridge.syncWorkout(watchTrainingPayload(training));
  publishWatchSyncStatus({
    ...watchSyncStatus,
    state: 'queued',
    queuedAt: new Date().toISOString(),
  });
  return true;
}

function workoutActivityNames(session: WorkoutSession, training: TrainingData) {
  return session.exercises
    .map((entry) => exerciseInfo(training, entry.exerciseId).name);
}

export function appleWorkoutActivityKey(session: WorkoutSession, training: TrainingData) {
  return classifyAppleWorkout(workoutActivityNames(session, training));
}

export async function startWorkoutOnAppleWatch(session: WorkoutSession, training: TrainingData) {
  if (Platform.OS !== 'ios' || !bridge) {
    throw new Error('Apple Watch workout tracking is available in the iOS app.');
  }
  if (Number.parseFloat(String(Platform.Version)) < 17) {
    throw new Error('Live Apple Watch workout tracking requires iOS 17 or later.');
  }
  syncTrainingToWatch(training);
  const healthkit = await import('@kingstinct/react-native-healthkit');
  const authorized = await healthkit.requestAuthorization({
    toRead: [
      'HKQuantityTypeIdentifierHeartRate',
      'HKQuantityTypeIdentifierActiveEnergyBurned',
    ],
  });
  if (!authorized) throw new Error('Apple Health authorization was not completed.');
  const activityKey = appleWorkoutActivityKey(session, training);
  const started = await healthkit.startWatchApp({
    activityType: healthkit.WorkoutActivityType[activityKey],
    locationType: healthkit.WorkoutSessionLocationType.indoor,
  });
  if (!started) throw new Error('The paired Apple Watch could not start the workout.');
}

export function sendAppleWatchWorkoutCommand(command: 'pause' | 'resume' | 'end' | 'discard') {
  bridge?.sendCommand(command);
}

export function subscribeToAppleWatchWorkout(
  listener: (event: LiveWorkoutEvent) => void,
): EmitterSubscription | null {
  if (Platform.OS !== 'ios' || !bridge) return null;
  void bridge.drainEvents().then((events) => events.forEach(listener)).catch(() => undefined);
  return new NativeEventEmitter(NativeModules.VigorlyWorkoutBridge)
    .addListener('VigorlyWorkoutEvent', listener);
}
