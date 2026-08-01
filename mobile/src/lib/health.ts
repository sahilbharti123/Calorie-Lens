import { Platform } from 'react-native';
import * as Device from 'expo-device';

import { startOfLocalDay } from '@/src/lib/date';
import type { HealthSnapshot } from '@/src/types';

export function healthSnapshotHasSamples(snapshot: HealthSnapshot) {
  if (snapshot.sampleCount != null) return snapshot.sampleCount > 0;
  return [snapshot.steps, snapshot.activeCalories, snapshot.sleepHours, snapshot.weightKg]
    .some((value) => value != null);
}

export function healthMetricHasSamples(
  snapshot: HealthSnapshot,
  metric: 'steps' | 'activeCalories' | 'sleep' | 'weight',
) {
  if (snapshot.sampleCounts) return snapshot.sampleCounts[metric] > 0;
  const value = metric === 'sleep'
    ? snapshot.sleepHours
    : metric === 'weight'
      ? snapshot.weightKg
      : snapshot[metric];
  return value != null;
}

export async function syncNativeHealth(): Promise<HealthSnapshot> {
  if (Platform.OS === 'ios') return syncAppleHealth();
  if (Platform.OS === 'android') return syncHealthConnect();
  throw new Error('Health sync is available in native iOS and Android builds.');
}

export function healthSetupCopy() {
  if (Platform.OS === 'ios') {
    if (!Device.isDevice) {
      return {
        title: 'Apple Health test mode',
        detail: 'The simulator can exercise the Health permission flow, but it cannot receive records from your Apple Watch and may have no fitness samples. Use a development build on your iPhone for real Watch data.',
        physicalDeviceRequired: true,
      };
    }
    return {
      title: 'Apple Health + Watch',
      detail: 'Watch workouts and activity appear after the Watch syncs them into Apple Health on this iPhone.',
      physicalDeviceRequired: false,
    };
  }
  if (Platform.OS === 'android') {
    return {
      title: 'Health Connect',
      detail: Device.isDevice
        ? 'Reads steps, activity, sleep and weight from apps you approve.'
        : 'An emulator needs Health Connect and sample records before a sync can return data.',
      physicalDeviceRequired: false,
    };
  }
  return {
    title: 'Health connection unavailable',
    detail: 'Health sync is available only in native iOS and Android builds.',
    physicalDeviceRequired: true,
  };
}

async function syncAppleHealth(): Promise<HealthSnapshot> {
  const healthkit = await import('@kingstinct/react-native-healthkit');
  if (!healthkit.isHealthDataAvailable()) {
    throw new Error(
      Device.isDevice
        ? 'Apple Health is unavailable on this device. Check Screen Time or device-management restrictions.'
        : 'HealthKit is unavailable in this simulator runtime. Use an iPhone development build for Apple Health and Watch data.',
    );
  }

  const authorized = await healthkit.requestAuthorization({
    toRead: [
      'HKQuantityTypeIdentifierStepCount',
      'HKQuantityTypeIdentifierActiveEnergyBurned',
      'HKQuantityTypeIdentifierBodyMass',
      'HKCategoryTypeIdentifierSleepAnalysis',
    ],
  });
  if (!authorized) {
    throw new Error('Apple Health did not finish authorization. Open Health › Sharing › Apps › Vigorly and review access.');
  }

  const date = { startDate: startOfLocalDay(), endDate: new Date() };
  const sleepStart = new Date(startOfLocalDay());
  sleepStart.setHours(sleepStart.getHours() - 12);
  const [steps, calories, weights, sleep] = await Promise.all([
    healthkit.queryQuantitySamples('HKQuantityTypeIdentifierStepCount', {
      filter: { date },
      limit: -1,
      unit: 'count',
    }),
    healthkit.queryQuantitySamples('HKQuantityTypeIdentifierActiveEnergyBurned', {
      filter: { date },
      limit: -1,
      unit: 'kcal',
    }),
    healthkit.queryQuantitySamples('HKQuantityTypeIdentifierBodyMass', {
      limit: 1,
      ascending: false,
      unit: 'kg',
    }),
    healthkit.queryCategorySamples('HKCategoryTypeIdentifierSleepAnalysis', {
      filter: { date: { startDate: sleepStart, endDate: new Date() } },
      limit: -1,
    }),
  ]);

  const asleepMs = sleep
    .filter((sample) => [1, 3, 4, 5].includes(sample.value))
    .reduce((sum, sample) => sum + (sample.endDate.getTime() - sample.startDate.getTime()), 0);
  return {
    steps: Math.round(steps.reduce((sum, sample) => sum + sample.quantity, 0)),
    activeCalories: Math.round(calories.reduce((sum, sample) => sum + sample.quantity, 0)),
    weightKg: weights[0]?.quantity,
    sleepHours: Math.round((asleepMs / 3_600_000) * 10) / 10,
    sampleCount: steps.length + calories.length + weights.length + sleep.length,
    sampleCounts: {
      steps: steps.length,
      activeCalories: calories.length,
      sleep: sleep.length,
      weight: weights.length,
    },
    source: 'Apple Health',
  };
}

async function syncHealthConnect(): Promise<HealthSnapshot> {
  const health = await import('react-native-health-connect');
  const available = await health.initialize();
  if (!available) throw new Error('Health Connect is not available. Install or update it, then try again.');

  const permissions = await health.requestPermission([
    { accessType: 'read', recordType: 'Steps' },
    { accessType: 'read', recordType: 'ActiveCaloriesBurned' },
    { accessType: 'read', recordType: 'Weight' },
    { accessType: 'read', recordType: 'SleepSession' },
  ]);
  if (!permissions.length) {
    throw new Error('No Health Connect categories were approved. Review Vigorly permissions in Health Connect.');
  }

  const startTime = startOfLocalDay().toISOString();
  const endTime = new Date().toISOString();
  const options = { timeRangeFilter: { operator: 'between' as const, startTime, endTime } };
  const [steps, calories, weights, sleep] = await Promise.all([
    health.readRecords('Steps', options),
    health.readRecords('ActiveCaloriesBurned', options),
    health.readRecords('Weight', { timeRangeFilter: { operator: 'before', endTime }, pageSize: 1 }),
    health.readRecords('SleepSession', options),
  ]);

  const sleepMs = sleep.records.reduce(
    (sum, record) => sum + (new Date(record.endTime).getTime() - new Date(record.startTime).getTime()),
    0,
  );
  return {
    steps: Math.round(steps.records.reduce((sum, record) => sum + record.count, 0)),
    activeCalories: Math.round(
      calories.records.reduce((sum, record) => sum + record.energy.inKilocalories, 0),
    ),
    weightKg: weights.records[0]?.weight.inKilograms,
    sleepHours: Math.round((sleepMs / 3_600_000) * 10) / 10,
    sampleCount: steps.records.length + calories.records.length + weights.records.length + sleep.records.length,
    sampleCounts: {
      steps: steps.records.length,
      activeCalories: calories.records.length,
      sleep: sleep.records.length,
      weight: weights.records.length,
    },
    source: 'Health Connect',
  };
}
