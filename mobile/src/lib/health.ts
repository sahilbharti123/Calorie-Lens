import { Platform } from 'react-native';

import { startOfLocalDay } from '@/src/lib/date';
import type { HealthSnapshot } from '@/src/types';

export async function syncNativeHealth(): Promise<HealthSnapshot> {
  if (Platform.OS === 'ios') return syncAppleHealth();
  if (Platform.OS === 'android') return syncHealthConnect();
  throw new Error('Health sync is available in native iOS and Android builds.');
}

async function syncAppleHealth(): Promise<HealthSnapshot> {
  const healthkit = await import('@kingstinct/react-native-healthkit');
  if (!healthkit.isHealthDataAvailable()) throw new Error('Apple Health is not available on this device.');

  await healthkit.requestAuthorization({
    toRead: [
      'HKQuantityTypeIdentifierStepCount',
      'HKQuantityTypeIdentifierActiveEnergyBurned',
      'HKQuantityTypeIdentifierBodyMass',
      'HKCategoryTypeIdentifierSleepAnalysis',
    ],
  });

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
    source: 'Apple Health',
  };
}

async function syncHealthConnect(): Promise<HealthSnapshot> {
  const health = await import('react-native-health-connect');
  const available = await health.initialize();
  if (!available) throw new Error('Health Connect is not available. Install or update it, then try again.');

  await health.requestPermission([
    { accessType: 'read', recordType: 'Steps' },
    { accessType: 'read', recordType: 'ActiveCaloriesBurned' },
    { accessType: 'read', recordType: 'Weight' },
    { accessType: 'read', recordType: 'SleepSession' },
  ]);

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
    source: 'Health Connect',
  };
}
