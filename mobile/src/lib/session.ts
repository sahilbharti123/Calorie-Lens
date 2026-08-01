import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

import type { AuthSession } from '@/src/types';

/**
 * Persisted-storage keys keep the pre-rename `calorie-lens.` prefix on purpose.
 * They identify data already written to the device — renaming them would
 * orphan every existing vault, session and backup rather than migrate it.
 * The name a user sees is set in app.json; these are invisible.
 */
const SESSION_KEY = 'calorie-lens.auth-session.v1';
const RECOVERY_KEY = 'calorie-lens.recovery-code.v1';
const ONBOARDING_KEY = 'calorie-lens.onboarding-complete.v2';
const OPTIONS: SecureStore.SecureStoreOptions = {
  keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

async function getItem(key: string) {
  if (Platform.OS === 'web') return globalThis.localStorage?.getItem(key) ?? null;
  return SecureStore.getItemAsync(key, OPTIONS);
}

async function setItem(key: string, value: string) {
  if (Platform.OS === 'web') {
    globalThis.localStorage?.setItem(key, value);
    return;
  }
  await SecureStore.setItemAsync(key, value, OPTIONS);
}

async function removeItem(key: string) {
  if (Platform.OS === 'web') {
    globalThis.localStorage?.removeItem(key);
    return;
  }
  await SecureStore.deleteItemAsync(key, OPTIONS);
}

export async function readSession() {
  const value = await getItem(SESSION_KEY);
  if (!value) return null;
  try {
    return JSON.parse(value) as AuthSession;
  } catch {
    await removeItem(SESSION_KEY);
    return null;
  }
}

export async function saveSession(session: AuthSession) {
  await setItem(SESSION_KEY, JSON.stringify(session));
}

export async function clearSession() {
  await removeItem(SESSION_KEY);
}

export async function saveRecoveryCode(userId: string, code: string) {
  await setItem(
    RECOVERY_KEY,
    JSON.stringify({ userId, code }),
  );
}

export async function readRecoveryCode(userId?: string) {
  const value = await getItem(RECOVERY_KEY);
  if (!value) return null;
  try {
    const saved = JSON.parse(value) as { userId: string; code: string };
    if (userId && saved.userId !== userId) return null;
    return saved.code;
  } catch {
    return null;
  }
}

export async function clearRecoveryCode() {
  await removeItem(RECOVERY_KEY);
}

export async function readOnboardingComplete() {
  return (await getItem(ONBOARDING_KEY)) === 'true';
}

export async function saveOnboardingComplete(complete: boolean) {
  if (complete) {
    await setItem(ONBOARDING_KEY, 'true');
  } else {
    await removeItem(ONBOARDING_KEY);
  }
}
