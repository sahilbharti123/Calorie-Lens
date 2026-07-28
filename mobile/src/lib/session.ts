import * as SecureStore from 'expo-secure-store';

import type { AuthSession } from '@/src/types';

const SESSION_KEY = 'calorie-lens.auth-session.v1';
const RECOVERY_KEY = 'calorie-lens.recovery-code.v1';
const OPTIONS: SecureStore.SecureStoreOptions = {
  keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

export async function readSession() {
  const value = await SecureStore.getItemAsync(SESSION_KEY, OPTIONS);
  if (!value) return null;
  try {
    return JSON.parse(value) as AuthSession;
  } catch {
    await SecureStore.deleteItemAsync(SESSION_KEY, OPTIONS);
    return null;
  }
}

export async function saveSession(session: AuthSession) {
  await SecureStore.setItemAsync(SESSION_KEY, JSON.stringify(session), OPTIONS);
}

export async function clearSession() {
  await SecureStore.deleteItemAsync(SESSION_KEY, OPTIONS);
}

export async function saveRecoveryCode(userId: string, code: string) {
  await SecureStore.setItemAsync(
    RECOVERY_KEY,
    JSON.stringify({ userId, code }),
    OPTIONS,
  );
}

export async function readRecoveryCode(userId?: string) {
  const value = await SecureStore.getItemAsync(RECOVERY_KEY, OPTIONS);
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
  await SecureStore.deleteItemAsync(RECOVERY_KEY, OPTIONS);
}
