import Constants from 'expo-constants';
import { Platform } from 'react-native';

import type { AuthSession } from '@/src/types';

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    const message = typeof detail === 'string'
      ? detail
      : (detail as { message?: string } | null)?.message ?? `Request failed with ${status}`;
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

/**
 * In development the API host is discovered from the Metro bundler address,
 * so a physical phone reaches the computer running `uvicorn api:app` without
 * any .env setup. `EXPO_PUBLIC_API_URL` always wins when set.
 */
function devHostUrl() {
  const hostUri = Constants.expoConfig?.hostUri
    ?? (Constants as unknown as { expoGoConfig?: { debuggerHost?: string } }).expoGoConfig?.debuggerHost;
  const host = hostUri?.split(':')[0];
  if (host && host !== 'localhost' && host !== '127.0.0.1') {
    return `http://${host}:8000`;
  }
  return Platform.OS === 'android'
    ? 'http://10.0.2.2:8000'
    : 'http://127.0.0.1:8000';
}

export function apiUrl() {
  const configured = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');
  if (configured) return configured;
  if (!__DEV__) return '';
  return devHostUrl();
}

export type ServiceHealth = {
  ok: boolean;
  ai_enabled: boolean;
  accounts_enabled: boolean;
  model: string;
};

export async function readServiceHealth() {
  return apiRequest<ServiceHealth>('/health');
}

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  session?: AuthSession | null,
): Promise<T> {
  const base = apiUrl();
  if (!base) throw new ApiError(0, 'Set EXPO_PUBLIC_API_URL to connect your account.');
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  if (session?.token) headers.set('Authorization', `Bearer ${session.token}`);
  let response: Response;
  try {
    response = await fetch(`${base}${path}`, { ...options, headers });
  } catch {
    throw new ApiError(
      0,
      `Could not reach the service at ${base}. `
      + (__DEV__
        ? 'Start it with `uvicorn api:app --host 0.0.0.0 --port 8000` and keep this device on the same Wi-Fi. Your changes stay saved offline.'
        : 'Check your connection — your changes stay saved offline.'),
    );
  }
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    const detail = (payload as { detail?: unknown } | null)?.detail ?? payload;
    throw new ApiError(response.status, detail);
  }
  return payload as T;
}
