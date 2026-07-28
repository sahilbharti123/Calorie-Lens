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

export function apiUrl() {
  return process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '') ?? '';
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
    throw new ApiError(0, 'The service is unreachable. Your changes remain saved offline.');
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
