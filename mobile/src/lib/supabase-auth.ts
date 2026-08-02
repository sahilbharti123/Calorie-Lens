import type { Session as SupabaseSession, User } from '@supabase/supabase-js';

import type { AuthSession } from '@/src/types';

function displayName(user: User) {
  const metadataName = user.user_metadata?.display_name ?? user.user_metadata?.full_name;
  if (typeof metadataName === 'string' && metadataName.trim()) return metadataName.trim();
  return user.email?.split('@')[0] ?? 'Vigorly member';
}

export function toAuthSession(session: SupabaseSession): AuthSession {
  return {
    token: session.access_token,
    expiresAt: new Date((session.expires_at ?? 0) * 1000).toISOString(),
    user: {
      id: session.user.id,
      email: session.user.email ?? '',
      displayName: displayName(session.user),
      createdAt: session.user.created_at,
    },
  };
}

export type AuthLinkTokens = {
  accessToken: string;
  refreshToken: string;
  type?: string;
};

/** Supabase may place auth tokens in either the query string or URL fragment. */
export function parseAuthLink(url: string): AuthLinkTokens | null {
  try {
    const parsed = new URL(url);
    const fragment = new URLSearchParams(parsed.hash.replace(/^#/, ''));
    const accessToken = parsed.searchParams.get('access_token') ?? fragment.get('access_token');
    const refreshToken = parsed.searchParams.get('refresh_token') ?? fragment.get('refresh_token');
    if (!accessToken || !refreshToken) return null;
    return {
      accessToken,
      refreshToken,
      type: parsed.searchParams.get('type') ?? fragment.get('type') ?? undefined,
    };
  } catch {
    return null;
  }
}

export function authErrorMessage(error: unknown) {
  const fallback = error instanceof Error ? error.message : 'That did not work. Try again.';
  const value = fallback.toLowerCase();
  if (value.includes('invalid login credentials')) return 'The email or password is incorrect.';
  if (value.includes('email not confirmed')) return 'Confirm your email before signing in.';
  if (value.includes('user already registered')) return 'An account already exists for this email. Sign in instead.';
  if (value.includes('password should be')) return 'Use a stronger password with at least 10 characters.';
  if (value.includes('rate limit')) return 'Too many attempts. Wait a moment and try again.';
  if (value.includes('network request failed') || value.includes('fetch failed')) {
    return 'Vigorly could not reach Supabase. Check your connection and try again.';
  }
  return fallback;
}
