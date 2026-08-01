import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { ApiError, apiRequest, apiUrl } from '@/src/lib/api-client';
import {
  clearRecoveryCode,
  clearSession,
  readOnboardingComplete,
  readSession,
  saveOnboardingComplete,
  saveRecoveryCode,
  saveSession,
} from '@/src/lib/session';
import type { AuthSession } from '@/src/types';

type SignupResult = { recoveryCode: string };
type AuthContextValue = {
  session: AuthSession | null;
  loading: boolean;
  offlineMode: boolean;
  onboardingComplete: boolean;
  justCreated: boolean;
  serviceConfigured: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (displayName: string, email: string, password: string) => Promise<SignupResult>;
  recover: (email: string, recoveryCode: string, newPassword: string) => Promise<SignupResult>;
  continueOffline: () => void;
  completeOnboarding: () => Promise<void>;
  restartOnboarding: () => Promise<void>;
  exitOfflineMode: () => void;
  signOut: () => Promise<void>;
  deleteAccount: () => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

type AuthResponse = AuthSession & { recoveryCode?: string };

export function AuthProvider({ children }: React.PropsWithChildren) {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [offlineMode, setOfflineMode] = useState(false);
  const [onboardingComplete, setOnboardingComplete] = useState(false);
  const [justCreated, setJustCreated] = useState(false);
  const serviceConfigured = Boolean(apiUrl());

  useEffect(() => {
    Promise.all([readSession(), readOnboardingComplete()])
      .then(async ([stored, hasOnboarded]) => {
        setOnboardingComplete(hasOnboarded);
        if (!stored) {
          // Completing onboarding opts this device into the private offline
          // vault. Do not turn account creation into a recurring launch gate.
          setOfflineMode(hasOnboarded);
          return;
        }
        setSession(stored);
        try {
          const response = await apiRequest<{ user: AuthSession['user'] }>('/v1/auth/me', {}, stored);
          const refreshed = { ...stored, user: response.user };
          setSession(refreshed);
          await saveSession(refreshed);
        } catch (error) {
          if (error instanceof ApiError && error.status === 401) {
            await clearSession();
            setSession(null);
          }
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const acceptSession = useCallback(async (response: AuthResponse) => {
    const next: AuthSession = {
      token: response.token,
      expiresAt: response.expiresAt,
      user: response.user,
    };
    await saveSession(next);
    setOfflineMode(false);
    setSession(next);
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const response = await apiRequest<AuthResponse>('/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    await acceptSession(response);
    setJustCreated(false);
  }, [acceptSession]);

  const signUp = useCallback(async (displayName: string, email: string, password: string) => {
    const response = await apiRequest<AuthResponse>('/v1/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ display_name: displayName, email, password }),
    });
    await acceptSession(response);
    setJustCreated(true);
    if (response.recoveryCode) {
      await saveRecoveryCode(response.user.id, response.recoveryCode);
    }
    return { recoveryCode: response.recoveryCode ?? '' };
  }, [acceptSession]);

  const recover = useCallback(async (email: string, recoveryCode: string, newPassword: string) => {
    const response = await apiRequest<{
      userId: string;
      recoveryCode: string;
    }>('/v1/auth/recover', {
      method: 'POST',
      body: JSON.stringify({
        email,
        recovery_code: recoveryCode,
        new_password: newPassword,
      }),
    });
    await saveRecoveryCode(response.userId, response.recoveryCode);
    return { recoveryCode: response.recoveryCode };
  }, []);

  const signOut = useCallback(async () => {
    if (session) {
      try {
        await apiRequest('/v1/auth/logout', { method: 'POST' }, session);
      } catch {
        // The local credential still has to be removed when the service is offline.
      }
    }
    await clearSession();
    setSession(null);
    setOfflineMode(false);
    setJustCreated(false);
  }, [session]);

  const deleteAccount = useCallback(async () => {
    if (!session) return;
    await apiRequest('/v1/account', { method: 'DELETE' }, session);
    await Promise.all([clearSession(), clearRecoveryCode()]);
    setSession(null);
    setOfflineMode(false);
    setJustCreated(false);
  }, [session]);

  const changePassword = useCallback(async (currentPassword: string, newPassword: string) => {
    if (!session) throw new ApiError(401, 'Sign in to change your password.');
    await apiRequest('/v1/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
      }),
    }, session);
    await clearSession();
    setSession(null);
  }, [session]);

  const completeOnboarding = useCallback(async () => {
    await saveOnboardingComplete(true);
    setOnboardingComplete(true);
  }, []);

  const restartOnboarding = useCallback(async () => {
    await saveOnboardingComplete(false);
    setOnboardingComplete(false);
  }, []);

  const value = useMemo<AuthContextValue>(() => ({
    session,
    loading,
    offlineMode,
    onboardingComplete,
    justCreated,
    serviceConfigured,
    signIn,
    signUp,
    recover,
    continueOffline: () => setOfflineMode(true),
    completeOnboarding,
    restartOnboarding,
    exitOfflineMode: () => setOfflineMode(false),
    signOut,
    deleteAccount,
    changePassword,
  }), [
    session,
    loading,
    offlineMode,
    onboardingComplete,
    justCreated,
    serviceConfigured,
    signIn,
    signUp,
    recover,
    completeOnboarding,
    restartOnboarding,
    signOut,
    deleteAccount,
    changePassword,
  ]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error('useAuth must be used inside AuthProvider');
  return value;
}
