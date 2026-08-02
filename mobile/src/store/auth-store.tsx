import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { Linking } from 'react-native';

import { authErrorMessage, parseAuthLink, toAuthSession } from '@/src/lib/supabase-auth';
import {
  clearSession,
  readOfflineMode,
  readOnboardingComplete,
  saveOfflineMode,
  saveOnboardingComplete,
  saveSession,
} from '@/src/lib/session';
import { requireSupabase, supabase, supabaseConfigured } from '@/src/lib/supabase';
import type { AuthSession } from '@/src/types';

type SignupResult = { needsEmailConfirmation: boolean };
type AuthContextValue = {
  session: AuthSession | null;
  loading: boolean;
  offlineMode: boolean;
  onboardingComplete: boolean;
  serviceConfigured: boolean;
  passwordRecoveryReady: boolean;
  recoveryError: string;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (displayName: string, email: string, password: string) => Promise<SignupResult>;
  requestPasswordReset: (email: string) => Promise<void>;
  completeRecoveredPassword: (newPassword: string) => Promise<void>;
  continueOffline: () => void;
  completeOnboarding: () => Promise<void>;
  restartOnboarding: () => Promise<void>;
  exitOfflineMode: () => void;
  signOut: () => Promise<void>;
  deleteAccount: () => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: React.PropsWithChildren) {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [offlineMode, setOfflineMode] = useState(false);
  const [onboardingComplete, setOnboardingComplete] = useState(false);
  const [passwordRecoveryReady, setPasswordRecoveryReady] = useState(false);
  const [recoveryError, setRecoveryError] = useState('');

  useEffect(() => {
    let active = true;
    void Promise.all([
      readOnboardingComplete(),
      readOfflineMode(),
      supabase?.auth.getSession() ?? Promise.resolve({ data: { session: null }, error: null }),
    ]).then(([hasOnboarded, storedOfflineMode, result]) => {
      if (!active) return;
      setOnboardingComplete(hasOnboarded);
      if (result.error) setRecoveryError(authErrorMessage(result.error));
      if (result.data.session) {
        const next = toAuthSession(result.data.session);
        setSession(next);
        setOfflineMode(false);
        void saveSession(next);
      } else {
        setOfflineMode(storedOfflineMode);
        void clearSession();
      }
    }).finally(() => {
      if (active) setLoading(false);
    });

    const subscription = supabase?.auth.onAuthStateChange((event, nextSession) => {
      if (!active) return;
      if (event === 'PASSWORD_RECOVERY') setPasswordRecoveryReady(true);
      if (nextSession) {
        const next = toAuthSession(nextSession);
        setSession(next);
        setOfflineMode(false);
        void Promise.all([saveSession(next), saveOfflineMode(false)]);
      } else if (event === 'SIGNED_OUT') {
        setSession(null);
        void clearSession();
      }
    });

    return () => {
      active = false;
      subscription?.data.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    async function handleUrl(url: string | null) {
      if (!url || !supabase) return;
      const tokens = parseAuthLink(url);
      if (!tokens) return;
      setRecoveryError('');
      const { error } = await supabase.auth.setSession({
        access_token: tokens.accessToken,
        refresh_token: tokens.refreshToken,
      });
      if (error) {
        setRecoveryError(authErrorMessage(error));
        return;
      }
      if (tokens.type === 'recovery' || url.includes('auth-reset')) {
        setPasswordRecoveryReady(true);
      }
    }

    void Linking.getInitialURL().then(handleUrl);
    const subscription = Linking.addEventListener('url', ({ url }) => void handleUrl(url));
    return () => subscription.remove();
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const client = requireSupabase();
    const { error } = await client.auth.signInWithPassword({ email: email.trim(), password });
    if (error) throw new Error(authErrorMessage(error));
  }, []);

  const signUp = useCallback(async (displayName: string, email: string, password: string) => {
    const client = requireSupabase();
    const { data, error } = await client.auth.signUp({
      email: email.trim(),
      password,
      options: {
        data: { display_name: displayName.trim() },
        emailRedirectTo: 'vigorly://auth',
      },
    });
    if (error) throw new Error(authErrorMessage(error));
    return { needsEmailConfirmation: !data.session };
  }, []);

  const requestPasswordReset = useCallback(async (email: string) => {
    const client = requireSupabase();
    const { error } = await client.auth.resetPasswordForEmail(email.trim(), {
      redirectTo: 'vigorly://auth-reset',
    });
    if (error) throw new Error(authErrorMessage(error));
  }, []);

  const completeRecoveredPassword = useCallback(async (newPassword: string) => {
    const client = requireSupabase();
    const { error } = await client.auth.updateUser({ password: newPassword });
    if (error) throw new Error(authErrorMessage(error));
    setPasswordRecoveryReady(false);
  }, []);

  const signOut = useCallback(async () => {
    const client = supabase;
    if (client) {
      const { error } = await client.auth.signOut({ scope: 'global' });
      if (error) await client.auth.signOut({ scope: 'local' });
    }
    await Promise.all([clearSession(), saveOfflineMode(false)]);
    setSession(null);
    setOfflineMode(false);
  }, []);

  const deleteAccount = useCallback(async () => {
    const client = requireSupabase();
    const { error } = await client.rpc('delete_own_account');
    if (error) throw new Error(authErrorMessage(error));
    // The user row and its auth session no longer exist after the RPC.
    await client.auth.signOut({ scope: 'local' });
    await Promise.all([clearSession(), saveOfflineMode(false)]);
    setSession(null);
    setOfflineMode(false);
  }, []);

  const changePassword = useCallback(async (currentPassword: string, newPassword: string) => {
    const client = requireSupabase();
    if (!session?.user.email) throw new Error('Sign in to change your password.');
    const reauthenticated = await client.auth.signInWithPassword({
      email: session.user.email,
      password: currentPassword,
    });
    if (reauthenticated.error) throw new Error(authErrorMessage(reauthenticated.error));
    const updated = await client.auth.updateUser({ password: newPassword });
    if (updated.error) throw new Error(authErrorMessage(updated.error));
    const signedOut = await client.auth.signOut({ scope: 'global' });
    if (signedOut.error) await client.auth.signOut({ scope: 'local' });
    await clearSession();
    setSession(null);
  }, [session?.user.email]);

  const completeOnboarding = useCallback(async () => {
    await saveOnboardingComplete(true);
    setOnboardingComplete(true);
  }, []);

  const restartOnboarding = useCallback(async () => {
    await saveOnboardingComplete(false);
    setOnboardingComplete(false);
  }, []);

  const continueOffline = useCallback(() => {
    setOfflineMode(true);
    void saveOfflineMode(true);
  }, []);

  const exitOfflineMode = useCallback(() => {
    setOfflineMode(false);
    void saveOfflineMode(false);
  }, []);

  const value = useMemo<AuthContextValue>(() => ({
    session,
    loading,
    offlineMode,
    onboardingComplete,
    serviceConfigured: supabaseConfigured,
    passwordRecoveryReady,
    recoveryError,
    signIn,
    signUp,
    requestPasswordReset,
    completeRecoveredPassword,
    continueOffline,
    completeOnboarding,
    restartOnboarding,
    exitOfflineMode,
    signOut,
    deleteAccount,
    changePassword,
  }), [
    session,
    loading,
    offlineMode,
    onboardingComplete,
    passwordRecoveryReady,
    recoveryError,
    signIn,
    signUp,
    requestPasswordReset,
    completeRecoveredPassword,
    continueOffline,
    completeOnboarding,
    restartOnboarding,
    exitOfflineMode,
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
