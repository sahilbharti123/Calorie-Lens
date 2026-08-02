import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient, processLock } from '@supabase/supabase-js';
import { AppState, Platform } from 'react-native';
import 'react-native-url-polyfill/auto';

const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL?.trim() ?? '';
const supabasePublishableKey =
  process.env.EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY?.trim()
  ?? process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY?.trim()
  ?? '';

export const supabaseConfigured = Boolean(supabaseUrl && supabasePublishableKey);

export const supabase = supabaseConfigured
  ? createClient(supabaseUrl, supabasePublishableKey, {
      auth: {
        // AsyncStorage is native-only during Expo Router static rendering.
        // On web, Supabase selects its guarded browser storage adapter.
        storage: Platform.OS === 'web' ? undefined : AsyncStorage,
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: false,
        lock: processLock,
        storageKey: 'vigorly.supabase.auth.v1',
      },
    })
  : null;

export function requireSupabase() {
  if (!supabase) {
    throw new Error('Account service is not configured in this build. Add the Supabase URL and publishable key, then rebuild the app.');
  }
  return supabase;
}

// Supabase cannot refresh an expired token while the native app is suspended.
// Starting and stopping refresh with AppState avoids unnecessary background work.
if (supabase && Platform.OS !== 'web') {
  if (AppState.currentState === 'active') supabase.auth.startAutoRefresh();
  AppState.addEventListener('change', (state) => {
    if (state === 'active') supabase.auth.startAutoRefresh();
    else supabase.auth.stopAutoRefresh();
  });
}
