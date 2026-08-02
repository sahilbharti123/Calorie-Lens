import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
  Inter_800ExtraBold,
} from '@expo-google-fonts/inter';
import {
  SpaceGrotesk_500Medium,
  SpaceGrotesk_700Bold,
} from '@expo-google-fonts/space-grotesk';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import * as SystemUI from 'expo-system-ui';
import { useEffect, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import 'react-native-reanimated';

import { BrandMark } from '@/src/components/brand-mark';
import { AppProvider } from '@/src/store/app-store';
import { AuthProvider, useAuth } from '@/src/store/auth-store';
import { WorkoutProvider } from '@/src/store/workout-store';
import { font, palette } from '@/src/theme';

export const unstable_settings = {
  anchor: '(tabs)',
};

void SplashScreen.preventAutoHideAsync();
void SystemUI.setBackgroundColorAsync(palette.bg);

const modalScreen = (title: string) => ({
  presentation: 'modal' as const,
  title,
  headerTitleStyle: { fontFamily: font.semi, fontSize: 16, color: palette.ink },
});

/**
 * Longest the app will wait for bundled fonts before starting anyway.
 * A missing typeface is a cosmetic problem; a splash screen that never
 * leaves is a broken app, so the timeout always wins.
 */
const FONT_TIMEOUT_MS = 4000;

export default function RootLayout() {
  const [fontsLoaded, fontError] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
    Inter_800ExtraBold,
    SpaceGrotesk_500Medium,
    SpaceGrotesk_700Bold,
  });

  const [fontsTimedOut, setFontsTimedOut] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setFontsTimedOut(true), FONT_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, []);

  const ready = fontsLoaded || Boolean(fontError) || fontsTimedOut;

  // Hand off from the native splash as soon as JS can paint, rather than
  // waiting for the session to resolve — otherwise a slow or failed auth
  // check leaves the native splash on screen with nothing behind it.
  useEffect(() => {
    if (ready) void SplashScreen.hideAsync();
  }, [ready]);

  if (!ready) return <Splash />;

  return (
    <AuthProvider>
      <SessionRouter />
    </AuthProvider>
  );
}

/** Matches the native splash so the handoff to JS is invisible. */
function Splash() {
  return (
    <View style={styles.splash}>
      <BrandMark size={78} />
    </View>
  );
}

function SessionRouter() {
  const { loading, offlineMode, onboardingComplete, session } = useAuth();

  if (loading) return <Splash />;

  const canEnterApp = Boolean(session || offlineMode);

  return (
    <AppProvider>
      <WorkoutProvider>
        <Stack
          screenOptions={{
            contentStyle: { backgroundColor: palette.bg },
            headerShadowVisible: false,
            headerStyle: { backgroundColor: palette.bg },
            headerTintColor: palette.lime,
            headerTitleStyle: { fontFamily: font.semi, fontSize: 16, color: palette.ink },
          }}>
          <Stack.Protected guard={!onboardingComplete}>
            <Stack.Screen name="onboarding" options={{ headerShown: false }} />
          </Stack.Protected>
          <Stack.Protected guard={onboardingComplete && !canEnterApp}>
            <Stack.Screen name="auth" options={{ headerShown: false }} />
          </Stack.Protected>
          <Stack.Protected guard={onboardingComplete && canEnterApp}>
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen name="quick-log" options={{ presentation: 'modal', headerShown: false }} />
            <Stack.Screen name="weight-log" options={{ presentation: 'modal', headerShown: false }} />
            <Stack.Screen name="water-log" options={{ presentation: 'modal', headerShown: false }} />
            <Stack.Screen name="settings" options={modalScreen('Profile & goals')} />
            <Stack.Screen name="account" options={modalScreen('Account & privacy')} />
            <Stack.Screen
              name="workout-session"
              options={{ presentation: 'fullScreenModal', headerShown: false }}
            />
            <Stack.Screen name="routine-editor" options={modalScreen('Routine')} />
            <Stack.Screen name="exercise-picker" options={modalScreen('Add exercises')} />
            <Stack.Screen name="workout-history" options={{ title: 'History' }} />
            <Stack.Screen name="workout/[id]" options={{ title: 'Workout' }} />
            <Stack.Screen name="exercise/[id]" options={{ headerShown: false }} />
          </Stack.Protected>
          <Stack.Screen name="auth-reset" options={{ headerShown: false }} />
        </Stack>
        <StatusBar style="light" />
      </WorkoutProvider>
    </AppProvider>
  );
}

const styles = StyleSheet.create({
  splash: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.bg,
  },
});
