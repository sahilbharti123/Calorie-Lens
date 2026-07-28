import { type Href, Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import 'react-native-reanimated';

import { AppProvider } from '@/src/store/app-store';
import { AuthProvider, useAuth } from '@/src/store/auth-store';
import { palette, type } from '@/src/theme';

export const unstable_settings = {
  anchor: '(tabs)',
};

export default function RootLayout() {
  return (
    <AuthProvider>
      <SessionRouter />
    </AuthProvider>
  );
}

function SessionRouter() {
  const router = useRouter();
  const segments = useSegments();
  const { loading, offlineMode, session } = useAuth();

  useEffect(() => {
    if (loading) return;
    const isAuthScreen = (segments[0] as string | undefined) === 'auth';
    const canEnterApp = Boolean(session || offlineMode);
    if (!canEnterApp && !isAuthScreen) router.replace('/auth' as Href);
    if (canEnterApp && isAuthScreen) router.replace('/(tabs)');
  }, [loading, offlineMode, router, segments, session]);

  if (loading) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator color={palette.forest} />
      </View>
    );
  }

  return (
    <AppProvider>
      <Stack
        screenOptions={{
          contentStyle: { backgroundColor: palette.canvas },
          headerShadowVisible: false,
          headerStyle: { backgroundColor: palette.canvas },
          headerTintColor: palette.ink,
        }}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="quick-log"
          options={{
            presentation: 'modal',
            title: 'Quick log',
            headerTitleStyle: { fontFamily: type.demi },
          }}
        />
        <Stack.Screen
          name="settings"
          options={{
            presentation: 'modal',
            title: 'Profile & goals',
          headerTitleStyle: { fontFamily: type.demi },
          }}
        />
        <Stack.Screen name="auth" options={{ headerShown: false }} />
        <Stack.Screen
          name="account"
          options={{
            presentation: 'modal',
            title: 'Account & privacy',
            headerTitleStyle: { fontFamily: type.demi },
          }}
        />
      </Stack>
      <StatusBar style="dark" />
    </AppProvider>
  );
}

const styles = StyleSheet.create({
  loader: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.canvas,
  },
});
