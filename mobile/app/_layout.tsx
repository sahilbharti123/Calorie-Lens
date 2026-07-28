import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import 'react-native-reanimated';

import { AppProvider } from '@/src/store/app-store';
import { palette, type } from '@/src/theme';

export const unstable_settings = {
  anchor: '(tabs)',
};

export default function RootLayout() {
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
      </Stack>
      <StatusBar style="dark" />
    </AppProvider>
  );
}
