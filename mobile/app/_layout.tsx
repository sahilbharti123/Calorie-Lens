import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import 'react-native-reanimated';

import { AppProvider } from '@/src/store/app-store';
import { AuthProvider, useAuth } from '@/src/store/auth-store';
import { WorkoutProvider } from '@/src/store/workout-store';
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
  const { loading, offlineMode, onboardingComplete, session } = useAuth();

  if (loading) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator color={palette.forest} />
      </View>
    );
  }

  const canEnterApp = Boolean(session || offlineMode);

  return (
    <AppProvider>
      <WorkoutProvider>
        <Stack
          screenOptions={{
            contentStyle: { backgroundColor: palette.canvas },
            headerShadowVisible: false,
            headerStyle: { backgroundColor: palette.canvas },
            headerTintColor: palette.ink,
          }}>
          <Stack.Protected guard={!onboardingComplete}>
            <Stack.Screen name="onboarding" options={{ headerShown: false }} />
          </Stack.Protected>
          <Stack.Protected guard={onboardingComplete && !canEnterApp}>
            <Stack.Screen name="auth" options={{ headerShown: false }} />
          </Stack.Protected>
          <Stack.Protected guard={onboardingComplete && canEnterApp}>
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
            <Stack.Screen
              name="account"
              options={{
                presentation: 'modal',
                title: 'Account & privacy',
                headerTitleStyle: { fontFamily: type.demi },
              }}
            />
            <Stack.Screen
              name="workout-session"
              options={{ presentation: 'fullScreenModal', headerShown: false }}
            />
            <Stack.Screen
              name="routine-editor"
              options={{
                presentation: 'modal',
                title: 'Routine',
                headerTitleStyle: { fontFamily: type.demi },
              }}
            />
            <Stack.Screen
              name="exercise-picker"
              options={{
                presentation: 'modal',
                title: 'Add exercises',
                headerTitleStyle: { fontFamily: type.demi },
              }}
            />
            <Stack.Screen
              name="workout-history"
              options={{
                title: 'Workout history',
                headerTitleStyle: { fontFamily: type.demi },
              }}
            />
            <Stack.Screen
              name="workout/[id]"
              options={{
                title: 'Workout',
                headerTitleStyle: { fontFamily: type.demi },
              }}
            />
            <Stack.Screen
              name="exercise/[id]"
              options={{
                title: 'Exercise',
                headerTitleStyle: { fontFamily: type.demi },
              }}
            />
          </Stack.Protected>
        </Stack>
        <StatusBar style="dark" />
      </WorkoutProvider>
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
