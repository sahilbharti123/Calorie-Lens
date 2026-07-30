import { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { apiUrl, readServiceHealth } from '@/src/lib/api-client';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, type } from '@/src/theme';

type Mode = 'login' | 'signup' | 'recover';
type ServiceProbe = 'checking' | 'online' | 'offline';

export default function AuthScreen() {
  const {
    continueOffline,
    recover,
    serviceConfigured,
    signIn,
    signUp,
  } = useAuth();
  const [mode, setMode] = useState<Mode>('signup');
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [recoveryCode, setRecoveryCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [probe, setProbe] = useState<ServiceProbe>('checking');

  const checkService = useCallback(async () => {
    if (!serviceConfigured) {
      setProbe('offline');
      return;
    }
    setProbe('checking');
    try {
      await readServiceHealth();
      setProbe('online');
    } catch {
      setProbe('offline');
    }
  }, [serviceConfigured]);

  useEffect(() => {
    void checkService();
  }, [checkService]);

  async function submit() {
    setError('');
    if (!email.trim() || !password) {
      setError('Enter your email and password.');
      return;
    }
    if (mode === 'signup' && !displayName.trim()) {
      setError('Tell us what to call you.');
      return;
    }
    if (password.length < 10) {
      setError('Use at least 10 characters for your password.');
      return;
    }
    setBusy(true);
    try {
      if (mode === 'login') {
        await signIn(email, password);
      } else if (mode === 'signup') {
        const result = await signUp(displayName, email, password);
        Alert.alert(
          'Save your recovery code',
          `${result.recoveryCode}\n\nThis is the only way to reset your password. It is also saved securely on this device.`,
        );
      } else {
        if (!recoveryCode.trim()) {
          setError('Enter your recovery code.');
          return;
        }
        const result = await recover(email, recoveryCode, password);
        setPassword('');
        setRecoveryCode('');
        setMode('login');
        Alert.alert(
          'Password reset',
          `You can now sign in with your new password.\n\nNew recovery code:\n${result.recoveryCode}`,
        );
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'That did not work. Try again.');
    } finally {
      setBusy(false);
    }
  }

  const title = mode === 'login'
    ? 'Your fitness, remembered.'
    : mode === 'signup'
      ? 'Build your private fitness vault.'
      : 'Recover your account.';

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled">
          <View style={styles.brandRow}>
            <View style={styles.mark}><Glyph name="spark" color={palette.lime} size={24} /></View>
            <Text style={styles.brand}>CALORIE LENS</Text>
          </View>

          <View style={styles.intro}>
            <Text style={styles.kicker}>PRIVATE · LIGHT · YOURS</Text>
            <Text style={styles.title}>{title}</Text>
            <Text style={styles.subtitle}>
              Log food, workouts, water and health data in seconds. Your account keeps it synced across your devices.
            </Text>
          </View>

          <View style={styles.modePicker}>
            <ModeButton active={mode === 'login'} label="Sign in" onPress={() => setMode('login')} />
            <ModeButton active={mode === 'signup'} label="Create" onPress={() => setMode('signup')} />
            <ModeButton active={mode === 'recover'} label="Recover" onPress={() => setMode('recover')} />
          </View>

          <View style={styles.form}>
            {mode === 'signup' ? (
              <AuthField
                autoCapitalize="words"
                label="Your name"
                placeholder="Sahil"
                value={displayName}
                onChangeText={setDisplayName}
              />
            ) : null}
            <AuthField
              autoCapitalize="none"
              keyboardType="email-address"
              label="Email"
              placeholder="you@example.com"
              value={email}
              onChangeText={setEmail}
            />
            {mode === 'recover' ? (
              <AuthField
                autoCapitalize="none"
                label="Recovery code"
                placeholder="xxxxxx-xxxxxx-xxxxxx-xxxxxx"
                value={recoveryCode}
                onChangeText={setRecoveryCode}
              />
            ) : null}
            <AuthField
              autoCapitalize="none"
              label={mode === 'recover' ? 'New password' : 'Password'}
              placeholder="At least 10 characters"
              secureTextEntry
              value={password}
              onChangeText={setPassword}
            />
          </View>

          {error ? <Text style={styles.error}>{error}</Text> : null}
          {!serviceConfigured ? (
            <Text style={[styles.serviceNote, styles.serviceNoteAlone]}>
              Account service is not configured in this release. You can keep using the encrypted offline vault.
            </Text>
          ) : (
            <Pressable onPress={() => void checkService()} style={styles.serviceRow}>
              <View style={[
                styles.serviceDot,
                probe === 'online' && styles.serviceDotOnline,
                probe === 'offline' && styles.serviceDotOffline,
              ]} />
              <Text style={styles.serviceNote}>
                {probe === 'checking'
                  ? `Checking the account service at ${apiUrl()}…`
                  : probe === 'online'
                    ? 'Account service is reachable.'
                    : `Can’t reach the service at ${apiUrl()}. ${__DEV__
                      ? 'Start it with `uvicorn api:app --host 0.0.0.0 --port 8000` on your computer, keep both devices on the same Wi-Fi, then tap to retry.'
                      : 'Check your connection, then tap to retry.'}`}
              </Text>
            </Pressable>
          )}

          <Pressable
            accessibilityRole="button"
            disabled={busy || !serviceConfigured}
            onPress={submit}
            style={({ pressed }) => [
              styles.primary,
              (busy || !serviceConfigured) && styles.disabled,
              pressed && styles.pressed,
            ]}>
            {busy
              ? <ActivityIndicator color={palette.lime} />
              : <Text style={styles.primaryText}>
                  {mode === 'login' ? 'Sign in securely' : mode === 'signup' ? 'Create my account' : 'Reset password'}
                </Text>}
          </Pressable>

          <Pressable onPress={continueOffline} style={styles.offline}>
            <Text style={styles.offlineText}>Continue with encrypted offline mode</Text>
          </Pressable>
          <Text style={styles.privacy}>
            Calorie Lens is a fitness tracker, not medical care. Food and exercise values are estimates.
          </Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function ModeButton({
  active,
  label,
  onPress,
}: {
  active: boolean;
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} style={[styles.modeButton, active && styles.modeButtonActive]}>
      <Text style={[styles.modeText, active && styles.modeTextActive]}>{label}</Text>
    </Pressable>
  );
}

function AuthField(props: React.ComponentProps<typeof TextInput> & { label: string }) {
  const { label, ...inputProps } = props;
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        {...inputProps}
        placeholderTextColor="#929A93"
        style={styles.input}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { flexGrow: 1, paddingHorizontal: space.lg, paddingTop: 12, paddingBottom: 32 },
  brandRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  mark: { width: 42, height: 42, borderRadius: 15, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  brand: { color: palette.ink, fontFamily: type.demi, fontSize: 11, letterSpacing: 1.8 },
  intro: { marginTop: 54, marginBottom: 28 },
  kicker: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10, letterSpacing: 1.6, marginBottom: 10 },
  title: { color: palette.ink, fontFamily: type.demi, fontSize: 38, lineHeight: 43, letterSpacing: -1.6, maxWidth: 330 },
  subtitle: { color: palette.muted, fontFamily: type.regular, fontSize: 13, lineHeight: 20, marginTop: 12, maxWidth: 335 },
  modePicker: { flexDirection: 'row', backgroundColor: '#E8ECE3', borderRadius: radius.pill, padding: 4, marginBottom: 14 },
  modeButton: { flex: 1, height: 39, borderRadius: radius.pill, alignItems: 'center', justifyContent: 'center' },
  modeButtonActive: { backgroundColor: palette.paper },
  modeText: { color: palette.muted, fontFamily: type.medium, fontSize: 12 },
  modeTextActive: { color: palette.ink, fontFamily: type.demi },
  form: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 16 },
  field: { paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: palette.line },
  label: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.1, textTransform: 'uppercase', marginBottom: 5 },
  input: { color: palette.ink, fontFamily: type.medium, fontSize: 15, paddingVertical: 4 },
  error: { color: palette.coral, fontFamily: type.medium, fontSize: 11, lineHeight: 16, marginTop: 10, paddingHorizontal: 3 },
  serviceRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 7, marginTop: 10, paddingHorizontal: 3 },
  serviceDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#C9CFC6', marginTop: 4 },
  serviceDotOnline: { backgroundColor: palette.limeDark },
  serviceDotOffline: { backgroundColor: palette.coral },
  serviceNote: { flex: 1, color: palette.muted, fontFamily: type.regular, fontSize: 11, lineHeight: 16 },
  serviceNoteAlone: { flex: 0, marginTop: 10, paddingHorizontal: 3 },
  primary: { height: 56, borderRadius: radius.md, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center', marginTop: 16 },
  primaryText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
  disabled: { opacity: 0.48 },
  pressed: { transform: [{ scale: 0.99 }] },
  offline: { height: 48, alignItems: 'center', justifyContent: 'center', marginTop: 5 },
  offlineText: { color: palette.ink, fontFamily: type.medium, fontSize: 12 },
  privacy: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 14, textAlign: 'center', marginTop: 'auto', paddingTop: 28 },
});
