import { type ComponentProps, useCallback, useEffect, useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { BrandMark } from '@/src/components/brand-mark';
import { Glyph, type GlyphName } from '@/src/components/glyph';
import {
  GhostButton,
  PrimaryButton,
  Reveal,
  Screen,
  Segmented,
  Tap,
} from '@/src/components/ui';
import { apiUrl, readServiceHealth } from '@/src/lib/api-client';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, text } from '@/src/theme';

type Mode = 'login' | 'signup' | 'recover';
type ServiceProbe = 'checking' | 'online' | 'offline';

const MODES: { value: Mode; label: string }[] = [
  { value: 'login', label: 'Sign in' },
  { value: 'signup', label: 'Create' },
  { value: 'recover', label: 'Recover' },
];

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

  const probeTone = probe === 'online'
    ? palette.lime
    : probe === 'offline'
      ? palette.danger
      : palette.inkMid;
  const probeIcon: GlyphName = probe === 'online' ? 'shield' : probe === 'offline' ? 'alert' : 'cloud';

  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}>
          <Reveal>
            <View style={styles.brandRow}>
              <BrandMark size={64} />
              <View style={styles.brandCopy}>
                <Text style={styles.wordmark}>CALORIE LENS</Text>
                <Text style={styles.kicker}>PRIVATE · LIGHT · YOURS</Text>
              </View>
            </View>
          </Reveal>

          <Reveal index={1} style={styles.intro}>
            <Text style={styles.title}>{title}</Text>
            <Text style={styles.subtitle}>
              Log food, workouts, water and health data in seconds. Your account keeps it synced across your devices.
            </Text>
          </Reveal>

          <Reveal index={2}>
            <Segmented onChange={setMode} options={MODES} value={mode} />
          </Reveal>

          <Reveal index={3} style={styles.form}>
            {mode === 'signup' ? (
              <Field
                autoCapitalize="words"
                label="Your name"
                onChangeText={setDisplayName}
                placeholder="Sahil"
                value={displayName}
              />
            ) : null}
            <Field
              autoCapitalize="none"
              keyboardType="email-address"
              label="Email"
              onChangeText={setEmail}
              placeholder="you@example.com"
              value={email}
            />
            {mode === 'recover' ? (
              <Field
                autoCapitalize="none"
                label="Recovery code"
                onChangeText={setRecoveryCode}
                placeholder="xxxxxx-xxxxxx-xxxxxx-xxxxxx"
                value={recoveryCode}
              />
            ) : null}
            <Field
              autoCapitalize="none"
              label={mode === 'recover' ? 'New password' : 'Password'}
              onChangeText={setPassword}
              placeholder="At least 10 characters"
              secureTextEntry
              value={password}
            />
          </Reveal>

          <Reveal index={4} style={styles.status}>
            {error ? <Notice body={error} icon="alert" tone="danger" /> : null}

            {!serviceConfigured ? (
              <Notice
                body="Account service is not configured in this release. You can keep using the encrypted offline vault."
                icon="info"
                tone="info"
              />
            ) : (
              <Tap
                accessibilityLabel="Check the account service again"
                haptic="none"
                onPress={() => void checkService()}
                scaleTo={0.99}>
                <Notice
                  body={probe === 'checking'
                    ? `Checking the account service at ${apiUrl()}…`
                    : probe === 'online'
                      ? 'Account service is reachable.'
                      : `Can’t reach the service at ${apiUrl()}. ${__DEV__
                        ? 'Start it with `uvicorn api:app --host 0.0.0.0 --port 8000` on your computer, keep both devices on the same Wi-Fi, then tap to retry.'
                        : 'Check your connection, then tap to retry.'}`}
                  color={probeTone}
                  icon={probeIcon}
                />
              </Tap>
            )}
          </Reveal>

          <Reveal index={5} style={styles.actions}>
            <PrimaryButton
              disabled={busy || !serviceConfigured}
              icon={mode === 'login' ? 'lock' : mode === 'signup' ? 'spark' : 'shield'}
              label={mode === 'login' ? 'Sign in securely' : mode === 'signup' ? 'Create my account' : 'Reset password'}
              loading={busy}
              onPress={() => void submit()}
            />

            <View style={styles.divider}>
              <View style={styles.rule} />
              <Text style={styles.dividerLabel}>OR</Text>
              <View style={styles.rule} />
            </View>

            <GhostButton
              icon="shield"
              label="Continue without an account"
              onPress={continueOffline}
            />
            <Text style={styles.offlineNote}>
              Everything stays encrypted on this device — an account is only needed to sync across devices.
            </Text>
          </Reveal>

          <Reveal index={6} style={styles.privacyWrap}>
            <Text style={styles.privacy}>
              Vigorly is a fitness tracker, not medical care. Food and exercise values are estimates.
            </Text>
          </Reveal>
        </ScrollView>
      </KeyboardAvoidingView>
    </Screen>
  );
}

/** Dark text field: inset well, hairline border that lights up on focus. */
function Field({
  label,
  ...props
}: ComponentProps<typeof TextInput> & { label: string }) {
  const [focused, setFocused] = useState(false);
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label.toUpperCase()}</Text>
      <TextInput
        {...props}
        onBlur={() => setFocused(false)}
        onFocus={() => setFocused(true)}
        placeholderTextColor={palette.inkLow}
        style={[styles.input, focused && styles.inputFocused]}
      />
    </View>
  );
}

/** Inline message card — never a bare line of red text. */
function Notice({
  body,
  color,
  icon,
  tone = 'info',
}: {
  body: string;
  color?: string;
  icon: GlyphName;
  tone?: 'info' | 'danger' | 'accent';
}) {
  const hue = color
    ?? (tone === 'danger' ? palette.danger : tone === 'accent' ? palette.lime : palette.info);
  return (
    <View style={[styles.notice, { borderColor: `${hue}33`, backgroundColor: `${hue}10` }]}>
      <Glyph color={hue} name={icon} size={15} />
      <Text style={[styles.noticeText, { color: hue }]}>{body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: {
    flexGrow: 1,
    paddingHorizontal: space.md,
    paddingTop: space.sm,
    paddingBottom: space.xxl,
  },

  brandRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  brandCopy: { flex: 1 },
  wordmark: { ...text.label, fontSize: 11, letterSpacing: 2.2, color: palette.ink },
  kicker: { ...text.label, color: palette.lime, marginTop: 6 },

  intro: { marginTop: space.xl, marginBottom: space.lg },
  title: { ...text.title, color: palette.ink, maxWidth: 320 },
  subtitle: { ...text.body, color: palette.inkMid, marginTop: space.sm, maxWidth: 330 },

  form: { gap: 12, marginTop: space.md },
  field: { gap: 7 },
  fieldLabel: { ...text.label, color: palette.inkLow },
  input: {
    ...text.row,
    color: palette.ink,
    minHeight: 52,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surfaceLo,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  inputFocused: { borderColor: palette.lime },

  status: { gap: 10, marginTop: space.md },
  notice: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 9,
    borderWidth: 1,
    borderRadius: radius.sm,
    paddingHorizontal: 12,
    paddingVertical: 11,
  },
  noticeText: { ...text.caption, flex: 1 },

  actions: { gap: 12, marginTop: space.lg },
  divider: { flexDirection: 'row', alignItems: 'center', gap: 12, marginVertical: 2 },
  rule: { flex: 1, height: 1, backgroundColor: palette.line },
  dividerLabel: { ...text.label, color: palette.inkLow },
  offlineNote: {
    ...text.caption,
    fontSize: 11,
    color: palette.inkLow,
    textAlign: 'center',
    paddingHorizontal: space.sm,
  },

  privacyWrap: { marginTop: 'auto', paddingTop: space.xl },
  privacy: {
    ...text.caption,
    fontSize: 10.5,
    color: palette.inkLow,
    textAlign: 'center',
    paddingHorizontal: space.sm,
  },
});
