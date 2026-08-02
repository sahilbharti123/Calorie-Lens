import { type ComponentProps, useState } from 'react';
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
import { GhostButton, PrimaryButton, Reveal, Screen, Segmented, Tap } from '@/src/components/ui';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, text } from '@/src/theme';

type Mode = 'login' | 'signup' | 'recover';
const MODES: { value: 'login' | 'signup'; label: string }[] = [
  { value: 'login', label: 'Sign in' },
  { value: 'signup', label: 'Create account' },
];

export default function AuthScreen() {
  const {
    continueOffline,
    requestPasswordReset,
    serviceConfigured,
    signIn,
    signUp,
  } = useAuth();
  const [mode, setMode] = useState<Mode>('login');
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  function changeMode(next: 'login' | 'signup') {
    setMode(next);
    setError('');
    setMessage('');
  }

  async function submit() {
    setError('');
    setMessage('');
    if (!email.trim()) {
      setError('Enter your email address.');
      return;
    }
    if (mode !== 'recover' && !password) {
      setError('Enter your password.');
      return;
    }
    if (mode === 'signup' && !displayName.trim()) {
      setError('Tell us what to call you.');
      return;
    }
    if (mode !== 'recover' && password.length < 10) {
      setError('Use at least 10 characters for your password.');
      return;
    }
    setBusy(true);
    try {
      if (mode === 'login') {
        await signIn(email, password);
      } else if (mode === 'signup') {
        const result = await signUp(displayName, email, password);
        if (result.needsEmailConfirmation) {
          setMode('login');
          setPassword('');
          Alert.alert(
            'Confirm your email',
            'We sent you a secure confirmation link. Open it on this phone, then sign in.',
          );
        }
      } else {
        await requestPasswordReset(email);
        setMessage('Reset link sent. Open the email on this phone to choose a new password.');
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'That did not work. Try again.');
    } finally {
      setBusy(false);
    }
  }

  const title = mode === 'login'
    ? 'Your progress, on every device.'
    : mode === 'signup'
      ? 'Create your Vigorly account.'
      : 'Reset your password.';

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
                <Text style={styles.wordmark}>VIGORLY</Text>
                <Text style={styles.kicker}>SECURE · SYNCED · YOURS</Text>
              </View>
            </View>
          </Reveal>

          <Reveal index={1} style={styles.intro}>
            <Text accessibilityRole="header" style={styles.title}>{title}</Text>
            <Text style={styles.subtitle}>
              {mode === 'recover'
                ? 'We’ll email a secure link. Your password is handled by Supabase Auth and is never stored in your fitness data.'
                : 'Sign in to securely back up your plan, meals, workouts and progress with Supabase.'}
            </Text>
          </Reveal>

          {mode !== 'recover' ? (
            <Reveal index={2}>
              <Segmented onChange={changeMode} options={MODES} value={mode} />
            </Reveal>
          ) : (
            <Reveal index={2}>
              <Tap accessibilityLabel="Back to sign in" onPress={() => changeMode('login')} style={styles.backLink}>
                <Glyph color={palette.lime} name="chevron" size={14} />
                <Text style={styles.linkText}>Back to sign in</Text>
              </Tap>
            </Reveal>
          )}

          <Reveal index={3} style={styles.form}>
            {mode === 'signup' ? (
              <Field
                autoCapitalize="words"
                autoComplete="name"
                label="Your name"
                onChangeText={setDisplayName}
                placeholder="Sahil"
                value={displayName}
              />
            ) : null}
            <Field
              autoCapitalize="none"
              autoComplete="email"
              keyboardType="email-address"
              label="Email"
              onChangeText={setEmail}
              placeholder="you@example.com"
              value={email}
            />
            {mode !== 'recover' ? (
              <Field
                autoCapitalize="none"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                label="Password"
                onChangeText={setPassword}
                placeholder="At least 10 characters"
                secureTextEntry
                value={password}
              />
            ) : null}
          </Reveal>

          {mode === 'login' ? (
            <Tap accessibilityLabel="Forgot password" onPress={() => {
              setMode('recover');
              setError('');
              setMessage('');
            }} style={styles.forgot}>
              <Text style={styles.linkText}>Forgot password?</Text>
            </Tap>
          ) : null}

          <Reveal index={4} style={styles.status}>
            {error ? <Notice body={error} icon="alert" tone="danger" /> : null}
            {message ? <Notice body={message} icon="check" tone="accent" /> : null}
            {!serviceConfigured ? (
              <Notice
                body="Supabase is not configured in this build yet. Add the project URL and publishable key to enable accounts. Guest mode remains available."
                icon="info"
              />
            ) : (
              <Notice
                body="Supabase Auth protects your session. Row-level security keeps your fitness vault scoped to your user ID."
                icon="shield"
                tone="accent"
              />
            )}
          </Reveal>

          <Reveal index={5} style={styles.actions}>
            <PrimaryButton
              disabled={busy || !serviceConfigured}
              icon={mode === 'login' ? 'lock' : mode === 'signup' ? 'spark' : 'cloud'}
              label={mode === 'login' ? 'Sign in' : mode === 'signup' ? 'Create account' : 'Email reset link'}
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
              label="Use guest mode on this device"
              onPress={continueOffline}
            />
            <Text style={styles.offlineNote}>
              Guest data stays encrypted on this device. If you create an account later, Vigorly will merge it into your private cloud vault.
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

function Notice({
  body,
  icon,
  tone = 'info',
}: {
  body: string;
  icon: GlyphName;
  tone?: 'info' | 'danger' | 'accent';
}) {
  const hue = tone === 'danger' ? palette.danger : tone === 'accent' ? palette.lime : palette.info;
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
  title: { ...text.title, color: palette.ink, maxWidth: 340 },
  subtitle: { ...text.body, color: palette.inkMid, marginTop: space.sm, maxWidth: 350 },
  backLink: { flexDirection: 'row', alignItems: 'center', gap: 7, alignSelf: 'flex-start' },
  linkText: { ...text.row, color: palette.lime },
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
  forgot: { alignSelf: 'flex-end', paddingVertical: 10, paddingLeft: 16 },
  status: { gap: 10, marginTop: space.sm },
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
