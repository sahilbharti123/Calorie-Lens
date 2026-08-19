import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { BrandMark } from '@/src/components/brand-mark';
import { Glyph } from '@/src/components/glyph';
import { GhostButton, PrimaryButton, Screen } from '@/src/components/ui';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, text } from '@/src/theme';

export default function AuthResetScreen() {
  const router = useRouter();
  const {
    completeRecoveredPassword,
    passwordRecoveryReady,
    recoveryError,
    session,
  } = useAuth();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function savePassword() {
    setError('');
    if (password.length < 10) {
      setError('Use at least 10 characters for your new password.');
      return;
    }
    if (password !== confirmPassword) {
      setError('The passwords do not match.');
      return;
    }
    setBusy(true);
    try {
      await completeRecoveredPassword(password);
      router.replace('/(tabs)');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not update your password.');
    } finally {
      setBusy(false);
    }
  }

  const ready = passwordRecoveryReady && Boolean(session);
  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          onTouchMove={Keyboard.dismiss}
          showsVerticalScrollIndicator={false}>
          <View style={styles.brand}>
            <BrandMark size={60} />
            <Text style={styles.wordmark}>VIGORLY</Text>
          </View>

          <View style={styles.hero}>
            <Text style={styles.eyebrow}>ACCOUNT RECOVERY</Text>
            <Text accessibilityRole="header" style={styles.title}>
              {ready ? 'Choose a new password.' : 'Open your secure reset link.'}
            </Text>
            <Text style={styles.subtitle}>
              {ready
                ? 'This updates your Supabase account immediately. Use the new password on every device.'
                : 'Request a reset from the sign-in screen, then open the email link on this phone.'}
            </Text>
          </View>

          {ready ? (
            <View style={styles.form}>
              <PasswordField label="New password" onChangeText={setPassword} value={password} />
              <PasswordField label="Confirm password" onChangeText={setConfirmPassword} value={confirmPassword} />
              {error ? <Notice body={error} danger /> : null}
              <PrimaryButton
                disabled={busy}
                icon="lock"
                label="Save new password"
                loading={busy}
                onPress={() => void savePassword()}
              />
            </View>
          ) : (
            <View style={styles.waiting}>
              <View style={styles.iconWell}>
                <Glyph color={recoveryError ? palette.danger : palette.lime} name={recoveryError ? 'alert' : 'link'} size={22} />
              </View>
              <Text style={styles.waitingTitle}>{recoveryError ? 'That link could not be verified' : 'Waiting for a verified link'}</Text>
              <Text style={styles.waitingBody}>
                {recoveryError || 'For your security, a password can only be changed after Supabase verifies the reset link.'}
              </Text>
              <GhostButton
                icon="chevron"
                label="Return to sign in"
                onPress={() => router.replace('/auth')}
                style={styles.returnButton}
              />
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </Screen>
  );
}

function PasswordField({
  label,
  onChangeText,
  value,
}: {
  label: string;
  onChangeText: (value: string) => void;
  value: string;
}) {
  const [focused, setFocused] = useState(false);
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label.toUpperCase()}</Text>
      <TextInput
        accessibilityLabel={label}
        autoCapitalize="none"
        autoComplete="new-password"
        onBlur={() => setFocused(false)}
        onChangeText={onChangeText}
        onFocus={() => setFocused(true)}
        placeholder="At least 10 characters"
        placeholderTextColor={palette.inkLow}
        secureTextEntry
        style={[styles.input, focused && styles.inputFocused]}
        value={value}
      />
    </View>
  );
}

function Notice({ body, danger = false }: { body: string; danger?: boolean }) {
  const hue = danger ? palette.danger : palette.info;
  return (
    <View style={[styles.notice, { borderColor: `${hue}33`, backgroundColor: `${hue}10` }]}>
      <Glyph color={hue} name={danger ? 'alert' : 'info'} size={15} />
      <Text style={[styles.noticeText, { color: hue }]}>{body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { flexGrow: 1, padding: space.md, paddingBottom: space.xxl },
  brand: { flexDirection: 'row', alignItems: 'center', gap: 13 },
  wordmark: { ...text.label, color: palette.ink, letterSpacing: 2.2 },
  hero: { marginTop: space.xxl },
  eyebrow: { ...text.label, color: palette.lime, marginBottom: 10 },
  title: { ...text.title, color: palette.ink, maxWidth: 350 },
  subtitle: { ...text.body, color: palette.inkMid, marginTop: space.sm, maxWidth: 350 },
  form: { gap: 14, marginTop: space.xl },
  field: { gap: 7 },
  fieldLabel: { ...text.label, color: palette.inkLow },
  input: {
    ...text.row,
    minHeight: 54,
    color: palette.ink,
    borderWidth: 1,
    borderColor: palette.line,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceLo,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  inputFocused: { borderColor: palette.lime },
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
  waiting: {
    marginTop: space.xl,
    borderWidth: 1,
    borderColor: palette.line,
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    padding: space.lg,
    alignItems: 'flex-start',
  },
  iconWell: {
    width: 48,
    height: 48,
    borderRadius: 16,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  waitingTitle: { ...text.headline, color: palette.ink, marginTop: 16 },
  waitingBody: { ...text.body, color: palette.inkMid, marginTop: 7 },
  returnButton: { marginTop: 20, alignSelf: 'stretch' },
});
