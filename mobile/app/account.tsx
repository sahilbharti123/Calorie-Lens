import * as DocumentPicker from 'expo-document-picker';
import { File, Paths } from 'expo-file-system';
import * as Haptics from 'expo-haptics';
import { type Href, useRouter } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { type ComponentProps, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import {
  Bar,
  Card,
  GhostButton,
  ListRow,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  SectionTitle,
  Well,
} from '@/src/components/ui';
import { apiRequest } from '@/src/lib/api-client';
import { readRecoveryCode } from '@/src/lib/session';
import { mergeAppData, normalizeData, useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, tabular, text } from '@/src/theme';
import type { AppData, CoachMemory } from '@/src/types';

type BackupFile = {
  format: 'calorie-lens-backup-v1';
  exportedAt: string;
  payload: AppData;
};

type AIUsage = {
  used: number;
  limit: number;
  remaining: number;
  kinds: Record<string, { used: number; limit: number; remaining: number }>;
};

function commaList(value: string) {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 30);
}

export default function AccountScreen() {
  const router = useRouter();
  const {
    data,
    clearLocalData,
    replaceData,
    syncError,
    syncNow,
    syncState,
    updateCoachMemory,
  } = useApp();
  const {
    changePassword,
    deleteAccount,
    exitOfflineMode,
    restartOnboarding,
    session,
    signOut,
  } = useAuth();
  const [memory, setMemory] = useState({
    dietaryPreferences: data.coachMemory.dietaryPreferences.join(', '),
    injuries: data.coachMemory.injuries.join(', '),
    workoutPreferences: data.coachMemory.workoutPreferences.join(', '),
    coachingStyle: data.coachMemory.coachingStyle,
    notes: data.coachMemory.notes,
  });
  const [recoveryCode, setRecoveryCode] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState('');
  const [aiUsage, setAIUsage] = useState<AIUsage | null>(null);

  useEffect(() => {
    void readRecoveryCode(session?.user.id).then((value) => setRecoveryCode(value ?? ''));
  }, [session?.user.id]);

  useEffect(() => {
    if (!session) {
      setAIUsage(null);
      return;
    }
    void apiRequest<AIUsage>('/v1/ai/usage', {}, session)
      .then(setAIUsage)
      .catch(() => setAIUsage(null));
  }, [session]);

  const syncLabel = useMemo(() => {
    if (!session) return 'Encrypted on this device';
    if (syncState === 'syncing') return 'Syncing securely…';
    if (syncState === 'synced') return 'Up to date';
    if (syncState === 'error') return 'Saved offline · sync needs attention';
    return 'Saved offline';
  }, [session, syncState]);

  function saveMemory() {
    const next: Partial<CoachMemory> = {
      dietaryPreferences: commaList(memory.dietaryPreferences),
      injuries: commaList(memory.injuries),
      workoutPreferences: commaList(memory.workoutPreferences),
      coachingStyle: memory.coachingStyle.trim() || 'supportive and concise',
      notes: memory.notes.trim(),
    };
    updateCoachMemory(next);
    setMessage('Coach memory saved.');
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }

  async function exportBackup() {
    setBusy('export');
    setMessage('');
    try {
      const backup: BackupFile = {
        format: 'calorie-lens-backup-v1',
        exportedAt: new Date().toISOString(),
        payload: data,
      };
      const date = new Date().toISOString().slice(0, 10);
      const file = new File(Paths.cache, `calorie-lens-backup-${date}.json`);
      if (file.exists) file.delete();
      file.create();
      file.write(JSON.stringify(backup, null, 2));
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(file.uri, {
          mimeType: 'application/json',
          dialogTitle: 'Save Calorie Lens backup',
          UTI: 'public.json',
        });
      } else {
        setMessage(`Backup created at ${file.uri}`);
      }
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Could not export the backup.');
    } finally {
      setBusy('');
    }
  }

  async function importBackup() {
    setBusy('import');
    setMessage('');
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'application/json',
        copyToCacheDirectory: true,
      });
      if (result.canceled) return;
      const contents = await new File(result.assets[0].uri).text();
      const parsed = JSON.parse(contents) as Partial<BackupFile>;
      if (parsed.format !== 'calorie-lens-backup-v1' || !parsed.payload) {
        throw new Error('This is not a Calorie Lens backup.');
      }
      replaceData(mergeAppData(data, normalizeData(parsed.payload)));
      setMessage('Backup restored. It will sync to your account automatically.');
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Could not restore the backup.');
    } finally {
      setBusy('');
    }
  }

  async function updatePassword() {
    if (!currentPassword || newPassword.length < 10) {
      setMessage('Enter your current password and a new password of at least 10 characters.');
      return;
    }
    setBusy('password');
    try {
      await changePassword(currentPassword, newPassword);
      Alert.alert('Password changed', 'All sessions were signed out. Sign in again with your new password.');
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Could not change the password.');
    } finally {
      setBusy('');
    }
  }

  function confirmDelete() {
    Alert.alert(
      'Delete account and cloud data?',
      'This permanently removes your account and encrypted server vault. Export a backup first if you may need the data later.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete permanently',
          style: 'destructive',
          onPress: () => {
            setBusy('delete');
            void clearLocalData()
              .then(deleteAccount)
              .catch((reason) => setMessage(
                reason instanceof Error ? reason.message : 'Could not delete the account.',
              ))
              .finally(() => setBusy(''));
          },
        },
      ],
    );
  }

  function confirmClear() {
    Alert.alert(
      'Clear fitness history?',
      session
        ? 'This clears the app and the next sync will replace your cloud history with an empty vault.'
        : 'This permanently clears the encrypted fitness history on this device.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear history',
          style: 'destructive',
          onPress: () => void clearLocalData(),
        },
      ],
    );
  }

  const initials = (session?.user.displayName ?? 'Offline').slice(0, 2).toUpperCase();
  const aiRatio = aiUsage && aiUsage.limit > 0 ? aiUsage.used / aiUsage.limit : 0;

  return (
    <Screen edges={['bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}>

          {/* ---------- Identity + sync ---------- */}
          <Reveal>
            <Card glow raised>
              <View style={styles.identity}>
                <View style={styles.avatar}>
                  <Text style={styles.avatarText}>{initials}</Text>
                </View>
                <View style={styles.identityCopy}>
                  <Text numberOfLines={1} style={styles.name}>
                    {session?.user.displayName ?? 'Offline profile'}
                  </Text>
                  <Text numberOfLines={1} style={styles.email}>
                    {session?.user.email ?? 'No account connected'}
                  </Text>
                </View>
              </View>

              <View style={styles.pillRow}>
                <Pill
                  icon={session ? 'cloud' : 'lock'}
                  label={syncLabel}
                  tone={syncState === 'error' ? 'danger' : 'accent'}
                />
              </View>

              <Text style={styles.heroBody}>
                {session
                  ? 'Automatic encrypted sync is active for this account.'
                  : 'Local records are encrypted with a device-only key.'}
              </Text>

              {syncError ? (
                <Notice body={syncError} icon="alert" style={styles.heroNotice} tone="danger" />
              ) : null}

              {session ? (
                <PrimaryButton
                  compact
                  disabled={syncState === 'syncing'}
                  icon="cloud"
                  label="Sync now"
                  loading={syncState === 'syncing'}
                  onPress={() => void syncNow()}
                  style={styles.heroAction}
                />
              ) : null}
            </Card>
          </Reveal>

          {/* ---------- AI budget ---------- */}
          {session && aiUsage ? (
            <Reveal index={1} style={styles.gap}>
              <Card>
                <View style={styles.aiHead}>
                  <View style={styles.aiIcon}>
                    <Glyph color={palette.lime} name="spark" size={16} />
                  </View>
                  <View style={styles.flex}>
                    <Text style={styles.aiTitle}>
                      AI budget · {aiUsage.used}/{aiUsage.limit} today
                    </Text>
                    <Text style={styles.aiBody}>
                      Local logging is unlimited · voice {aiUsage.kinds.audio?.used ?? 0}/{aiUsage.kinds.audio?.limit ?? 0} · online coach {aiUsage.kinds.coach?.used ?? 0}/{aiUsage.kinds.coach?.limit ?? 0}
                    </Text>
                  </View>
                </View>
                <Bar style={styles.aiBar} value={aiRatio} />
              </Card>
            </Reveal>
          ) : null}

          {/* ---------- Fitness profile ---------- */}
          <Reveal index={2}>
            <SectionTitle title="Fitness profile" />
            <Card padded={false} style={styles.rowCard}>
              <Well style={styles.planWell}>
                <Text style={styles.planLabel}>CURRENT PLAN</Text>
                <Text style={styles.planValue}>{data.plan.summary}</Text>
              </Well>
              <ListRow
                detail="Adjust targets and bowl size"
                icon="target"
                onPress={() => router.push('/settings')}
                title="Goals & calibration"
              />
              <ListRow
                detail="Recalculate from your goal, body, routine and food"
                icon="spark"
                last
                onPress={() => {
                  void restartOnboarding().then(
                    () => router.replace('/onboarding' as Href),
                  );
                }}
                title="Personalize again"
              />
            </Card>
          </Reveal>

          {/* ---------- Coach memory ---------- */}
          <Reveal index={3}>
            <SectionTitle title="Coach memory" />
            <Card>
              <Text style={styles.cardIntro}>
                These details follow you across devices and help the coach give relevant advice. Separate items with commas.
              </Text>
              <View style={styles.fields}>
                <Field
                  label="Dietary preferences"
                  onChangeText={(value) => setMemory((current) => ({ ...current, dietaryPreferences: value }))}
                  placeholder="vegetarian, high protein"
                  value={memory.dietaryPreferences}
                />
                <Field
                  label="Injuries or limits"
                  onChangeText={(value) => setMemory((current) => ({ ...current, injuries: value }))}
                  placeholder="sensitive left knee"
                  value={memory.injuries}
                />
                <Field
                  label="Workout preferences"
                  onChangeText={(value) => setMemory((current) => ({ ...current, workoutPreferences: value }))}
                  placeholder="strength, morning walks"
                  value={memory.workoutPreferences}
                />
                <Field
                  label="Coaching style"
                  onChangeText={(value) => setMemory((current) => ({ ...current, coachingStyle: value }))}
                  placeholder="direct but supportive"
                  value={memory.coachingStyle}
                />
                <Field
                  label="Anything else to remember"
                  multiline
                  onChangeText={(value) => setMemory((current) => ({ ...current, notes: value }))}
                  placeholder="My usual schedule, equipment, or motivation"
                  value={memory.notes}
                />
              </View>
              <PrimaryButton
                compact
                icon="check"
                label="Save coach memory"
                onPress={saveMemory}
                style={styles.cardAction}
              />
            </Card>
          </Reveal>

          {/* ---------- Backup & recovery ---------- */}
          <Reveal index={4}>
            <SectionTitle title="Backup & recovery" />
            <Card padded={false} style={styles.rowCard}>
              <ListRow
                detail="Portable JSON copy of all fitness records"
                icon="download"
                last={!recoveryCode}
                onPress={busy === 'export' ? undefined : () => void exportBackup()}
                right={busy === 'export'
                  ? <ActivityIndicator color={palette.lime} size="small" />
                  : undefined}
                title="Export a backup"
              />
              <ListRow
                detail="Import and merge into this device and account"
                icon="folder"
                last
                onPress={busy === 'import' ? undefined : () => void importBackup()}
                right={busy === 'import'
                  ? <ActivityIndicator color={palette.lime} size="small" />
                  : undefined}
                title="Restore a backup"
              />
            </Card>
            {recoveryCode ? (
              <Card style={styles.recovery}>
                <View style={styles.recoveryHead}>
                  <Glyph color={palette.lime} name="lock" size={14} />
                  <Text style={styles.recoveryLabel}>ACCOUNT RECOVERY CODE</Text>
                </View>
                <Well style={styles.recoveryWell}>
                  <Text selectable style={styles.recoveryCode}>{recoveryCode}</Text>
                </Well>
                <Text style={styles.recoveryHelp}>
                  Keep a second copy somewhere private. It is not included in fitness backups.
                </Text>
              </Card>
            ) : null}
          </Reveal>

          {/* ---------- Security ---------- */}
          {session ? (
            <Reveal index={5}>
              <SectionTitle title="Security" />
              <Card>
                <View style={styles.fields}>
                  <Field
                    label="Current password"
                    onChangeText={setCurrentPassword}
                    secureTextEntry
                    value={currentPassword}
                  />
                  <Field
                    label="New password"
                    onChangeText={setNewPassword}
                    placeholder="At least 10 characters"
                    secureTextEntry
                    value={newPassword}
                  />
                </View>
                <PrimaryButton
                  compact
                  disabled={busy === 'password'}
                  icon="lock"
                  label="Change password"
                  loading={busy === 'password'}
                  onPress={() => void updatePassword()}
                  style={styles.cardAction}
                />
              </Card>
            </Reveal>
          ) : null}

          {message ? (
            <Reveal index={6} style={styles.gapLarge}>
              <Notice body={message} icon="info" tone="info" />
            </Reveal>
          ) : null}

          {/* ---------- Account actions ---------- */}
          <Reveal index={7}>
            <SectionTitle title="Account actions" />
            <Card padded={false} style={styles.rowCard}>
              {session ? (
                <ListRow
                  detail="Your local encrypted copy stays on this device"
                  icon="logout"
                  last
                  onPress={() => void signOut()}
                  title="Sign out"
                />
              ) : (
                <ListRow
                  detail="Enable recovery and cross-device sync"
                  icon="cloud"
                  last
                  onPress={exitOfflineMode}
                  title="Connect an account"
                />
              )}
            </Card>

            <View style={styles.danger}>
              <GhostButton
                icon="trash"
                label="Clear fitness history"
                onPress={confirmClear}
                tone="danger"
              />
              <Text style={styles.dangerNote}>
                Erase meals, workouts, water, progress and coach memory
              </Text>

              {session ? (
                <>
                  <GhostButton
                    icon="trash"
                    label={busy === 'delete' ? 'Deleting…' : 'Delete account'}
                    onPress={() => {
                      if (busy === 'delete') return;
                      confirmDelete();
                    }}
                    style={styles.dangerButton}
                    tone="danger"
                  />
                  <Text style={styles.dangerNote}>
                    Permanently remove the account and cloud vault
                  </Text>
                </>
              ) : null}
            </View>
          </Reveal>

          <Reveal index={8}>
            <Text style={styles.footnote}>
              Local fitness data uses authenticated encryption. Account sessions and recovery data are held in the platform secure credential store.
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
        style={[
          styles.input,
          props.multiline && styles.inputMultiline,
          focused && styles.inputFocused,
        ]}
      />
    </View>
  );
}

/** Inline message card — never a bare line of red text. */
function Notice({
  body,
  icon,
  style,
  tone = 'info',
}: {
  body: string;
  icon: GlyphName;
  style?: ComponentProps<typeof View>['style'];
  tone?: 'info' | 'danger' | 'accent';
}) {
  const hue = tone === 'danger' ? palette.danger : tone === 'accent' ? palette.lime : palette.info;
  return (
    <View style={[styles.notice, { borderColor: `${hue}33`, backgroundColor: `${hue}10` }, style]}>
      <Glyph color={hue} name={icon} size={15} />
      <Text style={[styles.noticeText, { color: hue }]}>{body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { paddingHorizontal: space.md, paddingTop: space.sm, paddingBottom: space.xxl },
  gap: { marginTop: space.sm },
  gapLarge: { marginTop: space.lg },

  identity: { flexDirection: 'row', alignItems: 'center', gap: 13 },
  identityCopy: { flex: 1 },
  avatar: {
    width: 54,
    height: 54,
    borderRadius: 19,
    backgroundColor: palette.limeSoft,
    borderWidth: 1,
    borderColor: `${palette.lime}33`,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: { ...text.headline, fontSize: 17, color: palette.lime },
  name: { ...text.headline, color: palette.ink },
  email: { ...text.caption, color: palette.inkMid, marginTop: 3 },
  pillRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 7, marginTop: 16 },
  heroBody: { ...text.caption, color: palette.inkMid, marginTop: 10 },
  heroNotice: { marginTop: 12 },
  heroAction: { marginTop: 16 },

  aiHead: { flexDirection: 'row', alignItems: 'flex-start', gap: 11 },
  aiIcon: {
    width: 32,
    height: 32,
    borderRadius: 11,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  aiTitle: { ...text.row, color: palette.ink, ...tabular },
  aiBody: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: 3, ...tabular },
  aiBar: { marginTop: 13 },

  rowCard: { paddingHorizontal: 14 },
  planWell: { marginTop: 12, marginBottom: 2 },
  planLabel: { ...text.label, color: palette.inkLow },
  planValue: { ...text.value, color: palette.ink, marginTop: 6, ...tabular },

  cardIntro: { ...text.caption, color: palette.inkMid },
  cardAction: { marginTop: 16 },

  fields: { gap: 12, marginTop: 14 },
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
  inputMultiline: { minHeight: 92, textAlignVertical: 'top' },
  inputFocused: { borderColor: palette.lime },

  recovery: { marginTop: space.sm },
  recoveryHead: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  recoveryLabel: { ...text.label, color: palette.lime },
  recoveryWell: { marginTop: 11 },
  recoveryCode: { ...text.value, fontSize: 14, letterSpacing: 0.7, color: palette.lime, ...tabular },
  recoveryHelp: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: 11 },

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

  danger: { marginTop: space.md, gap: 8 },
  dangerButton: { marginTop: 10 },
  dangerNote: { ...text.caption, fontSize: 11, color: palette.inkLow, paddingHorizontal: 4 },

  footnote: {
    ...text.caption,
    fontSize: 10.5,
    color: palette.inkLow,
    textAlign: 'center',
    paddingHorizontal: space.md,
    marginTop: space.xl,
  },
});
