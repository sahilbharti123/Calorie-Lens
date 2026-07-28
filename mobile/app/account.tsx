import * as DocumentPicker from 'expo-document-picker';
import { File, Paths } from 'expo-file-system';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useEffect, useMemo, useState } from 'react';
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

import { Glyph, type GlyphName } from '@/src/components/glyph';
import { readRecoveryCode } from '@/src/lib/session';
import { mergeAppData, normalizeData, useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, type } from '@/src/theme';
import type { AppData, CoachMemory } from '@/src/types';

type BackupFile = {
  format: 'calorie-lens-backup-v1';
  exportedAt: string;
  payload: AppData;
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

  useEffect(() => {
    void readRecoveryCode(session?.user.id).then((value) => setRecoveryCode(value ?? ''));
  }, [session?.user.id]);

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

  return (
    <SafeAreaView edges={['bottom']} style={styles.safe}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled">
          <View style={styles.identity}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>
                {(session?.user.displayName ?? 'Offline').slice(0, 2).toUpperCase()}
              </Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.name}>{session?.user.displayName ?? 'Offline profile'}</Text>
              <Text style={styles.email}>{session?.user.email ?? 'No account connected'}</Text>
            </View>
            <View style={[styles.syncDot, syncState === 'error' && styles.syncDotError]} />
          </View>

          <View style={styles.syncCard}>
            <View style={styles.syncIcon}><Glyph name="spark" color={palette.forest} size={19} /></View>
            <View style={{ flex: 1 }}>
              <Text style={styles.cardTitle}>{syncLabel}</Text>
              <Text style={styles.cardBody}>
                {session
                  ? 'Automatic encrypted sync is active for this account.'
                  : 'Local records are encrypted with a device-only key.'}
              </Text>
              {syncError ? <Text style={styles.inlineError}>{syncError}</Text> : null}
            </View>
            {session ? (
              <Pressable disabled={syncState === 'syncing'} onPress={() => void syncNow()} style={styles.smallButton}>
                <Text style={styles.smallButtonText}>Sync</Text>
              </Pressable>
            ) : null}
          </View>

          <SectionLabel text="FITNESS PROFILE" />
          <ActionRow
            icon="chart"
            title="Goals & calibration"
            body="Calories, protein, water, steps, weight and bowl size"
            onPress={() => router.push('/settings')}
          />

          <SectionLabel text="COACH MEMORY" />
          <View style={styles.panel}>
            <Text style={styles.panelIntro}>
              These details follow you across devices and help the coach give relevant advice. Separate items with commas.
            </Text>
            <MemoryField label="Dietary preferences" value={memory.dietaryPreferences} onValueChange={(value) => setMemory((current) => ({ ...current, dietaryPreferences: value }))} placeholder="vegetarian, high protein" />
            <MemoryField label="Injuries or limits" value={memory.injuries} onValueChange={(value) => setMemory((current) => ({ ...current, injuries: value }))} placeholder="sensitive left knee" />
            <MemoryField label="Workout preferences" value={memory.workoutPreferences} onValueChange={(value) => setMemory((current) => ({ ...current, workoutPreferences: value }))} placeholder="strength, morning walks" />
            <MemoryField label="Coaching style" value={memory.coachingStyle} onValueChange={(value) => setMemory((current) => ({ ...current, coachingStyle: value }))} placeholder="direct but supportive" />
            <MemoryField label="Anything else to remember" value={memory.notes} onValueChange={(value) => setMemory((current) => ({ ...current, notes: value }))} placeholder="My usual schedule, equipment, or motivation" multiline />
            <Pressable onPress={saveMemory} style={styles.panelButton}><Text style={styles.panelButtonText}>Save coach memory</Text></Pressable>
          </View>

          <SectionLabel text="BACKUP & RECOVERY" />
          <View style={styles.actionGroup}>
            <ActionRow icon="spark" title="Export a backup" body="Portable JSON copy of all fitness records" busy={busy === 'export'} onPress={() => void exportBackup()} />
            <ActionRow icon="plus" title="Restore a backup" body="Import and merge into this device and account" busy={busy === 'import'} onPress={() => void importBackup()} />
          </View>
          {recoveryCode ? (
            <View style={styles.recovery}>
              <Text style={styles.recoveryLabel}>ACCOUNT RECOVERY CODE</Text>
              <Text selectable style={styles.recoveryCode}>{recoveryCode}</Text>
              <Text style={styles.recoveryHelp}>Keep a second copy somewhere private. It is not included in fitness backups.</Text>
            </View>
          ) : null}

          {session ? (
            <>
              <SectionLabel text="SECURITY" />
              <View style={styles.panel}>
                <MemoryField label="Current password" value={currentPassword} onValueChange={setCurrentPassword} secureTextEntry />
                <MemoryField label="New password" value={newPassword} onValueChange={setNewPassword} secureTextEntry placeholder="At least 10 characters" />
                <Pressable disabled={busy === 'password'} onPress={() => void updatePassword()} style={styles.panelButton}>
                  {busy === 'password' ? <ActivityIndicator color={palette.lime} /> : <Text style={styles.panelButtonText}>Change password</Text>}
                </Pressable>
              </View>
            </>
          ) : null}

          {message ? <Text style={styles.message}>{message}</Text> : null}

          <SectionLabel text="ACCOUNT ACTIONS" />
          <View style={styles.actionGroup}>
            {session ? (
              <ActionRow icon="chevron" title="Sign out" body="Your local encrypted copy stays on this device" onPress={() => void signOut()} />
            ) : (
              <ActionRow icon="chevron" title="Connect an account" body="Enable recovery and cross-device sync" onPress={exitOfflineMode} />
            )}
            <ActionRow icon="trash" title="Clear fitness history" body="Erase meals, workouts, water, progress and coach memory" destructive onPress={confirmClear} />
            {session ? (
              <ActionRow icon="trash" title="Delete account" body="Permanently remove the account and cloud vault" destructive busy={busy === 'delete'} onPress={confirmDelete} />
            ) : null}
          </View>

          <Text style={styles.footnote}>
            Local fitness data uses authenticated encryption. Account sessions and recovery data are held in the platform secure credential store.
          </Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function SectionLabel({ text }: { text: string }) {
  return <Text style={styles.sectionLabel}>{text}</Text>;
}

function ActionRow({
  body,
  busy,
  destructive,
  icon,
  onPress,
  title,
}: {
  body: string;
  busy?: boolean;
  destructive?: boolean;
  icon: GlyphName;
  onPress: () => void;
  title: string;
}) {
  return (
    <Pressable disabled={busy} onPress={onPress} style={styles.actionRow}>
      <View style={[styles.actionIcon, destructive && styles.actionIconDestructive]}>
        {busy
          ? <ActivityIndicator size="small" color={palette.forest} />
          : <Glyph name={icon} color={destructive ? palette.coral : palette.forest} size={18} />}
      </View>
      <View style={{ flex: 1 }}>
        <Text style={[styles.actionTitle, destructive && { color: palette.coral }]}>{title}</Text>
        <Text style={styles.actionBody}>{body}</Text>
      </View>
      <Glyph name="chevron" color={palette.muted} size={15} />
    </Pressable>
  );
}

function MemoryField({
  label,
  onValueChange,
  value,
  ...props
}: {
  label: string;
  onValueChange: (value: string) => void;
  value: string;
} & Omit<React.ComponentProps<typeof TextInput>, 'onChange' | 'onChangeText' | 'value'>) {
  return (
    <View style={styles.memoryField}>
      <Text style={styles.memoryLabel}>{label}</Text>
      <TextInput
        {...props}
        onChangeText={onValueChange}
        placeholderTextColor="#929A93"
        style={[styles.memoryInput, props.multiline && { minHeight: 74, textAlignVertical: 'top' }]}
        value={value}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { padding: space.md, paddingBottom: 42 },
  identity: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 14 },
  avatar: { width: 52, height: 52, borderRadius: 18, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
  name: { color: palette.ink, fontFamily: type.demi, fontSize: 19, letterSpacing: -0.4 },
  email: { color: palette.muted, fontFamily: type.regular, fontSize: 11, marginTop: 2 },
  syncDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: palette.limeDark },
  syncDotError: { backgroundColor: palette.coral },
  syncCard: { flexDirection: 'row', gap: 11, alignItems: 'center', backgroundColor: palette.softLime, borderRadius: radius.md, padding: 14, marginBottom: 20 },
  syncIcon: { width: 38, height: 38, borderRadius: 12, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center' },
  cardTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 13 },
  cardBody: { color: palette.muted, fontFamily: type.regular, fontSize: 10, lineHeight: 15, marginTop: 2 },
  inlineError: { color: palette.coral, fontFamily: type.regular, fontSize: 9.5, marginTop: 3 },
  smallButton: { paddingHorizontal: 13, height: 35, borderRadius: radius.pill, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  smallButtonText: { color: palette.lime, fontFamily: type.demi, fontSize: 10 },
  sectionLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.3, marginBottom: 8, marginTop: 8 },
  panel: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14, marginBottom: 18 },
  panelIntro: { color: palette.muted, fontFamily: type.regular, fontSize: 10.5, lineHeight: 16, marginBottom: 7 },
  memoryField: { marginTop: 10 },
  memoryLabel: { color: palette.ink, fontFamily: type.medium, fontSize: 11, marginBottom: 6 },
  memoryInput: { minHeight: 44, borderWidth: 1, borderColor: palette.line, borderRadius: radius.sm, backgroundColor: palette.canvas, paddingHorizontal: 12, paddingVertical: 10, color: palette.ink, fontFamily: type.regular, fontSize: 12 },
  panelButton: { height: 49, borderRadius: radius.sm, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center', marginTop: 14 },
  panelButtonText: { color: palette.lime, fontFamily: type.demi, fontSize: 12 },
  actionGroup: { overflow: 'hidden', backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, marginBottom: 18 },
  actionRow: { minHeight: 70, flexDirection: 'row', alignItems: 'center', gap: 11, paddingHorizontal: 13, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: palette.line },
  actionIcon: { width: 34, height: 34, borderRadius: 11, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  actionIconDestructive: { backgroundColor: palette.softCoral },
  actionTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 12 },
  actionBody: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 14, marginTop: 2 },
  recovery: { backgroundColor: palette.forest, borderRadius: radius.md, padding: 16, marginBottom: 18 },
  recoveryLabel: { color: '#AEB9B0', fontFamily: type.demi, fontSize: 9, letterSpacing: 1.3 },
  recoveryCode: { color: palette.lime, fontFamily: type.demi, fontSize: 14, marginTop: 9, letterSpacing: 0.7 },
  recoveryHelp: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 9.5, lineHeight: 14, marginTop: 8 },
  message: { color: palette.ink, fontFamily: type.medium, fontSize: 11, lineHeight: 16, backgroundColor: palette.softLime, borderRadius: radius.sm, padding: 11, marginBottom: 14 },
  footnote: { color: palette.muted, fontFamily: type.regular, fontSize: 9, lineHeight: 14, textAlign: 'center', paddingHorizontal: 20, marginTop: 4 },
});
