import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from 'expo-audio';
import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
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
import { parseFitnessCommand, parseVoiceCommand } from '@/src/lib/nutrition';
import { slotLabels } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';
import type { LogOperation, MealSlot, ParsedCommand } from '@/src/types';

export default function QuickLogScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ slot?: MealSlot; prefill?: string }>();
  const { applyOperations } = useApp();
  const [text, setText] = useState(params.prefill ?? '');
  const [parsed, setParsed] = useState<ParsedCommand | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder, 120);

  useEffect(() => {
    void setAudioModeAsync({ playsInSilentMode: true, allowsRecording: true });
  }, []);

  async function understandTyped() {
    if (!text.trim()) return;
    setBusy(true);
    setError('');
    try {
      setParsed(await parseFitnessCommand(text.trim(), params.slot));
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not understand that update.');
    } finally {
      setBusy(false);
    }
  }

  async function toggleRecording() {
    setError('');
    if (recorderState.isRecording) {
      setBusy(true);
      try {
        await recorder.stop();
        if (!recorder.uri) throw new Error('No recording was created.');
        setParsed(await parseVoiceCommand(recorder.uri, params.slot));
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Voice logging failed.');
      } finally {
        setBusy(false);
      }
      return;
    }

    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) {
      setError('Microphone access is needed for voice logging.');
      return;
    }
    await recorder.prepareToRecordAsync();
    recorder.record();
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  }

  function confirm() {
    if (!parsed) return;
    applyOperations(parsed.operations);
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    router.dismiss();
  }

  function reset() {
    setParsed(null);
    setError('');
  }

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <View style={styles.intro}>
            <View style={styles.spark}><Glyph name="spark" color={palette.forest} size={23} /></View>
            <Text style={styles.title}>Log anything in one go.</Text>
            <Text style={styles.subtitle}>
              Meals, water, workouts, steps, sleep or weight — say it naturally.
              {params.slot ? ` This entry will start in ${slotLabels[params.slot]}.` : ''}
            </Text>
          </View>

          {!parsed ? (
            <>
              <View style={styles.composer}>
                <TextInput
                  autoFocus={Boolean(params.prefill)}
                  multiline
                  value={text}
                  onChangeText={setText}
                  placeholder="e.g. I had 2 rotis and a bowl of dal for lunch"
                  placeholderTextColor="#8A938B"
                  style={styles.input}
                />
                <Pressable
                  disabled={busy || !text.trim()}
                  onPress={understandTyped}
                  style={({ pressed }) => [
                    styles.parseButton,
                    (!text.trim() || busy) && styles.disabled,
                    pressed && { opacity: 0.85 },
                  ]}>
                  {busy ? <ActivityIndicator color={palette.lime} /> : <Text style={styles.parseText}>Review update</Text>}
                </Pressable>
              </View>

              <View style={styles.orRow}><View style={styles.orLine} /><Text style={styles.orText}>OR USE VOICE</Text><View style={styles.orLine} /></View>

              <View style={styles.voicePanel}>
                <Pressable
                  disabled={busy}
                  onPress={toggleRecording}
                  accessibilityLabel={recorderState.isRecording ? 'Stop recording' : 'Start recording'}
                  style={[styles.micButton, recorderState.isRecording && styles.micRecording]}>
                  {busy ? (
                    <ActivityIndicator color={palette.forest} />
                  ) : (
                    <Glyph name="mic" color={palette.forest} size={31} />
                  )}
                </Pressable>
                <Text style={styles.voiceTitle}>
                  {busy ? 'Understanding…' : recorderState.isRecording ? 'Listening… tap to finish' : 'Tap and speak naturally'}
                </Text>
                <Text style={styles.voiceMeta}>
                  {recorderState.isRecording
                    ? `${Math.floor(recorderState.durationMillis / 1000)} seconds`
                    : 'English, Hindi and Hinglish are supported by the configured AI service.'}
                </Text>
              </View>

              <View style={styles.examples}>
                <Text style={styles.examplesLabel}>TRY SAYING</Text>
                <Text style={styles.example}>“2 glasses of water and 8,000 steps”</Text>
                <Text style={styles.example}>“Had poha and chai for breakfast”</Text>
                <Text style={styles.example}>“45 minute hard leg workout”</Text>
              </View>
            </>
          ) : (
            <View>
              <View style={styles.reviewHeader}>
                <View>
                  <Text style={styles.reviewEyebrow}>{parsed.source === 'ai' ? 'AI ESTIMATE' : 'ON-DEVICE ESTIMATE'}</Text>
                  <Text style={styles.reviewTitle}>{parsed.confirmation}</Text>
                </View>
                <Pressable onPress={reset}><Text style={styles.editText}>Edit</Text></Pressable>
              </View>

              <View style={styles.transcript}>
                <Text style={styles.transcriptLabel}>YOU SAID</Text>
                <Text style={styles.transcriptText}>“{parsed.transcript}”</Text>
              </View>

              <View style={styles.operations}>
                {parsed.operations.map((operation, index) => (
                  <OperationCard key={`${operation.type}-${index}`} operation={operation} />
                ))}
              </View>

              <View style={styles.notice}>
                <Glyph name="spark" color={palette.limeDark} size={18} />
                <Text style={styles.noticeText}>Calories are estimates. Nothing is saved until you confirm.</Text>
              </View>
              <Pressable onPress={confirm} style={styles.confirmButton}>
                <Text style={styles.confirmText}>Confirm and log</Text>
              </Pressable>
            </View>
          )}

          {error ? <Text style={styles.error}>{error}</Text> : null}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function OperationCard({ operation }: { operation: LogOperation }) {
  let icon: 'bowl' | 'water' | 'dumbbell' | 'steps' | 'sleep' | 'chart' = 'chart';
  let title = '';
  let detail = '';
  if (operation.type === 'meal') {
    icon = 'bowl';
    title = slotLabels[operation.slot];
    detail = `${operation.items.map((item) => item.name).join(', ')} · ${operation.items.reduce((sum, item) => sum + item.calories, 0)} kcal`;
  } else if (operation.type === 'water') {
    icon = 'water';
    title = 'Water';
    detail = `${operation.action === 'add' ? '+' : ''}${operation.amount} ml`;
  } else if (operation.type === 'workout') {
    icon = 'dumbbell';
    title = operation.name;
    detail = `${operation.durationMin} min · ~${operation.calories} kcal`;
  } else if (operation.type === 'steps') {
    icon = 'steps';
    title = 'Steps';
    detail = `${operation.amount.toLocaleString()} steps`;
  } else if (operation.type === 'sleep') {
    icon = 'sleep';
    title = 'Sleep';
    detail = `${operation.amount} hours`;
  } else if (operation.type === 'weight') {
    title = 'Weight';
    detail = `${operation.amount} kg`;
  }
  return (
    <View style={styles.operation}>
      <View style={styles.operationIcon}><Glyph name={icon} color={palette.forest} size={20} /></View>
      <View style={{ flex: 1 }}>
        <Text style={styles.operationTitle}>{title}</Text>
        <Text style={styles.operationDetail}>{detail}</Text>
      </View>
      <Text style={styles.ready}>READY</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { padding: space.md, paddingBottom: 34 },
  intro: { alignItems: 'center', paddingTop: 8, paddingHorizontal: 18, marginBottom: 22 },
  spark: { width: 46, height: 46, borderRadius: 15, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  title: { color: palette.ink, fontFamily: type.demi, fontSize: 24, letterSpacing: -0.7, textAlign: 'center' },
  subtitle: { color: palette.muted, fontFamily: type.regular, fontSize: 12, lineHeight: 18, textAlign: 'center', marginTop: 6 },
  composer: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 12 },
  input: { minHeight: 112, color: palette.ink, fontFamily: type.medium, fontSize: 15, lineHeight: 22, textAlignVertical: 'top', padding: 5 },
  parseButton: { height: 48, borderRadius: radius.sm, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  parseText: { color: palette.lime, fontFamily: type.demi, fontSize: 13 },
  disabled: { opacity: 0.45 },
  orRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginVertical: 19 },
  orLine: { flex: 1, height: 1, backgroundColor: palette.line },
  orText: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2 },
  voicePanel: { minHeight: 180, backgroundColor: palette.forest, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', padding: 20 },
  micButton: { width: 70, height: 70, borderRadius: 35, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center', borderWidth: 0 },
  micRecording: { borderWidth: 8, borderColor: '#40503F' },
  voiceTitle: { color: palette.white, fontFamily: type.demi, fontSize: 14, marginTop: 13 },
  voiceMeta: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 10, lineHeight: 15, textAlign: 'center', marginTop: 4, maxWidth: 270 },
  examples: { marginTop: 18, gap: 8 },
  examplesLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2 },
  example: { color: palette.ink, fontFamily: type.medium, fontSize: 11, paddingVertical: 4 },
  reviewHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 13 },
  reviewEyebrow: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2 },
  reviewTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 19, marginTop: 3 },
  editText: { color: palette.coral, fontFamily: type.demi, fontSize: 12 },
  transcript: { backgroundColor: palette.softLime, borderRadius: radius.md, padding: 15, marginBottom: 12 },
  transcriptLabel: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.1 },
  transcriptText: { color: palette.ink, fontFamily: type.medium, fontSize: 13, lineHeight: 20, marginTop: 5 },
  operations: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 14 },
  operation: { minHeight: 70, flexDirection: 'row', alignItems: 'center', gap: 11, borderBottomWidth: 1, borderBottomColor: palette.line },
  operationIcon: { width: 38, height: 38, borderRadius: 12, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  operationTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 13 },
  operationDetail: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 2 },
  ready: { color: palette.limeDark, fontFamily: type.demi, fontSize: 8, letterSpacing: 1 },
  notice: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 13, marginTop: 12 },
  noticeText: { flex: 1, color: palette.muted, fontFamily: type.regular, fontSize: 10, lineHeight: 15 },
  confirmButton: { height: 54, borderRadius: radius.md, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  confirmText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
  error: { color: palette.coral, fontFamily: type.medium, fontSize: 11, lineHeight: 16, textAlign: 'center', marginTop: 14 },
});
