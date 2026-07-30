import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from 'expo-audio';
import * as Device from 'expo-device';
import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
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
import { readServiceHealth } from '@/src/lib/api-client';
import { parseFitnessCommand, parseVoiceCommand } from '@/src/lib/nutrition';
import { slotLabels } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, type } from '@/src/theme';
import type { EstimationContext, LogOperation, MealSlot, ParsedCommand } from '@/src/types';

export default function QuickLogScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ slot?: MealSlot; prefill?: string }>();
  const { applyOperations, data, updateEstimationProfile } = useApp();
  const { session } = useAuth();
  const [text, setText] = useState(params.prefill ?? '');
  const [showKeyboard, setShowKeyboard] = useState(Boolean(params.prefill));
  const [parsed, setParsed] = useState<ParsedCommand | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [voiceStatus, setVoiceStatus] = useState<{
    ready: boolean;
    title: string;
    detail: string;
  }>({
    ready: false,
    title: 'Checking voice setup…',
    detail: 'Typed logging is always available.',
  });
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder, 120);
  const context = useMemo<EstimationContext>(() => ({
    weightKg: data.weights.at(-1)?.kg,
    bowlMl: data.estimation.bowlMl,
    cupMl: data.estimation.cupMl || 200,
  }), [data.estimation.bowlMl, data.estimation.cupMl, data.weights]);

  useEffect(() => {
    void setAudioModeAsync({ playsInSilentMode: true, allowsRecording: true })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let active = true;
    async function checkVoiceReadiness() {
      if (Platform.OS === 'ios' && !Device.isDevice) {
        setVoiceStatus({
          ready: false,
          title: 'Voice needs a physical iPhone',
          detail: 'The iOS Simulator cannot record voice commands. Use the keyboard here, then test voice in an iPhone development build.',
        });
        return;
      }
      if (!session) {
        setVoiceStatus({
          ready: false,
          title: 'Sign in to use voice',
          detail: 'Voice transcription is protected by your account. Typed logging remains private and works offline.',
        });
        return;
      }
      try {
        const service = await readServiceHealth();
        if (!active) return;
        setVoiceStatus(service.ai_enabled
          ? {
              ready: true,
              title: 'Voice is ready',
              detail: 'Tap once, say one short update, then tap again.',
            }
          : {
              ready: false,
              title: 'Voice service needs an AI key',
              detail: 'The local API is running, but GOOGLE_API_KEY is not configured. Typed logging still works.',
            });
      } catch (reason) {
        if (!active) return;
        const deviceHint = Device.isDevice
          ? 'Set EXPO_PUBLIC_API_URL to your computer’s LAN address and keep both devices on the same Wi-Fi.'
          : 'Start the local API on port 8000.';
        setVoiceStatus({
          ready: false,
          title: 'Voice service is offline',
          detail: `${reason instanceof Error ? reason.message : 'The API is unreachable.'} ${deviceHint}`,
        });
      }
    }
    void checkVoiceReadiness();
    return () => {
      active = false;
    };
  }, [session]);

  function followUp() {
    if (!parsed?.clarification) return undefined;
    return {
      previousTranscript: parsed.transcript,
      clarificationQuestion: parsed.clarification.question,
    };
  }

  function calibrationFromAnswer(value: string) {
    const question = parsed?.clarification?.question.toLowerCase() ?? '';
    const numeric = Number.parseFloat(value);
    if (!Number.isFinite(numeric)) return {};
    if (question.includes('bowl') && numeric >= 50 && numeric <= 1000) return { bowlMl: numeric };
    if (question.includes('body weight') && numeric >= 20 && numeric <= 400) return { weightKg: numeric };
    return {};
  }

  function onSuggestion(suggestion: string) {
    // "e.g." chips are editable starting points, not literal answers.
    if (suggestion.toLowerCase().startsWith('e.g.')) {
      setText(suggestion.slice(4).trim());
      setShowKeyboard(true);
      return;
    }
    void understandTyped(suggestion);
  }

  function saveProfileUpdates(updates?: ParsedCommand['profileUpdates']) {
    if (!updates) return;
    updateEstimationProfile(
      updates.bowlMl ? { bowlMl: updates.bowlMl } : {},
      updates.weightKg,
    );
  }

  async function understandTyped(value = text.trim()) {
    if (!value) return;
    setBusy(true);
    setError('');
    try {
      const answerUpdate = calibrationFromAnswer(value);
      const effectiveContext = { ...context, ...answerUpdate };
      const result = await parseFitnessCommand(value, params.slot, effectiveContext, followUp());
      saveProfileUpdates({ ...answerUpdate, ...result.profileUpdates });
      setParsed(result);
      setText('');
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
        const result = await parseVoiceCommand(recorder.uri, params.slot, context, followUp());
        saveProfileUpdates(result.profileUpdates);
        setParsed(result);
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Voice logging failed.');
      } finally {
        setBusy(false);
      }
      return;
    }
    if (!voiceStatus.ready) {
      setError(voiceStatus.detail);
      setShowKeyboard(true);
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
    if (!parsed?.operations.length) return;
    applyOperations(parsed.operations);
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    router.dismiss();
  }

  function reset() {
    setParsed(null);
    setText(params.prefill ?? '');
    setError('');
  }

  const profileReady = Boolean(context.weightKg && context.bowlMl);

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          {!parsed ? (
            <>
              <View style={styles.intro}>
                <Text style={styles.eyebrow}>VOICE-FIRST LOGGING</Text>
                <Text style={styles.title}>Say what happened.</Text>
                <Text style={styles.subtitle}>
                  Include the amount. For tighter estimates, say grams, bowl size, workout time and intensity.
                </Text>
              </View>

              <View style={styles.voicePanel}>
                <View style={styles.waveRow}>
                  {[14, 24, 38, 27, 48, 31, 20, 34, 17].map((height, index) => (
                    <View key={`${height}-${index}`} style={[styles.wave, { height }]} />
                  ))}
                </View>
                <Pressable
                  disabled={busy || (!voiceStatus.ready && !recorderState.isRecording)}
                  onPress={toggleRecording}
                  accessibilityLabel={recorderState.isRecording ? 'Stop recording' : 'Start recording'}
                  style={({ pressed }) => [
                    styles.micButton,
                    recorderState.isRecording && styles.micRecording,
                    pressed && styles.pressed,
                  ]}>
                  {busy ? <ActivityIndicator color={palette.forest} /> : <Glyph name="mic" color={palette.forest} size={31} />}
                </Pressable>
                <Text style={styles.voiceTitle}>
                  {busy
                    ? 'Checking quantities…'
                    : recorderState.isRecording
                      ? 'Listening — tap when done'
                      : voiceStatus.title}
                </Text>
                <Text style={styles.voiceMeta}>
                  {recorderState.isRecording
                    ? `${Math.floor(recorderState.durationMillis / 1000)} seconds`
                    : voiceStatus.ready
                      ? 'English, Hindi and Hinglish'
                      : voiceStatus.detail}
                </Text>
              </View>

              <Pressable
                onPress={() => router.push('/settings')}
                style={({ pressed }) => [styles.profileStrip, pressed && styles.pressed]}>
                <View style={styles.profileDot}><Glyph name="chart" color={palette.forest} size={17} /></View>
                <View style={styles.profileCopy}>
                  <Text style={styles.profileTitle}>{profileReady ? 'Estimate profile ready' : 'Finish estimate setup'}</Text>
                  <Text style={styles.profileMeta}>
                    {context.weightKg ? `${context.weightKg} kg` : 'weight needed'} · {context.bowlMl ? `${context.bowlMl} ml bowl` : 'bowl size needed'}
                  </Text>
                </View>
                <Glyph name="chevron" color={palette.muted} size={17} />
              </Pressable>

              <View style={styles.examples}>
                <Text style={styles.examplesLabel}>GOOD VOICE LOGS</Text>
                <Text style={styles.example}>“Lunch: 2 rotis and one 200 ml bowl rajma”</Text>
                <Text style={styles.example}>“100 grams grilled chicken and 150 grams rice”</Text>
                <Text style={styles.example}>“30 minute brisk walk, moderate effort”</Text>
              </View>

              <Pressable onPress={() => setShowKeyboard((current) => !current)} style={styles.keyboardToggle}>
                <Text style={styles.keyboardToggleText}>{showKeyboard ? 'Hide keyboard' : 'Use keyboard instead'}</Text>
              </Pressable>
              {showKeyboard ? (
                <Composer
                  busy={busy}
                  text={text}
                  setText={setText}
                  onSubmit={() => void understandTyped()}
                  placeholder="e.g. 2 rotis and 200 ml rajma"
                />
              ) : null}
            </>
          ) : parsed.clarification ? (
            <View>
              <View style={styles.reviewTop}>
                <View style={styles.questionMark}><Text style={styles.questionMarkText}>?</Text></View>
                <Text style={styles.reviewEyebrow}>ONE QUICK DETAIL</Text>
                <Text style={styles.reviewTitle}>Let’s tighten the estimate.</Text>
              </View>
              <View style={styles.transcript}>
                <Text style={styles.transcriptLabel}>I HEARD</Text>
                <Text style={styles.transcriptText}>“{parsed.transcript}”</Text>
              </View>
              <View style={styles.questionCard}>
                <Text style={styles.question}>{parsed.clarification.question}</Text>
                <View style={styles.suggestions}>
                  {parsed.clarification.suggestions.map((suggestion) => (
                    <Pressable
                      key={suggestion}
                      disabled={busy}
                      onPress={() => onSuggestion(suggestion)}
                      style={({ pressed }) => [styles.suggestion, pressed && styles.pressed]}>
                      <Text style={styles.suggestionText}>{suggestion}</Text>
                    </Pressable>
                  ))}
                </View>
              </View>
              <Pressable
                disabled={busy}
                onPress={toggleRecording}
                style={({ pressed }) => [styles.followupVoice, pressed && styles.pressed]}>
                {busy ? <ActivityIndicator color={palette.lime} /> : <Glyph name="mic" color={palette.lime} size={20} />}
                <Text style={styles.followupVoiceText}>
                  {recorderState.isRecording ? 'Listening — tap when done' : 'Answer with voice'}
                </Text>
              </Pressable>
              <Pressable onPress={() => setShowKeyboard((current) => !current)} style={styles.keyboardToggle}>
                <Text style={styles.keyboardToggleText}>{showKeyboard ? 'Hide keyboard' : 'Type the answer'}</Text>
              </Pressable>
              {showKeyboard ? (
                <Composer
                  busy={busy}
                  text={text}
                  setText={setText}
                  onSubmit={() => void understandTyped()}
                  placeholder="Answer the question above"
                />
              ) : null}
              <Pressable onPress={reset} style={styles.startOver}><Text style={styles.startOverText}>Start over</Text></Pressable>
            </View>
          ) : (
            <View>
              <View style={styles.reviewHeader}>
                <View style={styles.reviewHeaderCopy}>
                  <Text style={styles.reviewEyebrow}>CHECK THE RANGE</Text>
                  <Text style={styles.reviewTitle}>{parsed.confirmation}</Text>
                </View>
                <Pressable onPress={reset}><Text style={styles.editText}>Start over</Text></Pressable>
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
                <Text style={styles.noticeText}>
                  The range reflects portion, recipe or activity variation. Nothing is saved until you confirm.
                </Text>
              </View>
              <Pressable onPress={confirm} style={({ pressed }) => [styles.confirmButton, pressed && styles.pressed]}>
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

function Composer({
  busy,
  text,
  setText,
  onSubmit,
  placeholder,
}: {
  busy: boolean;
  text: string;
  setText: (value: string) => void;
  onSubmit: () => void;
  placeholder: string;
}) {
  return (
    <View style={styles.composer}>
      <TextInput
        autoFocus
        multiline
        value={text}
        onChangeText={setText}
        placeholder={placeholder}
        placeholderTextColor="#8A938B"
        style={styles.input}
      />
      <Pressable
        disabled={busy || !text.trim()}
        onPress={onSubmit}
        style={({ pressed }) => [
          styles.parseButton,
          (!text.trim() || busy) && styles.disabled,
          pressed && styles.pressed,
        ]}>
        {busy ? <ActivityIndicator color={palette.lime} /> : <Text style={styles.parseText}>Review estimate</Text>}
      </Pressable>
    </View>
  );
}

function OperationCard({ operation }: { operation: LogOperation }) {
  if (operation.type === 'meal') {
    const total = operation.items.reduce((sum, item) => sum + item.calories, 0);
    const low = operation.items.reduce((sum, item) => sum + (item.calorieLow ?? item.calories), 0);
    const high = operation.items.reduce((sum, item) => sum + (item.calorieHigh ?? item.calories), 0);
    return (
      <View style={styles.operation}>
        <OperationHeading icon="bowl" title={slotLabels[operation.slot]} range={`${Math.round(low)}–${Math.round(high)} kcal`} />
        {operation.items.map((item) => (
          <View key={`${item.name}-${item.quantity}`} style={styles.itemBlock}>
            <View style={styles.itemLine}>
              <Text style={styles.itemName}>{item.name}</Text>
              <Text style={styles.itemCalories}>{item.calories} kcal</Text>
            </View>
            <Text style={styles.itemBasis}>{item.basis ?? item.quantity}</Text>
            <Text style={styles.itemSource}>{item.sourceLabel ?? 'Estimate'} · {item.sourceId ?? item.source} · {item.confidence ?? 'estimated'} confidence</Text>
          </View>
        ))}
        <Text style={styles.midpoint}>Midpoint used in totals: {Math.round(total)} kcal</Text>
      </View>
    );
  }
  if (operation.type === 'workout') {
    return (
      <View style={styles.operation}>
        <OperationHeading
          icon="dumbbell"
          title={operation.name}
          range={`${operation.calorieLow ?? operation.calories}–${operation.calorieHigh ?? operation.calories} active kcal`}
        />
        <Text style={styles.itemBasis}>{operation.basis ?? `${operation.durationMin} min · ${operation.intensity}`}</Text>
        <Text style={styles.itemSource}>{operation.sourceLabel ?? 'Estimate'} · {operation.confidence ?? 'estimated'} confidence</Text>
        <Text style={styles.midpoint}>Midpoint used in totals: {operation.calories} kcal</Text>
      </View>
    );
  }
  let icon: 'water' | 'steps' | 'sleep' | 'chart' = 'chart';
  let title = '';
  let detail = '';
  if (operation.type === 'water') {
    icon = 'water'; title = 'Water'; detail = `${operation.action === 'add' ? '+' : ''}${operation.amount} ml`;
  } else if (operation.type === 'steps') {
    icon = 'steps'; title = 'Steps'; detail = `${operation.amount.toLocaleString()} steps`;
  } else if (operation.type === 'sleep') {
    icon = 'sleep'; title = 'Sleep'; detail = `${operation.amount} hours`;
  } else {
    title = 'Weight'; detail = `${operation.amount} kg`;
  }
  return (
    <View style={styles.operation}>
      <OperationHeading icon={icon} title={title} range={detail} />
    </View>
  );
}

function OperationHeading({
  icon,
  title,
  range,
}: {
  icon: 'bowl' | 'water' | 'dumbbell' | 'steps' | 'sleep' | 'chart';
  title: string;
  range: string;
}) {
  return (
    <View style={styles.operationHeading}>
      <View style={styles.operationIcon}><Glyph name={icon} color={palette.forest} size={20} /></View>
      <View style={styles.operationHeadingCopy}>
        <Text style={styles.operationTitle}>{title}</Text>
        <Text style={styles.operationRange}>{range}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  fill: { flex: 1 },
  content: { padding: space.md, paddingBottom: 36 },
  intro: { paddingTop: 8, marginBottom: 18 },
  eyebrow: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.3 },
  title: { color: palette.ink, fontFamily: type.demi, fontSize: 28, letterSpacing: -1, marginTop: 5 },
  subtitle: { color: palette.muted, fontFamily: type.regular, fontSize: 12, lineHeight: 18, marginTop: 7, maxWidth: 330 },
  voicePanel: { minHeight: 230, backgroundColor: palette.forest, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', padding: 20, overflow: 'hidden' },
  waveRow: { position: 'absolute', top: 23, flexDirection: 'row', alignItems: 'center', gap: 7, opacity: 0.18 },
  wave: { width: 3, borderRadius: 2, backgroundColor: palette.lime },
  micButton: { width: 76, height: 76, borderRadius: 38, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center' },
  micRecording: { borderWidth: 8, borderColor: '#40503F' },
  voiceTitle: { color: palette.white, fontFamily: type.demi, fontSize: 15, marginTop: 15 },
  voiceMeta: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 10, lineHeight: 15, marginTop: 5, maxWidth: 290, textAlign: 'center' },
  pressed: { opacity: 0.82, transform: [{ scale: 0.99 }] },
  profileStrip: { minHeight: 68, flexDirection: 'row', alignItems: 'center', gap: 11, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 13, marginTop: 12 },
  profileDot: { width: 36, height: 36, borderRadius: 12, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  profileCopy: { flex: 1 },
  profileTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 12 },
  profileMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, marginTop: 3 },
  examples: { marginTop: 20, gap: 8 },
  examplesLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2 },
  example: { color: palette.ink, fontFamily: type.medium, fontSize: 11, lineHeight: 17 },
  keyboardToggle: { alignItems: 'center', paddingVertical: 15 },
  keyboardToggleText: { color: palette.coral, fontFamily: type.demi, fontSize: 11 },
  composer: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 12 },
  input: { minHeight: 86, color: palette.ink, fontFamily: type.medium, fontSize: 14, lineHeight: 21, textAlignVertical: 'top', padding: 5 },
  parseButton: { height: 46, borderRadius: radius.sm, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  parseText: { color: palette.lime, fontFamily: type.demi, fontSize: 13 },
  disabled: { opacity: 0.45 },
  reviewTop: { alignItems: 'center', paddingVertical: 8, marginBottom: 15 },
  questionMark: { width: 44, height: 44, borderRadius: 15, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center', marginBottom: 9 },
  questionMarkText: { color: palette.forest, fontFamily: type.demi, fontSize: 23 },
  reviewHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 13 },
  reviewHeaderCopy: { flex: 1, paddingRight: 12 },
  reviewEyebrow: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2 },
  reviewTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 20, lineHeight: 25, marginTop: 4 },
  editText: { color: palette.coral, fontFamily: type.demi, fontSize: 11 },
  transcript: { backgroundColor: palette.softLime, borderRadius: radius.md, padding: 15, marginBottom: 12 },
  transcriptLabel: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.1 },
  transcriptText: { color: palette.ink, fontFamily: type.medium, fontSize: 13, lineHeight: 20, marginTop: 5 },
  questionCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 17 },
  question: { color: palette.ink, fontFamily: type.demi, fontSize: 17, lineHeight: 24 },
  suggestions: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 15 },
  suggestion: { borderWidth: 1, borderColor: palette.line, backgroundColor: palette.canvas, borderRadius: radius.sm, paddingHorizontal: 12, paddingVertical: 10 },
  suggestionText: { color: palette.ink, fontFamily: type.medium, fontSize: 11 },
  followupVoice: { height: 52, borderRadius: radius.md, backgroundColor: palette.forest, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 9, marginTop: 12 },
  followupVoiceText: { color: palette.lime, fontFamily: type.demi, fontSize: 13 },
  startOver: { alignItems: 'center', padding: 8 },
  startOverText: { color: palette.muted, fontFamily: type.medium, fontSize: 10 },
  operations: { gap: 10 },
  operation: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14 },
  operationHeading: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  operationIcon: { width: 40, height: 40, borderRadius: 13, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  operationHeadingCopy: { flex: 1 },
  operationTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 14 },
  operationRange: { color: palette.limeDark, fontFamily: type.demi, fontSize: 12, marginTop: 3 },
  itemBlock: { borderTopWidth: 1, borderTopColor: palette.line, marginTop: 12, paddingTop: 11 },
  itemLine: { flexDirection: 'row', justifyContent: 'space-between', gap: 10 },
  itemName: { flex: 1, color: palette.ink, fontFamily: type.demi, fontSize: 12 },
  itemCalories: { color: palette.ink, fontFamily: type.demi, fontSize: 11 },
  itemBasis: { color: palette.muted, fontFamily: type.regular, fontSize: 10, lineHeight: 15, marginTop: 5 },
  itemSource: { color: palette.limeDark, fontFamily: type.medium, fontSize: 9, lineHeight: 14, marginTop: 4 },
  midpoint: { color: palette.muted, fontFamily: type.medium, fontSize: 9, marginTop: 11 },
  notice: { flexDirection: 'row', alignItems: 'flex-start', gap: 9, backgroundColor: palette.softLime, borderRadius: radius.md, padding: 13, marginTop: 12 },
  noticeText: { flex: 1, color: palette.limeDark, fontFamily: type.medium, fontSize: 10, lineHeight: 15 },
  confirmButton: { height: 54, borderRadius: radius.md, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center', marginTop: 12 },
  confirmText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
  error: { color: '#B64B45', backgroundColor: '#FBE9E6', fontFamily: type.medium, fontSize: 11, lineHeight: 16, padding: 12, borderRadius: radius.sm, marginTop: 12 },
});
