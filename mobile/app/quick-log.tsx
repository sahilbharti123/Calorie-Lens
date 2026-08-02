import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import { type Href, useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import Animated, {
  Easing,
  cancelAnimation,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import { useReducedMotion } from '@/src/lib/accessibility';
import {
  Bar,
  Card,
  Chip,
  CountUp,
  GhostButton,
  GlassFooter,
  ListRow,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  ScreenHeader,
  Tap,
  Well,
} from '@/src/components/ui';
import { parseFitnessCommand } from '@/src/lib/nutrition';
import { ensureSpeechPermission, speechAvailable, useDictation } from '@/src/lib/speech';
import { slotLabels } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import {
  alpha,
  gradient,
  motion,
  palette,
  radius,
  shadow,
  space,
  tabular,
  text,
} from '@/src/theme';
import type {
  EstimateConfidence,
  EstimationContext,
  LogOperation,
  MealSlot,
  ParsedCommand,
} from '@/src/types';

/** Bar heights of the level meter behind the record button. */
const WAVE = [14, 24, 38, 27, 48, 31, 20, 34, 17];

const EXAMPLES = [
  '“Lunch: 2 rotis and one 200 ml bowl rajma”',
  '“100 grams grilled chicken and 150 grams rice”',
  '“30 minute brisk walk, moderate effort”',
];

type Stage = 'capture' | 'clarify' | 'review';
type RecordState = 'idle' | 'listening' | 'thinking' | 'done';
type VoiceStatus = { ready: boolean; title: string; detail: string };

/**
 * Dictation is done by the operating system, so readiness is only ever about
 * this device: no account, no API key, no network. The wording below matches
 * what `useDictation` reports, so the card and the error banner agree.
 */
const VOICE_READY: VoiceStatus = {
  ready: true,
  title: 'Voice is ready',
  detail: 'Tap once, say one short update, then tap again.',
};

const VOICE_UNSUPPORTED: VoiceStatus = {
  ready: false,
  title: 'Voice needs on-device speech',
  detail: 'This device cannot transcribe speech. Use the keyboard to log.',
};

const HEADER: Record<Stage, { eyebrow: string; title: string }> = {
  capture: { eyebrow: 'Voice-first logging', title: 'Say what happened.' },
  clarify: { eyebrow: 'One quick detail', title: 'Let’s tighten the estimate.' },
  review: { eyebrow: 'Check the range', title: 'Review estimate' },
};

export default function QuickLogScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ slot?: MealSlot; prefill?: string }>();
  const { applyOperations, data, updateEstimationProfile } = useApp();
  const [draft, setText] = useState(params.prefill ?? '');
  const [showKeyboard, setShowKeyboard] = useState(Boolean(params.prefill));
  const [parsed, setParsed] = useState<ParsedCommand | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>({
    ready: false,
    title: 'Checking voice setup…',
    detail: 'Typed logging is always available.',
  });
  const dictation = useDictation({ onFinal: (spoken) => void understandSpoken(spoken) });
  const context = useMemo<EstimationContext>(() => ({
    weightKg: data.weights.at(-1)?.kg,
    bowlMl: data.estimation.bowlMl,
    cupMl: data.estimation.cupMl || 200,
  }), [data.estimation.bowlMl, data.estimation.cupMl, data.weights]);

  useEffect(() => {
    setVoiceStatus(speechAvailable() ? VOICE_READY : VOICE_UNSUPPORTED);
  }, []);

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
    // Category chips are navigation choices, not complete log entries. Sending
    // the word "Weight" back through the parser used to produce the same
    // question again, which made the chip feel dead.
    if (parsed?.clarification?.question === 'What would you like me to log?') {
      if (suggestion === 'Weight') {
        router.replace('/weight-log' as Href);
        return;
      }
      const starters: Record<string, string> = {
        'A meal': 'I ate ',
        Water: 'I drank ',
        'A workout': 'I did ',
      };
      setParsed(null);
      setText(starters[suggestion] ?? '');
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

  async function understandTyped(value = draft.trim()) {
    if (!value) return;
    setBusy(true);
    setError('');
    // A leftover dictation error should not sit above a typed answer.
    dictation.setError(null);
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

  /** The dictated text goes through the same local parser the keyboard uses. */
  async function understandSpoken(spoken: string) {
    const value = spoken.trim();
    if (!value) return;
    setBusy(true);
    setError('');
    try {
      const result = await parseFitnessCommand(value, params.slot, context, followUp());
      saveProfileUpdates(result.profileUpdates);
      setParsed(result);
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Voice logging failed.');
    } finally {
      setBusy(false);
    }
  }

  async function toggleDictation() {
    setError('');
    if (dictation.state === 'listening') {
      dictation.stop();
      return;
    }
    if (dictation.state === 'finishing' || busy) return;
    if (!speechAvailable()) {
      setVoiceStatus(VOICE_UNSUPPORTED);
      setError(VOICE_UNSUPPORTED.detail);
      setShowKeyboard(true);
      return;
    }
    const permission = await ensureSpeechPermission();
    if (!permission.granted) {
      const blocked: VoiceStatus = {
        ready: false,
        title: 'Voice needs microphone access',
        detail: permission.canAskAgain
          ? 'Vigorly needs microphone and speech access to log by voice.'
          : 'Microphone or speech access is off. Turn it on in Settings to log by voice.',
      };
      setVoiceStatus(blocked);
      setError(blocked.detail);
      return;
    }
    setVoiceStatus(VOICE_READY);
    const started = await dictation.start();
    if (!started) return;
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  }

  function confirm() {
    if (!parsed?.operations.length) return;
    applyOperations(parsed.operations);
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    router.dismiss();
  }

  function reset() {
    dictation.reset();
    setParsed(null);
    setText(params.prefill ?? '');
    setError('');
  }

  const profileReady = Boolean(context.weightKg && context.bowlMl);
  const listening = dictation.state === 'listening';
  const capturing = listening || dictation.state === 'finishing';
  const thinking = busy || dictation.state === 'finishing';
  const micState: RecordState = thinking ? 'thinking' : listening ? 'listening' : 'idle';
  const stage: Stage = !parsed ? 'capture' : parsed.clarification ? 'clarify' : 'review';
  // On-device dictation streams words as they are recognised, so the meta line
  // can show the sentence forming instead of a stopwatch.
  const heard = dictation.transcript.trim();
  const liveLine = heard ? `“${heard}”` : 'Words appear here as you speak.';
  const shownError = error || dictation.error || '';

  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.fill}>
        <ScreenHeader
          action={
            <Tap
              accessibilityLabel="Close quick log"
              hitSlop={10}
              onPress={() => router.dismiss()}
              scaleTo={0.9}
              style={styles.close}>
              <Glyph color={palette.ink} name="close" size={18} />
            </Tap>
          }
          eyebrow={HEADER[stage].eyebrow}
          key={stage}
          style={styles.header}
          title={HEADER[stage].title}
        />

        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          style={styles.fill}>
          {stage === 'capture' ? (
            <View key="capture">
              <Reveal index={1}>
                <Text style={styles.lede}>
                  Include the amount. For tighter estimates, say grams, bowl size, workout time and intensity.
                </Text>
              </Reveal>

              {/* ---------- The microphone is the hero ---------- */}
              <Reveal index={2} style={styles.blockGap}>
                <Card glow raised>
                  <View style={styles.voiceInner}>
                    <View style={styles.wave}>
                      {WAVE.map((height, index) => (
                        <WaveBar
                          active={listening}
                          height={height}
                          index={index}
                          key={`${height}-${index}`}
                          level={dictation.level}
                        />
                      ))}
                    </View>
                    <RecordControl
                      accessibilityLabel={listening ? 'Stop recording' : 'Start recording'}
                      disabled={thinking || (!voiceStatus.ready && !listening)}
                      onPress={() => void toggleDictation()}
                      size={112}
                      state={micState}
                    />
                    <Text style={styles.voiceTitle}>
                      {thinking
                        ? 'Checking quantities…'
                        : listening
                          ? 'Listening — tap when done'
                          : voiceStatus.title}
                    </Text>
                    <Text style={styles.voiceMeta}>
                      {capturing
                        ? liveLine
                        : voiceStatus.ready
                          ? 'English, Hindi and Hinglish'
                          : voiceStatus.detail}
                    </Text>
                  </View>
                </Card>
              </Reveal>

              {/* ---------- Keyboard is an equal path, not a fallback ---------- */}
              <Reveal index={3} style={styles.blockGap}>
                <GhostButton
                  icon={showKeyboard ? 'chevronDown' : 'keyboard'}
                  label={showKeyboard ? 'Hide keyboard' : 'Use keyboard instead'}
                  onPress={() => setShowKeyboard((current) => !current)}
                />
              </Reveal>
              {showKeyboard ? (
                <Reveal index={4} style={styles.groupGap}>
                  <Composer
                    busy={busy}
                    onSubmit={() => void understandTyped()}
                    placeholder="e.g. 2 rotis and 200 ml rajma"
                    setText={setText}
                    text={draft}
                  />
                </Reveal>
              ) : null}

              <Reveal index={5} style={styles.blockGap}>
                <Card padded={false} style={styles.rowCard}>
                  <ListRow
                    accent={profileReady ? palette.lime : palette.inkLow}
                    detail={`${context.weightKg ? `${context.weightKg} kg` : 'weight needed'} · ${context.bowlMl ? `${context.bowlMl} ml bowl` : 'bowl size needed'}`}
                    icon="chart"
                    last
                    onPress={() => router.push('/settings')}
                    title={profileReady ? 'Estimate profile ready' : 'Finish estimate setup'}
                  />
                </Card>
              </Reveal>

              <Reveal index={6} style={styles.blockGap}>
                <Card>
                  <Text style={styles.tinyLabel}>GOOD VOICE LOGS</Text>
                  <View style={styles.exampleList}>
                    {EXAMPLES.map((example) => (
                      <View key={example} style={styles.exampleRow}>
                        <Glyph color={palette.inkLow} name="mic" size={13} />
                        <Text style={styles.exampleText}>{example}</Text>
                      </View>
                    ))}
                  </View>
                </Card>
              </Reveal>
            </View>
          ) : null}

          {stage === 'clarify' && parsed?.clarification ? (
            <View key="clarify">
              {/* ---------- A conversation, not a form error ---------- */}
              <Reveal index={1}>
                <View style={styles.saidHead}>
                  <Text style={styles.tinyLabel}>I HEARD</Text>
                </View>
                <Card style={styles.userBubble}>
                  <Text style={styles.bubbleText}>“{parsed.transcript}”</Text>
                </Card>
              </Reveal>

              <Reveal index={2} style={styles.blockGap}>
                <View style={styles.coachRow}>
                  <View style={styles.coachAvatar}>
                    <Glyph color={palette.lime} name="spark" size={16} />
                  </View>
                  <View style={styles.coachSide}>
                    <Card style={styles.coachBubble}>
                      <Text style={styles.question}>{parsed.clarification.question}</Text>
                    </Card>
                    {parsed.clarification.suggestions.length ? (
                      <View
                        pointerEvents={busy ? 'none' : 'auto'}
                        style={[styles.chips, busy && styles.dim]}>
                        {parsed.clarification.suggestions.map((suggestion) => (
                          <Chip
                            key={suggestion}
                            label={suggestion}
                            onPress={() => {
                              if (busy) return;
                              onSuggestion(suggestion);
                            }}
                          />
                        ))}
                      </View>
                    ) : null}
                  </View>
                </View>
              </Reveal>

              <Reveal index={3} style={styles.blockGap}>
                <Card style={styles.answerCard}>
                  <RecordControl
                    accessibilityLabel={listening ? 'Stop recording' : 'Answer with voice'}
                    disabled={thinking}
                    onPress={() => void toggleDictation()}
                    size={58}
                    state={micState}
                  />
                  <View style={styles.answerCopy}>
                    <Text style={styles.answerTitle}>
                      {listening ? 'Listening — tap when done' : 'Answer with voice'}
                    </Text>
                    <Text style={styles.answerMeta}>
                      {capturing
                        ? liveLine
                        : thinking
                          ? 'Checking quantities…'
                          : 'One short sentence is enough.'}
                    </Text>
                  </View>
                </Card>
              </Reveal>

              <Reveal index={4} style={styles.groupGap}>
                <GhostButton
                  icon={showKeyboard ? 'chevronDown' : 'keyboard'}
                  label={showKeyboard ? 'Hide keyboard' : 'Type the answer'}
                  onPress={() => setShowKeyboard((current) => !current)}
                />
              </Reveal>
              {showKeyboard ? (
                <Reveal index={5} style={styles.groupGap}>
                  <Composer
                    busy={busy}
                    onSubmit={() => void understandTyped()}
                    placeholder="Answer the question above"
                    setText={setText}
                    text={draft}
                  />
                </Reveal>
              ) : null}

              <Reveal index={6} style={styles.blockGap}>
                <Tap accessibilityLabel="Start over" onPress={reset} scaleTo={0.95}>
                  <Text style={styles.link}>Start over</Text>
                </Tap>
              </Reveal>
            </View>
          ) : null}

          {stage === 'review' && parsed ? (
            <View key="review">
              <Reveal index={1}>
                <Text style={styles.confirmation}>{parsed.confirmation}</Text>
              </Reveal>

              <Reveal index={2} style={styles.groupGap}>
                <Well style={styles.heardRow}>
                  <RecordControl size={42} state="done" />
                  <View style={styles.coachSide}>
                    <Text style={styles.tinyLabel}>YOU SAID</Text>
                    <Text style={styles.heardText}>“{parsed.transcript}”</Text>
                  </View>
                </Well>
              </Reveal>

              {parsed.operations.map((operation, index) => (
                <Reveal
                  index={3 + index}
                  key={`${operation.type}-${index}`}
                  style={index === 0 ? styles.blockGap : styles.groupGap}>
                  <OperationCard hero={index === 0} operation={operation} />
                </Reveal>
              ))}

              <Reveal index={3 + parsed.operations.length} style={styles.blockGap}>
                <Well style={styles.notice}>
                  <Glyph color={palette.lime} name="info" size={15} />
                  <Text style={styles.noticeText}>
                    The range reflects portion, recipe or activity variation. Nothing is saved until you confirm.
                  </Text>
                </Well>
              </Reveal>
            </View>
          ) : null}

          {shownError ? (
            <Reveal style={styles.blockGap}>
              <View style={styles.error}>
                <Glyph color={palette.danger} name="alert" size={16} />
                <Text style={styles.errorText}>{shownError}</Text>
              </View>
            </Reveal>
          ) : null}
        </ScrollView>

        {stage === 'review' ? (
          <GlassFooter>
            <PrimaryButton icon="check" label="Confirm and log" onPress={confirm} />
            <Tap accessibilityLabel="Start over" onPress={reset} scaleTo={0.95}>
              <Text style={styles.link}>Start over</Text>
            </Tap>
          </GlassFooter>
        ) : null}
      </KeyboardAvoidingView>
    </Screen>
  );
}

/* ------------------------------------------------------------------ *
 * The record control — idle / listening / thinking / done
 * ------------------------------------------------------------------ */

function RecordControl({
  state,
  size = 112,
  onPress,
  disabled,
  accessibilityLabel,
}: {
  state: RecordState;
  size?: number;
  onPress?: () => void;
  disabled?: boolean;
  accessibilityLabel?: string;
}) {
  const reducedMotion = useReducedMotion();
  const live = useSharedValue(0);
  const rippleA = useSharedValue(0);
  const rippleB = useSharedValue(0);
  const think = useSharedValue(0);

  const listening = state === 'listening';
  const thinking = state === 'thinking';

  useEffect(() => {
    live.value = reducedMotion
      ? (listening ? 1 : 0)
      : withTiming(listening ? 1 : 0, { duration: motion.quick });
    if (!listening) {
      cancelAnimation(rippleA);
      cancelAnimation(rippleB);
      rippleA.value = withTiming(0, { duration: motion.quick });
      rippleB.value = withTiming(0, { duration: motion.quick });
      return;
    }
    if (reducedMotion) {
      rippleA.value = 0;
      rippleB.value = 0;
      return;
    }
    const sweep = () =>
      withRepeat(withTiming(1, { duration: 1700, easing: Easing.out(Easing.quad) }), -1, false);
    rippleA.value = 0;
    rippleA.value = sweep();
    rippleB.value = 0;
    rippleB.value = withDelay(850, sweep());
  }, [listening, live, rippleA, rippleB, reducedMotion]);

  useEffect(() => {
    if (reducedMotion) {
      cancelAnimation(think);
      think.value = 0;
      return;
    }
    if (thinking) {
      think.value = withRepeat(
        withTiming(1, { duration: 760, easing: Easing.inOut(Easing.quad) }),
        -1,
        true,
      );
      return;
    }
    cancelAnimation(think);
    think.value = withTiming(0, { duration: motion.quick });
  }, [thinking, think, reducedMotion]);

  const haloA = useAnimatedStyle(() => ({
    opacity: live.value * (1 - rippleA.value) * 0.85,
    transform: [{ scale: 0.96 + rippleA.value * 0.58 }],
  }));
  const haloB = useAnimatedStyle(() => ({
    opacity: live.value * (1 - rippleB.value) * 0.7,
    transform: [{ scale: 0.96 + rippleB.value * 0.58 }],
  }));
  const button = useAnimatedStyle(() => ({
    transform: [{ scale: 1 + live.value * 0.05 - think.value * 0.055 }],
  }));

  const muted = Boolean(disabled) && !thinking;
  // Leaves room for the ripple to breathe without clipping against the card.
  const box = onPress ? Math.round(size * 1.62) : size;
  const inset = (box - size) / 2;
  const disc = { top: inset, left: inset, width: size, height: size, borderRadius: size / 2 };

  const circle = (
    <Animated.View style={[muted ? undefined : shadow.glow, button]}>
      <LinearGradient
        colors={muted ? [palette.surfaceHi, palette.surface] : [...gradient.lime]}
        end={{ x: 1, y: 1 }}
        start={{ x: 0, y: 0 }}
        style={[styles.recordCircle, { width: size, height: size, borderRadius: size / 2 }]}>
        {thinking ? (
          <ActivityIndicator color={palette.onLime} />
        ) : (
          <Glyph
            color={muted ? palette.inkLow : palette.onLime}
            name={state === 'done' ? 'check' : 'mic'}
            size={Math.round(size * 0.31)}
          />
        )}
      </LinearGradient>
    </Animated.View>
  );

  return (
    <View style={[styles.recordBox, { width: box, height: box }]}>
      <Animated.View pointerEvents="none" style={[styles.halo, disc, haloA]} />
      <Animated.View pointerEvents="none" style={[styles.halo, styles.haloSoft, disc, haloB]} />
      {onPress ? (
        <Tap
          accessibilityLabel={accessibilityLabel}
          disabled={disabled}
          haptic="none"
          onPress={onPress}
          scaleTo={0.93}>
          {circle}
        </Tap>
      ) : (
        circle
      )}
    </View>
  );
}

/** One bar of the level meter. Follows the live microphone level, 0–1. */
function WaveBar({
  active,
  height,
  index,
  level,
}: {
  active: boolean;
  height: number;
  index: number;
  level: number;
}) {
  const reducedMotion = useReducedMotion();
  const amount = useSharedValue(0);

  useEffect(() => {
    if (!active) {
      cancelAnimation(amount);
      amount.value = reducedMotion ? 0 : withTiming(0, { duration: motion.base });
      return;
    }
    // Bars nearer the middle react hardest, so the row reads as one shape
    // rather than nine independent meters.
    const middle = (WAVE.length - 1) / 2;
    const weight = 0.5 + 0.5 * (1 - Math.abs(index - middle) / middle);
    const next = Math.min(1, level * weight * 1.7);
    amount.value = reducedMotion ? next : withTiming(next, {
      duration: 130,
      easing: Easing.out(Easing.quad),
    });
  }, [active, amount, index, level, reducedMotion]);

  const animated = useAnimatedStyle(() => ({
    transform: [{ scaleY: 0.3 + amount.value * 0.7 }],
    opacity: 0.18 + amount.value * 0.6,
  }));

  return <Animated.View style={[styles.waveBar, { height }, animated]} />;
}

/* ------------------------------------------------------------------ *
 * Keyboard mode
 * ------------------------------------------------------------------ */

function Composer({
  busy,
  text: value,
  setText,
  onSubmit,
  placeholder,
}: {
  busy: boolean;
  text: string;
  setText: (next: string) => void;
  onSubmit: () => void;
  placeholder: string;
}) {
  return (
    <Card>
      <Well style={styles.inputWell}>
        <TextInput
          autoFocus
          multiline
          onChangeText={setText}
          placeholder={placeholder}
          placeholderTextColor={palette.inkLow}
          style={styles.input}
          value={value}
        />
      </Well>
      <PrimaryButton
        compact
        disabled={!value.trim()}
        icon="spark"
        label="Review estimate"
        loading={busy}
        onPress={onSubmit}
        style={styles.parseButton}
      />
    </Card>
  );
}

/* ------------------------------------------------------------------ *
 * Estimate review
 * ------------------------------------------------------------------ */

function OperationCard({ operation, hero }: { operation: LogOperation; hero?: boolean }) {
  if (operation.type === 'meal') {
    const total = operation.items.reduce((sum, item) => sum + item.calories, 0);
    const low = operation.items.reduce((sum, item) => sum + (item.calorieLow ?? item.calories), 0);
    const high = operation.items.reduce((sum, item) => sum + (item.calorieHigh ?? item.calories), 0);
    return (
      <Card glow={hero} raised={hero}>
        <OperationHeading icon="bowl" label="Meal" title={slotLabels[operation.slot]} />
        <Midpoint hero={hero} unit="kcal" value={Math.round(total)} />
        <RangeStrip high={Math.round(high)} low={Math.round(low)} mid={total} unit="kcal" />
        {operation.items.map((item) => (
          <View key={`${item.name}-${item.quantity}`} style={styles.item}>
            <View style={styles.itemLine}>
              <Text style={styles.itemName}>{item.name}</Text>
              <Text style={styles.itemCalories}>{item.calories} kcal</Text>
            </View>
            <Text style={styles.itemBasis}>{item.basis ?? item.quantity}</Text>
            <View style={styles.itemMeta}>
              <Pill
                icon={confidenceIcon(item.confidence)}
                label={`${item.confidence ?? 'estimated'} confidence`}
                tone={confidenceTone(item.confidence)}
              />
              <Text numberOfLines={2} style={styles.itemSource}>
                {item.sourceLabel ?? 'Estimate'} · {item.sourceId ?? item.source}
              </Text>
            </View>
          </View>
        ))}
      </Card>
    );
  }

  if (operation.type === 'workout') {
    return (
      <Card glow={hero} raised={hero}>
        <OperationHeading icon="dumbbell" label="Workout" title={operation.name} />
        <Midpoint hero={hero} unit="kcal" value={operation.calories} />
        <RangeStrip
          high={operation.calorieHigh ?? operation.calories}
          low={operation.calorieLow ?? operation.calories}
          mid={operation.calories}
          unit="active kcal"
        />
        <View style={styles.item}>
          <Text style={styles.itemBasis}>
            {operation.basis ?? `${operation.durationMin} min · ${operation.intensity}`}
          </Text>
          <View style={styles.itemMeta}>
            <Pill
              icon={confidenceIcon(operation.confidence)}
              label={`${operation.confidence ?? 'estimated'} confidence`}
              tone={confidenceTone(operation.confidence)}
            />
            <Text numberOfLines={2} style={styles.itemSource}>
              {operation.sourceLabel ?? 'Estimate'}
            </Text>
          </View>
        </View>
      </Card>
    );
  }

  let icon: GlyphName = 'chart';
  let title = '';
  let prefix = '';
  let suffix = '';
  let amount = 0;
  if (operation.type === 'water') {
    icon = 'water';
    title = 'Water';
    prefix = operation.action === 'add' ? '+' : '';
    suffix = ' ml';
    amount = operation.amount;
  } else if (operation.type === 'steps') {
    icon = 'steps';
    title = 'Steps';
    suffix = ' steps';
    amount = operation.amount;
  } else if (operation.type === 'sleep') {
    icon = 'sleep';
    title = 'Sleep';
    suffix = ' hours';
    amount = operation.amount;
  } else {
    icon = 'scale';
    title = 'Weight';
    suffix = ' kg';
    amount = operation.amount;
  }
  return (
    <Card glow={hero} raised={hero}>
      <OperationHeading icon={icon} title={title} />
      <View
        accessible
        accessibilityLabel={`${title}: ${prefix}${amount}${suffix}`}
        style={[styles.midBlock, styles.midRow]}>
        <CountUp
          decimals={decimalsOf(amount)}
          prefix={prefix}
          style={[styles.midValue, !hero && styles.midValueQuiet]}
          suffix={suffix}
          value={amount}
        />
      </View>
    </Card>
  );
}

function OperationHeading({
  icon,
  label,
  title,
}: {
  icon: GlyphName;
  label?: string;
  title: string;
}) {
  return (
    <View style={styles.opHead}>
      <View style={styles.opIcon}>
        <Glyph color={palette.lime} name={icon} size={17} />
      </View>
      <View style={styles.coachSide}>
        {label ? <Text style={styles.tinyLabel}>{label.toUpperCase()}</Text> : null}
        <Text style={[styles.opTitle, !label && styles.opTitleAlone]}>{title}</Text>
      </View>
    </View>
  );
}

/** The number the totals actually use, stated big. */
function Midpoint({ unit, value, hero }: { unit: string; value: number; hero?: boolean }) {
  return (
    <View
      accessible
      accessibilityLabel={`Midpoint used in totals: ${value} ${unit}`}
      style={styles.midBlock}>
      <View style={styles.midRow}>
        <CountUp
          decimals={decimalsOf(value)}
          style={[styles.midValue, !hero && styles.midValueQuiet]}
          value={value}
        />
        <Text style={styles.midUnit}>{unit}</Text>
      </View>
      <Text style={styles.midCaption}>Midpoint used in totals</Text>
    </View>
  );
}

/** The honest part: the spread the estimate sits inside. Never hidden. */
function RangeStrip({
  low,
  high,
  mid,
  unit,
}: {
  low: number;
  high: number;
  mid: number;
  unit: string;
}) {
  const span = high - low;
  const position = span > 0 ? (mid - low) / span : 1;
  return (
    <View style={styles.range}>
      <View style={styles.rangeHead}>
        <Text style={styles.tinyLabel}>ESTIMATE RANGE</Text>
        <Text style={styles.rangeValue}>{`${low}–${high} ${unit}`}</Text>
      </View>
      <Bar delay={220} height={4} value={position} />
    </View>
  );
}

function confidenceTone(confidence?: EstimateConfidence): 'default' | 'accent' {
  return confidence === 'high' ? 'accent' : 'default';
}

function confidenceIcon(confidence?: EstimateConfidence): GlyphName {
  if (confidence === 'high') return 'check';
  if (confidence === 'low') return 'info';
  return 'target';
}

/** Keeps a value rendering exactly as many decimals as it actually has. */
function decimalsOf(value: number) {
  if (!Number.isFinite(value)) return 0;
  const fraction = String(value).split('.')[1];
  return fraction ? Math.min(fraction.length, 2) : 0;
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  header: { paddingHorizontal: space.md },
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  blockGap: { marginTop: space.md },
  groupGap: { marginTop: space.sm },

  close: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: palette.surfaceHi,
    borderWidth: 1,
    borderColor: palette.lineHi,
    alignItems: 'center',
    justifyContent: 'center',
  },

  lede: { ...text.body, color: palette.inkMid, maxWidth: 340 },
  tinyLabel: { ...text.label, color: palette.inkLow },
  link: { ...text.value, color: palette.inkMid, textAlign: 'center', paddingVertical: 6 },
  dim: { opacity: 0.5 },

  /* ---------- capture ---------- */
  voiceInner: { alignItems: 'center', paddingVertical: 4 },
  wave: { flexDirection: 'row', alignItems: 'center', gap: 7, height: 50 },
  waveBar: { width: 3, borderRadius: 2, backgroundColor: palette.lime },
  recordBox: { alignItems: 'center', justifyContent: 'center' },
  recordCircle: { alignItems: 'center', justifyContent: 'center' },
  halo: { position: 'absolute', backgroundColor: alpha.limeGlow },
  haloSoft: { backgroundColor: alpha.limeGlowSoft },
  voiceTitle: { ...text.section, color: palette.ink, marginTop: 4, textAlign: 'center' },
  voiceMeta: {
    ...text.caption,
    ...tabular,
    color: palette.inkMid,
    marginTop: 6,
    maxWidth: 300,
    textAlign: 'center',
  },

  rowCard: { paddingHorizontal: 14 },
  exampleList: { gap: 10, marginTop: 12 },
  exampleRow: { flexDirection: 'row', alignItems: 'center', gap: 9 },
  exampleText: { ...text.caption, color: palette.inkMid, flex: 1 },

  inputWell: { paddingHorizontal: 14, paddingVertical: 12 },
  input: {
    ...text.body,
    color: palette.ink,
    minHeight: 76,
    padding: 0,
    textAlignVertical: 'top',
  },
  parseButton: { marginTop: 12 },

  /* ---------- clarify ---------- */
  saidHead: { alignItems: 'flex-end', marginBottom: 7 },
  userBubble: {
    alignSelf: 'flex-end',
    maxWidth: '92%',
    backgroundColor: palette.limeSoft,
    borderColor: `${palette.lime}2E`,
    borderTopRightRadius: radius.xs,
  },
  bubbleText: { ...text.body, color: palette.ink },
  coachRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  coachAvatar: {
    width: 34,
    height: 34,
    borderRadius: 12,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  coachSide: { flex: 1 },
  coachBubble: { borderTopLeftRadius: radius.xs },
  question: { ...text.section, color: palette.ink },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 10 },
  answerCard: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  answerCopy: { flex: 1 },
  answerTitle: { ...text.row, color: palette.ink },
  answerMeta: { ...text.caption, ...tabular, color: palette.inkLow, marginTop: 3 },

  /* ---------- review ---------- */
  confirmation: { ...text.headline, color: palette.ink },
  heardRow: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 14 },
  heardText: { ...text.body, color: palette.ink, marginTop: 4 },

  opHead: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  opIcon: {
    width: 36,
    height: 36,
    borderRadius: 13,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  opTitle: { ...text.row, color: palette.ink, marginTop: 3 },
  opTitleAlone: { ...text.section, marginTop: 0 },

  midBlock: { marginTop: 16 },
  midRow: { flexDirection: 'row', alignItems: 'baseline', gap: 7 },
  midValue: { ...text.hero, ...tabular, color: palette.ink },
  midValueQuiet: { ...text.headline },
  midUnit: { ...text.caption, color: palette.inkMid },
  midCaption: { ...text.caption, color: palette.inkLow, marginTop: 2 },

  range: { marginTop: 16, gap: 7 },
  rangeHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 10 },
  rangeValue: { ...text.value, ...tabular, color: palette.lime },

  item: { borderTopWidth: 1, borderTopColor: palette.line, marginTop: 16, paddingTop: 13 },
  itemLine: { flexDirection: 'row', justifyContent: 'space-between', gap: 10 },
  itemName: { ...text.row, color: palette.ink, flex: 1 },
  itemCalories: { ...text.value, ...tabular, color: palette.ink },
  itemBasis: { ...text.caption, color: palette.inkMid, marginTop: 5 },
  itemMeta: { flexDirection: 'row', alignItems: 'center', gap: 9, marginTop: 9 },
  itemSource: { ...text.micro, ...tabular, color: palette.inkLow, flex: 1 },

  notice: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, padding: 14 },
  noticeText: { ...text.caption, color: palette.inkMid, flex: 1 },

  /* ---------- errors ---------- */
  error: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    padding: 14,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: `${palette.danger}33`,
    backgroundColor: `${palette.danger}14`,
  },
  errorText: { ...text.caption, color: palette.danger, flex: 1 },
});
