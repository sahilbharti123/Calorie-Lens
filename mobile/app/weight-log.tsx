import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { Glyph } from '@/src/components/glyph';
import {
  Card,
  GlassFooter,
  PrimaryButton,
  Reveal,
  Screen,
  ScreenHeader,
  Tap,
  Well,
} from '@/src/components/ui';
import { useDictation } from '@/src/lib/speech';
import { parseWeightInput, weightToTargetCopy } from '@/src/lib/weight';
import { useApp } from '@/src/store/app-store';
import { palette, radius, shadow, space, tabular, text } from '@/src/theme';

export default function WeightLogScreen() {
  const router = useRouter();
  const { applyOperations, data } = useApp();
  const currentWeight = data.weights.at(-1)?.kg;
  const targetWeight = data.profile.targetWeightKg;
  const [value, setValue] = useState(currentWeight ? currentWeight.toFixed(1) : '');
  const [heard, setHeard] = useState('');
  const [error, setError] = useState('');

  const parsed = useMemo(() => parseWeightInput(value), [value]);
  const dictation = useDictation({
    onFinal: (transcript) => {
      setHeard(transcript);
      const result = parseWeightInput(transcript);
      if (!result) {
        setError('I heard the words, but not a plausible scale reading. Type the number or try again.');
        return;
      }
      setValue(result.kg.toFixed(1));
      setError('');
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    },
  });

  async function toggleVoice() {
    setError('');
    if (dictation.state === 'listening') {
      dictation.stop();
      return;
    }
    if (dictation.state === 'finishing') return;
    const started = await dictation.start();
    if (started) void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  }

  function save() {
    if (!parsed) {
      setError('Enter a weight between 25 and 350 kg.');
      return;
    }
    applyOperations([{ type: 'weight', action: 'set', amount: parsed.kg }]);
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    router.dismiss();
  }

  const listening = dictation.state === 'listening';
  const shownTranscript = dictation.transcript.trim() || heard;
  const goalCopy = weightToTargetCopy(parsed?.kg ?? currentWeight, targetWeight);

  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <ScreenHeader
          action={(
            <Tap accessibilityLabel="Close weight log" hitSlop={10} onPress={() => router.dismiss()} scaleTo={0.9} style={styles.close}>
              <Glyph color={palette.ink} name="close" size={18} />
            </Tap>
          )}
          eyebrow="Progress check-in"
          style={styles.header}
          title="Log current weight"
        />

        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          style={styles.fill}>
          <Reveal>
            <Card glow raised>
              <Text style={styles.label}>SCALE READING</Text>
              <View style={styles.inputShell}>
                <TextInput
                  accessibilityLabel="Current weight in kilograms"
                  autoFocus
                  keyboardType="decimal-pad"
                  onChangeText={(next) => {
                    setValue(next.replace(/[^\d.,]/g, ''));
                    setError('');
                  }}
                  placeholder="89.0"
                  placeholderTextColor={palette.inkLow}
                  selectionColor={palette.lime}
                  selectTextOnFocus
                  style={styles.input}
                  value={value}
                />
                <Text style={styles.unit}>kg</Text>
              </View>
              <Text style={styles.helper}>Type only the scale number. Nothing is estimated.</Text>

              {targetWeight ? (
                <Well style={styles.targetWell}>
                  <View style={styles.targetRow}>
                    <View>
                      <Text style={styles.metaLabel}>TARGET</Text>
                      <Text style={styles.targetValue}>{targetWeight.toFixed(1)} kg</Text>
                    </View>
                    <Text style={styles.targetCopy}>{goalCopy}</Text>
                  </View>
                </Well>
              ) : (
                <Well style={styles.targetWell}>
                  <Text style={styles.targetCopy}>No target weight yet. Add one in Profile & goals to put this trend in context.</Text>
                </Well>
              )}
            </Card>
          </Reveal>

          <Reveal index={1} style={styles.gap}>
            <Card>
              <View style={styles.voiceRow}>
                <Tap
                  accessibilityLabel={listening ? 'Stop listening' : 'Say current weight'}
                  onPress={() => void toggleVoice()}
                  scaleTo={0.94}
                  style={[styles.mic, listening && styles.micListening]}>
                  <Glyph color={palette.onLime} name={listening ? 'pause' : 'mic'} size={23} />
                </Tap>
                <View style={styles.voiceCopy}>
                  <Text style={styles.voiceTitle}>{listening ? 'Listening…' : 'Say the scale number'}</Text>
                  <Text style={styles.voiceBody}>
                    {listening ? 'Say “89 kilos” or just “89”.' : 'This screen already knows you are logging weight.'}
                  </Text>
                </View>
              </View>
              {shownTranscript ? (
                <Well style={styles.heard}>
                  <Text style={styles.metaLabel}>I HEARD</Text>
                  <Text style={styles.heardText}>“{shownTranscript}”</Text>
                  {parsed ? <Text style={styles.resolved}>Ready to save as {parsed.kg.toFixed(1)} kg</Text> : null}
                </Well>
              ) : null}
            </Card>
          </Reveal>

          {error || dictation.error ? (
            <Reveal index={2} style={styles.gap}>
              <View style={styles.error}>
                <Glyph color={palette.danger} name="alert" size={16} />
                <Text style={styles.errorText}>{error || dictation.error}</Text>
              </View>
            </Reveal>
          ) : null}
        </ScrollView>

        <GlassFooter>
          <PrimaryButton disabled={!parsed} icon="check" label={parsed ? `Save ${parsed.kg.toFixed(1)} kg` : 'Enter current weight'} onPress={save} />
        </GlassFooter>
      </KeyboardAvoidingView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  header: { paddingHorizontal: space.md },
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },
  close: { width: 42, height: 42, borderRadius: 15, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: palette.line, backgroundColor: palette.surface },
  label: { ...text.label, color: palette.lime },
  inputShell: { flexDirection: 'row', alignItems: 'baseline', marginTop: 12, borderBottomWidth: 1, borderBottomColor: palette.lineHi },
  input: { ...text.hero, ...tabular, flex: 1, minWidth: 0, color: palette.ink, paddingVertical: 10 },
  unit: { ...text.section, color: palette.inkMid, paddingBottom: 17 },
  helper: { ...text.caption, color: palette.inkLow, marginTop: 10 },
  targetWell: { marginTop: 16 },
  targetRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: space.md },
  metaLabel: { ...text.label, fontSize: 8.5, color: palette.inkLow },
  targetValue: { ...text.value, ...tabular, color: palette.ink, marginTop: 3 },
  targetCopy: { ...text.caption, flex: 1, color: palette.inkMid, textAlign: 'right' },
  gap: { marginTop: space.md },
  voiceRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  mic: { ...shadow.glowSoft, width: 56, height: 56, borderRadius: 28, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.lime },
  micListening: { backgroundColor: palette.danger },
  voiceCopy: { flex: 1 },
  voiceTitle: { ...text.row, color: palette.ink },
  voiceBody: { ...text.caption, color: palette.inkMid, marginTop: 4 },
  heard: { marginTop: 14 },
  heardText: { ...text.body, color: palette.ink, marginTop: 5 },
  resolved: { ...text.caption, color: palette.lime, marginTop: 7 },
  error: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm, borderWidth: 1, borderColor: `${palette.danger}55`, borderRadius: radius.sm, backgroundColor: `${palette.danger}12`, padding: 14 },
  errorText: { ...text.caption, flex: 1, color: palette.danger },
});
