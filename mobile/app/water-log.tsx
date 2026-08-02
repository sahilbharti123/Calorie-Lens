import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useState } from 'react';
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
  Bar,
  Card,
  CountUp,
  GhostButton,
  GlassFooter,
  PrimaryButton,
  Reveal,
  Screen,
  ScreenHeader,
  Tap,
  Well,
} from '@/src/components/ui';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, tabular, text } from '@/src/theme';

/**
 * Water used to be logged by tapping a tile on the Today screen, which added
 * 250 ml on the spot with nothing to undo it. A single mis-tap was permanent
 * and invisible — the tile just read a slightly wrong number for the rest of
 * the day. Anything that writes to the log needs a way back, so every amount
 * here can be added, taken off again, or replaced outright.
 */
const QUICK_AMOUNTS = [150, 250, 500, 750];

export default function WaterLogScreen() {
  const router = useRouter();
  const { applyOperations, data, today } = useApp();
  const goal = Math.max(1, data.goals.waterMl);
  const [exact, setExact] = useState('');
  /** Amounts added on this visit, newest last — what Undo walks back. */
  const [added, setAdded] = useState<number[]>([]);

  function add(amount: number) {
    applyOperations([{ type: 'water', action: 'add', amount }]);
    setAdded((current) => [...current, amount]);
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  }

  function undo() {
    const last = added.at(-1);
    if (last === undefined) return;
    applyOperations([{ type: 'water', action: 'add', amount: -last }]);
    setAdded((current) => current.slice(0, -1));
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  }

  function setTotal(amount: number) {
    applyOperations([{ type: 'water', action: 'set', amount }]);
    // A total set outright replaces this visit's history — there is nothing
    // left to walk back to, and offering Undo would suggest otherwise.
    setAdded([]);
    setExact('');
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }

  const typed = Number.parseFloat(exact.replace(',', '.'));
  const typedValid = Number.isFinite(typed) && typed >= 0 && typed <= 10_000;
  const litres = (today.waterMl / 1000).toFixed(2);
  const percent = Math.round((today.waterMl / goal) * 100);
  const lastAdded = added.at(-1);

  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <ScreenHeader
          action={(
            <Tap
              accessibilityLabel="Close water log"
              hitSlop={10}
              onPress={() => router.dismiss()}
              scaleTo={0.9}
              style={styles.close}>
              <Glyph color={palette.ink} name="close" size={18} />
            </Tap>
          )}
          eyebrow="Hydration"
          style={styles.header}
          title="Water today"
        />

        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          style={styles.fill}>
          {/* ---------- Where the day stands ---------- */}
          <Reveal>
            <Card glow raised>
              <Text style={styles.label}>LOGGED TODAY</Text>
              <View
                accessible
                accessibilityLabel={`${today.waterMl} millilitres logged, ${percent} percent of your ${goal} millilitre goal`}
                style={styles.totalRow}>
                <CountUp decimals={2} style={styles.total} value={Number(litres)} />
                <Text style={styles.totalUnit}>L</Text>
              </View>
              <Text style={styles.helper}>
                {today.waterMl.toLocaleString()} ml of {goal.toLocaleString()} ml · {percent}%
              </Text>
              <View style={styles.bar}>
                <Bar height={5} value={today.waterMl / goal} />
              </View>
            </Card>
          </Reveal>

          {/* ---------- Add ---------- */}
          <Reveal index={1} style={styles.gap}>
            <Card>
              <Text style={styles.sectionLabel}>ADD A DRINK</Text>
              <View style={styles.chips}>
                {QUICK_AMOUNTS.map((amount) => (
                  <Tap
                    accessibilityLabel={`Add ${amount} millilitres`}
                    key={amount}
                    onPress={() => add(amount)}
                    scaleTo={0.94}
                    style={styles.chip}>
                    <Text style={styles.chipValue}>+{amount}</Text>
                    <Text style={styles.chipUnit}>ml</Text>
                  </Tap>
                ))}
              </View>

              {/* The way back. Present only when there is something to undo, so
                  it never implies an earlier entry can be walked back here. */}
              {lastAdded !== undefined ? (
                <View style={styles.undoRow}>
                  <GhostButton
                    icon="close"
                    label={`Undo the last ${lastAdded} ml`}
                    onPress={undo}
                  />
                </View>
              ) : null}
            </Card>
          </Reveal>

          {/* ---------- Correct ---------- */}
          <Reveal index={2} style={styles.gap}>
            <Card>
              <Text style={styles.sectionLabel}>OR SET THE TOTAL</Text>
              <Text style={styles.helper}>
                Replaces today&apos;s figure rather than adding to it. Use this when the running
                total has drifted, or set 0 to start the day again.
              </Text>
              <View style={styles.inputShell}>
                <TextInput
                  accessibilityLabel="Total water today in millilitres"
                  keyboardType="number-pad"
                  onChangeText={(next) => setExact(next.replace(/[^\d.,]/g, ''))}
                  placeholder={String(today.waterMl)}
                  placeholderTextColor={palette.inkLow}
                  selectionColor={palette.lime}
                  selectTextOnFocus
                  style={styles.input}
                  value={exact}
                />
                <Text style={styles.unit}>ml</Text>
              </View>
              <View style={styles.setRow}>
                <PrimaryButton
                  compact
                  disabled={!typedValid}
                  icon="check"
                  label={typedValid ? `Set today to ${typed.toLocaleString()} ml` : 'Enter a total'}
                  onPress={() => setTotal(Math.round(typed))}
                />
              </View>
              {today.waterMl > 0 ? (
                <View style={styles.clearRow}>
                  <GhostButton icon="trash" label="Clear today's water" onPress={() => setTotal(0)} />
                </View>
              ) : null}
            </Card>
          </Reveal>

          <Reveal index={3} style={styles.gap}>
            <Well style={styles.note}>
              <Glyph color={palette.inkMid} name="info" size={15} />
              <Text style={styles.noteText}>
                Water is the one number here you enter rather than estimate, so nothing on this
                screen carries a range. Your daily goal is set in Profile &amp; goals.
              </Text>
            </Well>
          </Reveal>
        </ScrollView>

        <GlassFooter>
          <PrimaryButton icon="check" label="Done" onPress={() => router.dismiss()} />
        </GlassFooter>
      </KeyboardAvoidingView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  header: { paddingHorizontal: space.md },
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },
  close: {
    width: 42,
    height: 42,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surface,
  },

  label: { ...text.label, color: palette.lime },
  sectionLabel: { ...text.label, color: palette.inkLow },
  totalRow: { flexDirection: 'row', alignItems: 'baseline', gap: 8, marginTop: 10 },
  total: { ...text.hero, ...tabular, color: palette.ink },
  totalUnit: { ...text.section, color: palette.inkMid },
  helper: { ...text.caption, color: palette.inkMid, marginTop: 8 },
  bar: { marginTop: 14 },

  gap: { marginTop: space.md },

  chips: { flexDirection: 'row', gap: space.xs, marginTop: 12 },
  chip: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceHi,
    alignItems: 'center',
  },
  chipValue: { ...text.value, ...tabular, color: palette.ink },
  chipUnit: { ...text.micro, color: palette.inkLow, marginTop: 2 },
  undoRow: { marginTop: 12 },

  inputShell: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginTop: 14,
    borderBottomWidth: 1,
    borderBottomColor: palette.lineHi,
  },
  input: { ...text.headline, ...tabular, flex: 1, minWidth: 0, color: palette.ink, paddingVertical: 8 },
  unit: { ...text.value, color: palette.inkMid, paddingBottom: 10 },
  setRow: { marginTop: 14 },
  clearRow: { marginTop: 10 },

  note: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm, padding: 14 },
  noteText: { ...text.caption, color: palette.inkMid, flex: 1 },
});
