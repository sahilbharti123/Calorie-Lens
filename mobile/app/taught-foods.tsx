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
  Card,
  GhostButton,
  PrimaryButton,
  Reveal,
  Screen,
  ScreenHeader,
  Segmented,
  Tap,
  Well,
} from '@/src/components/ui';
import { aliasesForName } from '@/src/lib/nutrition';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, tabular, text } from '@/src/theme';
import type { LearnedFood } from '@/src/types';

/**
 * Everything the app has learned, and a way to correct it.
 *
 * A name is derived from dictation, which means it is sometimes wrong — "whole
 * garden beer hoegarden" is a plausible thing to end up with. A saved figure
 * that cannot be renamed is worse than one that was never saved: it goes on
 * being applied under a name the user does not recognise, to entries they
 * cannot connect it to.
 */
type Unit = 'g' | 'ml' | 'piece' | 'bowl' | 'cup';
const UNITS: { value: Unit; label: string }[] = [
  { value: 'g', label: 'g' },
  { value: 'ml', label: 'ml' },
  { value: 'piece', label: 'piece' },
  { value: 'bowl', label: 'bowl' },
  { value: 'cup', label: 'cup' },
];

export default function TaughtFoodsScreen() {
  const router = useRouter();
  const { data, forgetFood, updateLearnedFood } = useApp();
  const [openId, setOpenId] = useState<string | null>(null);

  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <ScreenHeader
          action={(
            <Tap
              accessibilityLabel="Close"
              hitSlop={10}
              onPress={() => router.dismiss()}
              scaleTo={0.9}
              style={styles.close}>
              <Glyph color={palette.ink} name="close" size={18} />
            </Tap>
          )}
          eyebrow={data.learnedFoods.length ? `${data.learnedFoods.length} saved` : 'Nothing saved yet'}
          style={styles.header}
          title="Foods you've taught"
        />

        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          style={styles.fill}>
          {data.learnedFoods.length ? (
            data.learnedFoods.map((food, index) => (
              <Reveal index={index} key={food.id} style={index ? styles.gap : undefined}>
                <FoodRow
                  food={food}
                  onForget={() => {
                    forgetFood(food.id);
                    setOpenId(null);
                    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
                  }}
                  onSave={(patch) => {
                    updateLearnedFood(food.id, patch);
                    setOpenId(null);
                    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                  }}
                  onToggle={() => setOpenId(openId === food.id ? null : food.id)}
                  open={openId === food.id}
                />
              </Reveal>
            ))
          ) : (
            <Reveal>
              <Card>
                <Text style={styles.emptyTitle}>Nothing taught yet</Text>
                <Text style={styles.emptyBody}>
                  When the app meets a food it does not recognise and you give it the calories, it
                  saves that figure here and uses it next time instead of asking again. Everything
                  it learns can be renamed, corrected or removed on this screen.
                </Text>
              </Card>
            </Reveal>
          )}

          {data.learnedFoods.length ? (
            <Reveal index={data.learnedFoods.length} style={styles.gap}>
              <Well style={styles.note}>
                <Glyph color={palette.inkMid} name="info" size={15} />
                <Text style={styles.noteText}>
                  The name is what the app listens for. Rename it to what you would actually say —
                  a shorter, distinctive name matches more reliably than a long one.
                </Text>
              </Well>
            </Reveal>
          ) : null}
        </ScrollView>
      </KeyboardAvoidingView>
    </Screen>
  );
}

function FoodRow({
  food,
  open,
  onToggle,
  onSave,
  onForget,
}: {
  food: LearnedFood;
  open: boolean;
  onToggle: () => void;
  onSave: (patch: Partial<LearnedFood>) => void;
  onForget: () => void;
}) {
  const [name, setName] = useState(food.name);
  const [calories, setCalories] = useState(String(food.calories));
  const [amount, setAmount] = useState(food.servingAmount ? String(food.servingAmount) : '');
  const [unit, setUnit] = useState<Unit>((food.servingUnit as Unit) ?? 'g');

  const trimmed = name.trim();
  const kcal = Number.parseFloat(calories);
  const servingAmount = Number.parseFloat(amount);
  const valid = Boolean(trimmed) && Number.isFinite(kcal) && kcal > 0;
  const serving = food.servingAmount && food.servingUnit
    ? `${food.servingAmount} ${food.servingUnit}`
    : 'the saved amount';

  return (
    <Card padded={false} style={styles.rowCard}>
      <Tap accessibilityLabel={`Edit ${food.name}`} onPress={onToggle} style={styles.rowHead}>
        <View style={styles.rowCopy}>
          <Text numberOfLines={1} style={styles.rowTitle}>{food.name}</Text>
          <Text style={styles.rowDetail}>
            {food.calories.toLocaleString()} kcal per {serving}
          </Text>
        </View>
        <View style={open ? styles.chevronOpen : undefined}>
          <Glyph color={palette.inkLow} name="chevronDown" size={16} />
        </View>
      </Tap>

      {open ? (
        <View style={styles.editor}>
          <Text style={styles.label}>NAME</Text>
          <View style={styles.inputShell}>
            <TextInput
              accessibilityLabel="Food name"
              autoCapitalize="none"
              onChangeText={setName}
              placeholder="e.g. hoegaarden"
              placeholderTextColor={palette.inkLow}
              selectionColor={palette.lime}
              style={styles.input}
              value={name}
            />
          </View>

          <View style={styles.pair}>
            <View style={styles.pairCell}>
              <Text style={styles.label}>CALORIES</Text>
              <View style={styles.inputShell}>
                <TextInput
                  accessibilityLabel="Calories"
                  keyboardType="number-pad"
                  onChangeText={(next) => setCalories(next.replace(/[^\d.]/g, ''))}
                  placeholderTextColor={palette.inkLow}
                  selectionColor={palette.lime}
                  style={styles.input}
                  value={calories}
                />
              </View>
            </View>
            <View style={styles.pairCell}>
              <Text style={styles.label}>PER</Text>
              <View style={styles.inputShell}>
                <TextInput
                  accessibilityLabel="Serving amount"
                  keyboardType="number-pad"
                  onChangeText={(next) => setAmount(next.replace(/[^\d.]/g, ''))}
                  placeholder="1"
                  placeholderTextColor={palette.inkLow}
                  selectionColor={palette.lime}
                  style={styles.input}
                  value={amount}
                />
              </View>
            </View>
          </View>

          <Segmented onChange={setUnit} options={UNITS} style={styles.units} value={unit} />

          <Text style={styles.helper}>
            A different amount later is scaled from this one in proportion.
          </Text>

          <PrimaryButton
            compact
            disabled={!valid}
            icon="check"
            label="Save changes"
            onPress={() => onSave({
              name: trimmed,
              // The name is what matching listens for, so the phrases follow it.
              aliases: aliasesForName(trimmed),
              calories: Math.round(kcal),
              servingAmount: Number.isFinite(servingAmount) && servingAmount > 0
                ? servingAmount
                : undefined,
              servingUnit: Number.isFinite(servingAmount) && servingAmount > 0 ? unit : undefined,
            })}
            style={styles.save}
          />
          <GhostButton
            compact
            icon="trash"
            label="Forget this food"
            onPress={onForget}
            style={styles.forget}
            tone="danger"
          />
        </View>
      ) : null}
    </Card>
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
  gap: { marginTop: space.sm },

  rowCard: { paddingHorizontal: 14 },
  rowHead: { flexDirection: 'row', alignItems: 'center', gap: space.sm, paddingVertical: 14 },
  rowCopy: { flex: 1 },
  rowTitle: { ...text.row, color: palette.ink },
  rowDetail: { ...text.caption, ...tabular, color: palette.inkMid, marginTop: 3 },
  chevronOpen: { transform: [{ rotate: '180deg' }] },

  editor: { borderTopWidth: 1, borderTopColor: palette.line, paddingVertical: 14 },
  label: { ...text.label, color: palette.inkLow, marginBottom: 6 },
  inputShell: {
    borderWidth: 1,
    borderColor: palette.lineHi,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceHi,
    paddingHorizontal: 12,
  },
  input: { ...text.body, ...tabular, color: palette.ink, paddingVertical: 10 },
  pair: { flexDirection: 'row', gap: space.sm, marginTop: 12 },
  pairCell: { flex: 1 },
  units: { marginTop: 12 },
  helper: { ...text.caption, color: palette.inkLow, marginTop: 10 },
  save: { marginTop: 14 },
  forget: { marginTop: 8 },

  emptyTitle: { ...text.section, color: palette.ink },
  emptyBody: { ...text.body, color: palette.inkMid, marginTop: space.xs },
  note: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm, padding: 14 },
  noteText: { ...text.caption, color: palette.inkMid, flex: 1 },
});
