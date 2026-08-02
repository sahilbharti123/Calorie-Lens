import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { type ComponentProps, useState } from 'react';
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
  CountUp,
  GlassFooter,
  PrimaryButton,
  Reveal,
  Screen,
  SectionTitle,
} from '@/src/components/ui';
import { useApp } from '@/src/store/app-store';
import { targetWeightError } from '@/src/lib/weight';
import { palette, radius, space, tabular, text } from '@/src/theme';
import type { Goals } from '@/src/types';

export default function SettingsScreen() {
  const router = useRouter();
  const { data, savePersonalization, updateGoals } = useApp();
  const currentWeight = data.weights.at(-1)?.kg ?? data.profile.weightKg;
  const [values, setValues] = useState<Record<keyof Goals, string>>({
    calories: String(data.goals.calories),
    protein: String(data.goals.protein),
    carbs: String(data.goals.carbs),
    fat: String(data.goals.fat),
    waterMl: String(data.goals.waterMl),
    steps: String(data.goals.steps),
    weeklyWorkoutMinutes: String(data.goals.weeklyWorkoutMinutes),
    strengthDays: String(data.goals.strengthDays),
  });
  const [weightKg, setWeightKg] = useState(currentWeight ? String(currentWeight) : '');
  const [targetWeightKg, setTargetWeightKg] = useState(data.profile.targetWeightKg ? String(data.profile.targetWeightKg) : '');
  const [bowlMl, setBowlMl] = useState(data.estimation.bowlMl ? String(data.estimation.bowlMl) : '');

  function save() {
    const parsedWeight = Number(weightKg);
    const parsedTargetWeight = Number(targetWeightKg);
    const parsedBowl = Number(bowlMl);
    savePersonalization(
      {
        ...data.profile,
        weightKg: Number.isFinite(parsedWeight) && parsedWeight > 0
          ? Math.min(250, Math.max(35, parsedWeight))
          : data.profile.weightKg,
        targetWeightKg: Number.isFinite(parsedTargetWeight) && parsedTargetWeight > 0
          ? Math.min(250, Math.max(35, parsedTargetWeight))
          : undefined,
      },
      Number.isFinite(parsedBowl) && parsedBowl > 0
        ? Math.min(1000, Math.max(50, parsedBowl))
        : undefined,
    );
    updateGoals({
      calories: Math.max(500, Number(values.calories) || data.goals.calories),
      protein: Math.max(10, Number(values.protein) || data.goals.protein),
      carbs: Math.max(20, Number(values.carbs) || data.goals.carbs),
      fat: Math.max(20, Number(values.fat) || data.goals.fat),
      waterMl: Math.max(250, Number(values.waterMl) || data.goals.waterMl),
      steps: Math.max(500, Number(values.steps) || data.goals.steps),
      weeklyWorkoutMinutes: Math.max(
        10,
        Number(values.weeklyWorkoutMinutes) || data.goals.weeklyWorkoutMinutes,
      ),
      strengthDays: Math.min(
        7,
        Math.max(0, Number(values.strengthDays) || data.goals.strengthDays),
      ),
    });
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    router.dismiss();
  }

  function setGoal(key: keyof Goals) {
    return (value: string) => setValues((current) => ({ ...current, [key]: value }));
  }

  const previewCalories = Number(values.calories) || 0;
  const targetError = targetWeightError(
    data.profile.primaryGoal,
    Number(weightKg) || currentWeight,
    Number(targetWeightKg) || undefined,
  );

  return (
    <Screen edges={['bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          style={styles.flex}>

          {/* ---------- Hero ---------- */}
          <Reveal>
            <Card glow raised>
              <View style={styles.heroHead}>
                <View style={styles.heroIcon}>
                  <Glyph color={palette.lime} name="target" size={18} />
                </View>
                <Text style={styles.heroLabel}>DAILY ENERGY TARGET</Text>
              </View>
              <View style={styles.heroValueRow}>
                <CountUp style={styles.heroValue} value={previewCalories} />
                <Text style={styles.heroUnit}>kcal</Text>
              </View>
              <Text style={styles.heroTitle}>Calibrate once. Log faster.</Text>
              <Text style={styles.heroBody}>
                Current weight calibrates energy. Target weight gives the plan and Progress trend a direction.
              </Text>
            </Card>
          </Reveal>

          {/* ---------- Estimation profile ---------- */}
          <Reveal index={1}>
            <SectionTitle title="Estimation profile" />
            <Card>
              <View style={styles.grid}>
                <Field
                  label="Current weight"
                  onChangeText={setWeightKg}
                  placeholder="e.g. 72"
                  unit="kg"
                  value={weightKg}
                />
                <Field
                  label="Target weight"
                  onChangeText={setTargetWeightKg}
                  placeholder="e.g. 82"
                  unit="kg"
                  value={targetWeightKg}
                />
                <Field
                  label="Usual bowl size"
                  onChangeText={setBowlMl}
                  placeholder="e.g. 200"
                  unit="ml"
                  value={bowlMl}
                />
              </View>
              <Text style={styles.helper}>
                {targetError ?? 'The target should reflect the direction you selected during onboarding. Progress uses it as context, not judgment.'}
              </Text>
              <Text style={styles.helper}>
                Fill your usual bowl with water and measure it once. If left blank, logging asks before estimating a bowl.
              </Text>
            </Card>
          </Reveal>

          {/* ---------- Daily goals ---------- */}
          <Reveal index={2}>
            <SectionTitle title="Daily goals" />
            <Card>
              <View style={styles.grid}>
                <Field
                  label="Calories"
                  onChangeText={setGoal('calories')}
                  unit="kcal"
                  value={values.calories}
                />
                <Field
                  label="Protein"
                  onChangeText={setGoal('protein')}
                  unit="g"
                  value={values.protein}
                />
                <Field
                  label="Carbohydrates"
                  onChangeText={setGoal('carbs')}
                  unit="g"
                  value={values.carbs}
                />
                <Field
                  label="Fat"
                  onChangeText={setGoal('fat')}
                  unit="g"
                  value={values.fat}
                />
                <Field
                  label="Water"
                  onChangeText={setGoal('waterMl')}
                  unit="ml"
                  value={values.waterMl}
                />
                <Field
                  label="Steps"
                  onChangeText={setGoal('steps')}
                  unit="steps"
                  value={values.steps}
                />
                <Field
                  label="Weekly training"
                  onChangeText={setGoal('weeklyWorkoutMinutes')}
                  unit="min"
                  value={values.weeklyWorkoutMinutes}
                />
                <Field
                  label="Strength sessions"
                  onChangeText={setGoal('strengthDays')}
                  unit="/ week"
                  value={values.strengthDays}
                />
              </View>
            </Card>
          </Reveal>

          <Reveal index={3}>
            <Text style={styles.note}>
              Food and exercise values remain estimates. Vigorly shows the source and range before saving.
            </Text>
          </Reveal>
        </ScrollView>

        <GlassFooter>
          <PrimaryButton disabled={Boolean(targetError)} icon="check" label="Save profile and goals" onPress={save} />
        </GlassFooter>
      </KeyboardAvoidingView>
    </Screen>
  );
}

/** Dark numeric field: inset well, hairline border that lights up on focus. */
function Field({
  label,
  unit,
  ...props
}: Omit<ComponentProps<typeof TextInput>, 'onChangeText'> & {
  label: string;
  onChangeText: (value: string) => void;
  unit: string;
}) {
  const [focused, setFocused] = useState(false);
  const { onChangeText } = props;
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label.toUpperCase()}</Text>
      <View style={[styles.shell, focused && styles.shellFocused]}>
        <TextInput
          keyboardType="number-pad"
          selectTextOnFocus
          {...props}
          onBlur={() => setFocused(false)}
          onChangeText={(next) => onChangeText(next.replace(/[^\d.]/g, ''))}
          onFocus={() => setFocused(true)}
          placeholderTextColor={palette.inkLow}
          style={styles.input}
        />
        <Text numberOfLines={1} style={styles.unit}>{unit}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { paddingHorizontal: space.md, paddingTop: space.sm, paddingBottom: space.tabClearance },

  heroHead: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  heroIcon: {
    width: 32,
    height: 32,
    borderRadius: 11,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroLabel: { ...text.label, color: palette.lime },
  heroValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 7, marginTop: 14 },
  heroValue: { ...text.hero, color: palette.ink, ...tabular },
  heroUnit: { ...text.caption, color: palette.inkMid },
  heroTitle: { ...text.section, color: palette.ink, marginTop: 16 },
  heroBody: { ...text.caption, color: palette.inkMid, marginTop: 5 },

  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  field: { flexBasis: '40%', flexGrow: 1, gap: 7 },
  fieldLabel: { ...text.label, color: palette.inkLow },
  shell: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    minHeight: 52,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surfaceLo,
    paddingHorizontal: 13,
  },
  shellFocused: { borderColor: palette.lime },
  input: {
    ...text.row,
    ...tabular,
    flex: 1,
    color: palette.ink,
    paddingVertical: 14,
  },
  unit: { ...text.caption, fontSize: 10.5, color: palette.inkLow },

  helper: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: 14 },

  note: {
    ...text.caption,
    fontSize: 10.5,
    color: palette.inkLow,
    textAlign: 'center',
    marginTop: space.lg,
    paddingHorizontal: space.md,
  },
});
