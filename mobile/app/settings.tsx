import * as Haptics from 'expo-haptics';
import { type Href, useRouter } from 'expo-router';
import { type ComponentProps, useState } from 'react';
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

import { Glyph } from '@/src/components/glyph';
import {
  Card,
  CountUp,
  GlassFooter,
  ListRow,
  PrimaryButton,
  Reveal,
  Screen,
  SectionTitle,
  Tap,
} from '@/src/components/ui';
import { useApp } from '@/src/store/app-store';
import { targetWeightError } from '@/src/lib/weight';
import { palette, radius, space, tabular, text } from '@/src/theme';
import type { Goals } from '@/src/types';

const PROFILE_OPTIONS = {
  activityLevel: [
    ['mostly-seated', 'Mostly seated'], ['lightly-active', 'Lightly active'],
    ['active', 'Active'], ['very-active', 'Very active'],
  ],
  goalPace: [['gentle', 'Gentle'], ['steady', 'Steady'], ['ambitious', 'Ambitious']],
  workoutPreference: [
    ['gym', 'Gym'], ['walking', 'Walking'], ['home', 'Home'], ['mixed', 'Mixed'], ['restarting', 'Restarting'],
  ],
  experienceLevel: [['new', 'New'], ['some', 'Some'], ['experienced', 'Experienced']],
  dietStyle: [
    ['home-indian', 'Home Indian'], ['vegetarian', 'Vegetarian'], ['vegan', 'Vegan'],
    ['mixed', 'Mixed'], ['high-protein', 'High protein'],
  ],
  mainChallenge: [
    ['portions', 'Portions'], ['protein', 'Protein'], ['cravings', 'Cravings'],
    ['time', 'Time'], ['consistency', 'Consistency'],
  ],
  coachingTone: [['gentle', 'Gentle'], ['direct', 'Direct'], ['data-led', 'Data-led']],
} as const;

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
  const [profile, setProfile] = useState(data.profile);
  const [trainingDays, setTrainingDays] = useState(String(data.profile.trainingDays ?? 3));
  const [availableMinutes, setAvailableMinutes] = useState(String(data.profile.availableMinutes ?? 30));
  const [mealsPerDay, setMealsPerDay] = useState(String(data.profile.mealsPerDay ?? 3));
  const [allergies, setAllergies] = useState(data.profile.allergies.join(', '));
  const [injuries, setInjuries] = useState(data.profile.injuries.join(', '));

  function save() {
    const parsedWeight = Number(weightKg);
    const parsedTargetWeight = Number(targetWeightKg);
    const parsedBowl = Number(bowlMl);
    const parsedStrengthDays = Number(values.strengthDays);
    savePersonalization(
      {
        ...profile,
        weightKg: Number.isFinite(parsedWeight) && parsedWeight > 0
          ? Math.min(250, Math.max(35, parsedWeight))
          : data.profile.weightKg,
        targetWeightKg: Number.isFinite(parsedTargetWeight) && parsedTargetWeight > 0
          ? Math.min(250, Math.max(35, parsedTargetWeight))
          : undefined,
        trainingDays: clampInteger(trainingDays, 0, 7, profile.trainingDays ?? 3),
        availableMinutes: clampInteger(availableMinutes, 5, 180, profile.availableMinutes ?? 30),
        mealsPerDay: clampInteger(mealsPerDay, 1, 8, profile.mealsPerDay ?? 3),
        allergies: splitList(allergies),
        injuries: splitList(injuries),
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
        0,
        values.weeklyWorkoutMinutes.trim() !== '' && Number.isFinite(Number(values.weeklyWorkoutMinutes))
          ? Number(values.weeklyWorkoutMinutes)
          : data.goals.weeklyWorkoutMinutes,
      ),
      strengthDays: Math.min(
        7,
        Math.max(
          0,
          values.strengthDays.trim() !== '' && Number.isFinite(parsedStrengthDays)
            ? parsedStrengthDays
            : data.goals.strengthDays,
        ),
      ),
    });
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    router.dismiss();
  }

  function setGoal(key: keyof Goals) {
    return (value: string) => setValues((current) => ({ ...current, [key]: value }));
  }

  const previewCalories = Number(values.calories) || 0;
  // Target weight is optional — Progress says "Not set yet" and `save()` stores
  // undefined for it. Validating an empty field greyed out Save permanently, so
  // someone who skipped onboarding could never change their calories again.
  const target = Number(targetWeightKg) || undefined;
  const targetError = target
    ? targetWeightError(data.profile.primaryGoal, Number(weightKg) || currentWeight, target)
    : null;

  return (
    <Screen edges={['bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          onTouchMove={Keyboard.dismiss}
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

          {/* ---------- Foods the app has been taught ----------
              Anything the app learns has to be visible and removable. A figure
              it picked up from one entry would otherwise be applied to every
              future entry with no way to see it, let alone correct it. */}
          <Reveal index={2}>
            <SectionTitle title="Foods you've taught" />
            <Card padded={false} style={styles.linkCard}>
              <ListRow
                detail={data.learnedFoods.length
                  ? 'Rename, correct a figure, or remove one'
                  : 'Nothing yet — give a food its calories once and it is saved'}
                icon="bowl"
                last
                onPress={() => router.push('/taught-foods' as Href)}
                title={data.learnedFoods.length
                  ? `${data.learnedFoods.length} saved`
                  : 'Nothing saved yet'}
              />
            </Card>
          </Reveal>

          {/* ---------- Daily goals ---------- */}
          <Reveal index={3}>
            <SectionTitle title="Plan preferences" />
            <Card style={styles.preferencesCard}>
              <ChoiceField
                label="Daily activity"
                onChange={(value) => setProfile((current) => ({ ...current, activityLevel: value }))}
                options={PROFILE_OPTIONS.activityLevel}
                value={profile.activityLevel ?? 'lightly-active'}
              />
              <ChoiceField
                label="Goal pace"
                onChange={(value) => setProfile((current) => ({ ...current, goalPace: value }))}
                options={PROFILE_OPTIONS.goalPace}
                value={profile.goalPace ?? 'gentle'}
              />
              <ChoiceField
                label="Training style"
                onChange={(value) => setProfile((current) => ({ ...current, workoutPreference: value }))}
                options={PROFILE_OPTIONS.workoutPreference}
                value={profile.workoutPreference ?? 'mixed'}
              />
              <ChoiceField
                label="Experience"
                onChange={(value) => setProfile((current) => ({ ...current, experienceLevel: value }))}
                options={PROFILE_OPTIONS.experienceLevel}
                value={profile.experienceLevel ?? 'some'}
              />
              <ChoiceField
                label="Diet style"
                onChange={(value) => setProfile((current) => ({ ...current, dietStyle: value }))}
                options={PROFILE_OPTIONS.dietStyle}
                value={profile.dietStyle ?? 'mixed'}
              />
              <ChoiceField
                label="Main challenge"
                onChange={(value) => setProfile((current) => ({ ...current, mainChallenge: value }))}
                options={PROFILE_OPTIONS.mainChallenge}
                value={profile.mainChallenge ?? 'consistency'}
              />
              <ChoiceField
                label="Coach style"
                onChange={(value) => setProfile((current) => ({ ...current, coachingTone: value }))}
                options={PROFILE_OPTIONS.coachingTone}
                value={profile.coachingTone ?? 'gentle'}
              />
              <View style={styles.grid}>
                <Field label="Training days" onChangeText={setTrainingDays} unit="/ week" value={trainingDays} />
                <Field label="Time available" onChangeText={setAvailableMinutes} unit="min" value={availableMinutes} />
                <Field label="Meals per day" onChangeText={setMealsPerDay} unit="meals" value={mealsPerDay} />
              </View>
              <TextField label="Allergies" onChangeText={setAllergies} placeholder="e.g. dairy, peanuts" value={allergies} />
              <TextField label="Injuries or limitations" onChangeText={setInjuries} placeholder="e.g. sore right knee" value={injuries} />
              <Text style={styles.helper}>Separate multiple items with commas. Coach suggestions use these limits, but do not replace clinical advice.</Text>
            </Card>
          </Reveal>

          <Reveal index={4}>
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

          <Reveal index={5}>
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

function ChoiceField<T extends string>({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: T) => void;
  options: readonly (readonly [T, string])[];
  value: T;
}) {
  return (
    <View style={styles.choiceField}>
      <Text style={styles.fieldLabel}>{label.toUpperCase()}</Text>
      <View style={styles.choices}>
        {options.map(([option, title]) => {
          const selected = value === option;
          return (
            <Tap
              accessibilityLabel={`${label}: ${title}${selected ? ', selected' : ''}`}
              accessibilityRole="button"
              key={option}
              onPress={() => onChange(option)}
              style={[styles.choice, selected && styles.choiceSelected]}>
              <Text style={[styles.choiceText, selected && styles.choiceTextSelected]}>{title}</Text>
            </Tap>
          );
        })}
      </View>
    </View>
  );
}

function TextField(props: { label: string; onChangeText: (value: string) => void; placeholder: string; value: string }) {
  return (
    <View style={styles.choiceField}>
      <Text style={styles.fieldLabel}>{props.label.toUpperCase()}</Text>
      <TextInput
        accessibilityLabel={props.label}
        onChangeText={props.onChangeText}
        placeholder={props.placeholder}
        placeholderTextColor={palette.inkLow}
        returnKeyType="done"
        style={[styles.shell, styles.textInput]}
        value={props.value}
      />
    </View>
  );
}

function splitList(value: string) {
  return [...new Set(value.split(',').map((item) => item.trim()).filter(Boolean))];
}

function clampInteger(value: string, min: number, max: number, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(max, Math.max(min, Math.round(parsed))) : fallback;
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
          accessibilityLabel={props.accessibilityLabel ?? `${label}, ${unit}`}
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
  preferencesCard: { gap: 18 },
  choiceField: { gap: 7 },
  choices: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  choice: { minHeight: 44, justifyContent: 'center', paddingHorizontal: 13, borderRadius: radius.pill, borderWidth: 1, borderColor: palette.lineHi, backgroundColor: palette.surfaceLo },
  choiceSelected: { borderColor: palette.lime, backgroundColor: palette.limeSoft },
  choiceText: { ...text.caption, color: palette.inkMid },
  choiceTextSelected: { color: palette.lime },
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
  textInput: { ...text.body, color: palette.ink, paddingHorizontal: 13, paddingVertical: 12 },
  unit: { ...text.caption, fontSize: 10.5, color: palette.inkLow },

  linkCard: { paddingHorizontal: 14 },
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
