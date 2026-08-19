import { useLocalSearchParams, useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import {
  Alert,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { Card, GhostButton, GlassFooter, PrimaryButton, Screen, Segmented } from '@/src/components/ui';
import { slotLabels } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, text } from '@/src/theme';
import type { MealItem, MealSlot } from '@/src/types';

const SLOT_OPTIONS = (Object.keys(slotLabels) as MealSlot[]).map((value) => ({ value, label: slotLabels[value] }));

export default function EditMealScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { data, removeMeal, updateMeal } = useApp();
  const located = useMemo(() => {
    for (const [date, day] of Object.entries(data.days)) {
      const meal = day.meals.find((candidate) => candidate.id === id);
      if (meal) return { date, meal };
    }
    return undefined;
  }, [data.days, id]);
  const meal = located?.meal;
  const [name, setName] = useState(meal?.name ?? '');
  const [quantity, setQuantity] = useState(meal?.quantity ?? '');
  const [calories, setCalories] = useState(meal ? String(meal.calories) : '');
  const [protein, setProtein] = useState(meal ? String(meal.protein) : '');
  const [carbs, setCarbs] = useState(meal ? String(meal.carbs) : '');
  const [fat, setFat] = useState(meal ? String(meal.fat) : '');
  const [slot, setSlot] = useState<MealSlot>(meal?.slot ?? 'snack');
  const [date, setDate] = useState(located?.date ?? '');

  const dateValid = /^\d{4}-\d{2}-\d{2}$/.test(date) && !Number.isNaN(Date.parse(`${date}T12:00:00Z`));
  const numbersValid = [calories, protein, carbs, fat].every((value) => value.trim() !== '' && Number(value) >= 0);
  const valid = Boolean(meal && name.trim() && quantity.trim() && dateValid && numbersValid);

  function save() {
    if (!meal || !valid) return;
    const patch: Partial<MealItem> = {
      name: name.trim(),
      quantity: quantity.trim(),
      calories: Number(calories),
      protein: Number(protein),
      carbs: Number(carbs),
      fat: Number(fat),
      calorieLow: Number(calories),
      calorieHigh: Number(calories),
      confidence: 'high',
      basis: 'Corrected manually after logging.',
      sourceLabel: 'User correction',
      source: 'manual',
      assumptions: [],
      slot,
    };
    updateMeal(meal.id, patch, date);
    Keyboard.dismiss();
    router.back();
  }

  function remove() {
    if (!meal) return;
    Alert.alert('Delete this food?', 'This removes the logged item from every synced device.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => { removeMeal(meal.id); router.back(); } },
    ]);
  }

  if (!meal) {
    return (
      <Screen edges={['bottom']} style={styles.center}>
        <Text style={styles.title}>This food is no longer available.</Text>
        <GhostButton label="Go back" onPress={() => router.back()} />
      </Screen>
    );
  }

  return (
    <Screen edges={['bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          onTouchMove={Keyboard.dismiss}>
          <Card>
            <Text style={styles.title}>Correct the logged food</Text>
            <Text style={styles.body}>The corrected values become exact user-provided figures; the previous estimate range is removed.</Text>
            <Field label="Food name" onChange={setName} value={name} />
            <Field label="Amount or serving" onChange={setQuantity} value={quantity} />
          </Card>

          <Card>
            <Text style={styles.label}>MEAL</Text>
            <Segmented onChange={setSlot} options={SLOT_OPTIONS} value={slot} />
            <View style={styles.metrics}>
              <Field keyboard="decimal-pad" label="Calories" onChange={setCalories} unit="kcal" value={calories} />
              <Field keyboard="decimal-pad" label="Protein" onChange={setProtein} unit="g" value={protein} />
              <Field keyboard="decimal-pad" label="Carbs" onChange={setCarbs} unit="g" value={carbs} />
              <Field keyboard="decimal-pad" label="Fat" onChange={setFat} unit="g" value={fat} />
            </View>
            <Field label="Date" onChange={setDate} placeholder="YYYY-MM-DD" value={date} />
            {!dateValid ? <Text style={styles.error}>Use a real date in YYYY-MM-DD format.</Text> : null}
          </Card>

          <GhostButton icon="trash" label="Delete logged food" onPress={remove} tone="danger" />
        </ScrollView>
        <GlassFooter>
          <PrimaryButton disabled={!valid} icon="check" label="Save correction" onPress={save} />
        </GlassFooter>
      </KeyboardAvoidingView>
    </Screen>
  );
}

function Field({
  keyboard,
  label,
  onChange,
  placeholder,
  unit,
  value,
}: {
  keyboard?: 'decimal-pad';
  label: string;
  onChange: (value: string) => void;
  placeholder?: string;
  unit?: string;
  value: string;
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label.toUpperCase()}</Text>
      <View style={styles.inputShell}>
        <TextInput
          accessibilityLabel={label}
          keyboardType={keyboard}
          onChangeText={(next) => onChange(keyboard ? next.replace(/[^\d.]/g, '') : next)}
          placeholder={placeholder}
          placeholderTextColor={palette.inkLow}
          returnKeyType="done"
          style={styles.input}
          value={value}
        />
        {unit ? <Text style={styles.unit}>{unit}</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  center: { justifyContent: 'center', padding: space.md, gap: 16 },
  content: { padding: space.md, paddingBottom: space.tabClearance, gap: 12 },
  title: { ...text.section, color: palette.ink },
  body: { ...text.caption, color: palette.inkMid, marginTop: 6, marginBottom: 14 },
  label: { ...text.label, color: palette.inkLow, marginBottom: 7 },
  field: { flex: 1, marginTop: 12 },
  inputShell: { minHeight: 52, flexDirection: 'row', alignItems: 'center', borderRadius: radius.sm, borderWidth: 1, borderColor: palette.lineHi, backgroundColor: palette.surfaceLo, paddingHorizontal: 13 },
  input: { ...text.row, flex: 1, color: palette.ink, paddingVertical: 13 },
  unit: { ...text.caption, color: palette.inkLow },
  metrics: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  error: { ...text.caption, color: palette.danger, marginTop: 7 },
});
