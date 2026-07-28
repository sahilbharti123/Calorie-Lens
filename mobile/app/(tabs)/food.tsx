import { useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { EmptyState, ProgressBar, ScreenHeader, VoiceBar } from '@/src/components/ui';
import { dayTotals, slotLabels } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';
import type { MealSlot } from '@/src/types';

const slots: MealSlot[] = ['breakfast', 'lunch', 'snack', 'dinner'];

export default function FoodScreen() {
  const router = useRouter();
  const { data, today, removeMeal } = useApp();
  const totals = dayTotals(today);

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader eyebrow="Nutrition" title="Food" />
        <View style={styles.energy}>
          <View style={styles.energyLine}>
            <View>
              <Text style={styles.energyLabel}>TODAY</Text>
              <Text style={styles.energyValue}>~{Math.round(totals.calories)} <Text style={styles.energyUnit}>kcal midpoint</Text></Text>
            </View>
            <Text style={styles.remaining}>{Math.max(0, data.goals.calories - totals.calories)} left</Text>
          </View>
          <ProgressBar value={totals.calories / data.goals.calories} />
          <View style={styles.macros}>
            <Macro label="Protein" value={totals.protein} target={data.goals.protein} />
            <Macro label="Carbs" value={totals.carbs} target={250} />
            <Macro label="Fat" value={totals.fat} target={70} />
          </View>
        </View>

        <VoiceBar label="Say “2 rotis and one bowl dal”" />

        <Text style={styles.sectionLabel}>MEALS</Text>
        {slots.map((slot) => {
          const meals = today.meals.filter((meal) => meal.slot === slot);
          return (
            <View key={slot} style={styles.slotCard}>
              <Pressable
                onPress={() => router.push({ pathname: '/quick-log', params: { slot } })}
                style={styles.slotHeader}>
                <View>
                  <Text style={styles.slotTitle}>{slotLabels[slot]}</Text>
                  <Text style={styles.slotMeta}>
                    {meals.length
                      ? mealRange(meals)
                      : 'Nothing logged'}
                  </Text>
                </View>
                <View style={styles.addButton}><Glyph name="plus" color={palette.forest} size={18} /></View>
              </Pressable>

              {meals.map((meal) => (
                <View key={meal.id} style={styles.mealItem}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.mealName}>{meal.name}</Text>
                    <Text style={styles.mealQuantity}>{meal.quantity} · {meal.protein} g protein</Text>
                  </View>
                  <Text style={styles.mealKcal}>
                    {meal.calorieLow != null && meal.calorieHigh != null
                      ? `${meal.calorieLow}–${meal.calorieHigh}`
                      : meal.calories} kcal
                  </Text>
                  <Pressable
                    accessibilityLabel={`Delete ${meal.name}`}
                    hitSlop={10}
                    onPress={() => removeMeal(meal.id)}>
                    <Glyph name="trash" color={palette.muted} size={17} />
                  </Pressable>
                </View>
              ))}
            </View>
          );
        })}

        {!today.meals.length ? (
          <EmptyState
            icon="bowl"
            title="Your day starts with one sentence"
            body="Type or say what you ate. Calorie Lens estimates calories and macros, then lets you review before saving."
          />
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function Macro({ label, value, target }: { label: string; value: number; target: number }) {
  return (
    <View style={styles.macro}>
      <Text style={styles.macroValue}>{Math.round(value)}g</Text>
      <Text style={styles.macroLabel}>{label}</Text>
      <View style={styles.macroTrack}>
        <View style={[styles.macroFill, { width: `${Math.min(100, (value / target) * 100)}%` }]} />
      </View>
    </View>
  );
}

function mealRange(meals: { calories: number; calorieLow?: number; calorieHigh?: number }[]) {
  const low = meals.reduce((sum, meal) => sum + (meal.calorieLow ?? meal.calories), 0);
  const high = meals.reduce((sum, meal) => sum + (meal.calorieHigh ?? meal.calories), 0);
  return `${Math.round(low)}–${Math.round(high)} kcal`;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { paddingHorizontal: space.md, paddingBottom: 28 },
  energy: { backgroundColor: palette.forest, borderRadius: radius.lg, padding: 20, marginBottom: 12 },
  energyLine: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 15 },
  energyLabel: { color: palette.lime, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.4 },
  energyValue: { color: palette.white, fontFamily: type.demi, fontSize: 34, letterSpacing: -1.4, marginTop: 2 },
  energyUnit: { color: '#AEB9B0', fontFamily: type.medium, fontSize: 12 },
  remaining: { color: '#CBD2CC', fontFamily: type.medium, fontSize: 12 },
  macros: { flexDirection: 'row', gap: 12, marginTop: 18 },
  macro: { flex: 1 },
  macroValue: { color: palette.white, fontFamily: type.demi, fontSize: 15 },
  macroLabel: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 10, marginVertical: 3 },
  macroTrack: { height: 3, borderRadius: 2, backgroundColor: '#344138', overflow: 'hidden' },
  macroFill: { height: 3, backgroundColor: palette.lime },
  sectionLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 10, letterSpacing: 1.4, marginTop: 24, marginBottom: 10 },
  slotCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, marginBottom: 10, overflow: 'hidden' },
  slotHeader: { minHeight: 67, paddingHorizontal: 15, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  slotTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 15 },
  slotMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 11, marginTop: 2 },
  addButton: { width: 34, height: 34, borderRadius: 17, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  mealItem: { minHeight: 62, marginHorizontal: 15, borderTopWidth: 1, borderTopColor: palette.line, flexDirection: 'row', alignItems: 'center', gap: 10 },
  mealName: { color: palette.ink, fontFamily: type.medium, fontSize: 13 },
  mealQuantity: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 2 },
  mealKcal: { color: palette.ink, fontFamily: type.demi, fontSize: 11 },
});
