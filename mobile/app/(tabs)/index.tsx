import { useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Circle } from 'react-native-svg';

import { Glyph } from '@/src/components/glyph';
import { Metric, ProgressBar, ScreenHeader, SectionTitle, VoiceBar } from '@/src/components/ui';
import { friendlyDay, greeting } from '@/src/lib/date';
import { dayTotals, slotLabels } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';
import type { MealSlot } from '@/src/types';

const slots: MealSlot[] = ['breakfast', 'lunch', 'snack', 'dinner'];

export default function TodayScreen() {
  const router = useRouter();
  const { data, today, addWater } = useApp();
  const totals = dayTotals(today);
  const remaining = Math.max(0, data.goals.calories - totals.calories);
  const dayScore = Math.min(100, Math.round((totals.calories / data.goals.calories) * 100));

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader eyebrow={friendlyDay()} title={greeting()} action={
          <Pressable onPress={() => router.push('/settings')} style={styles.avatar}>
            <Text style={styles.avatarText}>SB</Text>
          </Pressable>
        } />

        <View style={styles.hero}>
          <View style={styles.heroTop}>
            <View>
              <Text style={styles.heroLabel}>ENERGY LEFT</Text>
              <View style={styles.heroNumberRow}>
                <Text style={styles.heroNumber}>{remaining.toLocaleString()}</Text>
                <Text style={styles.heroUnit}>kcal</Text>
              </View>
            </View>
            <ScoreRing value={dayScore} />
          </View>
          <ProgressBar value={totals.calories / data.goals.calories} />
          <View style={styles.heroMeta}>
            <Text style={styles.heroMetaText}>{Math.round(totals.calories)} eaten</Text>
            <Text style={styles.heroMetaText}>{data.goals.calories} target</Text>
          </View>
        </View>

        <VoiceBar />

        <View style={styles.metrics}>
          <Pressable style={{ flex: 1 }} onPress={() => addWater(250)}>
            <Metric icon="water" label="Water" value={`${(today.waterMl / 1000).toFixed(1)} L`} detail="Tap · +250 ml" />
          </Pressable>
          <Metric icon="steps" label="Steps" value={today.steps.toLocaleString()} detail={`${data.goals.steps.toLocaleString()} goal`} />
          <Metric icon="sleep" label="Sleep" value={today.sleepHours ? `${today.sleepHours} h` : '—'} detail="From health sync" />
        </View>

        <SectionTitle title="Today’s meals" aside={`${Math.round(totals.protein)} g protein`} />
        <View style={styles.mealList}>
          {slots.map((slot, index) => {
            const meals = today.meals.filter((meal) => meal.slot === slot);
            const calories = meals.reduce((sum, meal) => sum + meal.calories, 0);
            return (
              <Pressable
                key={slot}
                onPress={() => router.push({ pathname: '/quick-log', params: { slot } })}
                style={[styles.mealRow, index < slots.length - 1 && styles.mealRowBorder]}>
                <View style={[styles.slotDot, meals.length > 0 && styles.slotDotFilled]} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.mealTitle}>{slotLabels[slot]}</Text>
                  <Text numberOfLines={1} style={styles.mealDetail}>
                    {meals.length ? meals.map((meal) => meal.name).join(', ') : 'Tap to log'}
                  </Text>
                </View>
                <Text style={styles.mealCalories}>{calories ? `${Math.round(calories)} kcal` : 'Add'}</Text>
                <Glyph name="chevron" color={palette.muted} size={16} />
              </Pressable>
            );
          })}
        </View>

        <View style={styles.coach}>
          <View style={styles.coachIcon}><Glyph name="spark" color={palette.forest} size={21} /></View>
          <View style={{ flex: 1 }}>
            <Text style={styles.coachLabel}>COACH NOTE</Text>
            <Text style={styles.coachText}>
              {totals.protein < data.goals.protein * 0.5
                ? 'Protein is still light. Make the next meal eggs, paneer, chicken, curd, or a whey shake.'
                : 'Your protein pace looks solid. Keep dinner simple and stay near your energy target.'}
            </Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function ScoreRing({ value }: { value: number }) {
  const radius = 25;
  const circumference = 2 * Math.PI * radius;
  return (
    <View style={styles.score}>
      <Svg width={64} height={64} viewBox="0 0 64 64" style={StyleSheet.absoluteFill}>
        <Circle cx="32" cy="32" r={radius} fill="none" stroke="#344138" strokeWidth="5" />
        <Circle
          cx="32"
          cy="32"
          r={radius}
          fill="none"
          stroke={palette.lime}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={circumference * (1 - value / 100)}
          rotation="-90"
          origin="32, 32"
        />
      </Svg>
      <Text style={styles.scoreValue}>{value}</Text>
      <Text style={styles.scoreLabel}>DAY</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { paddingHorizontal: space.md, paddingBottom: 28 },
  avatar: { width: 40, height: 40, borderRadius: 20, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: palette.lime, fontFamily: type.demi, fontSize: 12 },
  hero: { backgroundColor: palette.forest, borderRadius: radius.lg, padding: 20, marginBottom: 12 },
  heroTop: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 },
  heroLabel: { color: '#AEB9B0', fontFamily: type.demi, fontSize: 10, letterSpacing: 1.4 },
  heroNumberRow: { flexDirection: 'row', alignItems: 'baseline', gap: 7, marginTop: 3 },
  heroNumber: { color: palette.white, fontFamily: type.demi, fontSize: 42, letterSpacing: -2 },
  heroUnit: { color: '#AEB9B0', fontFamily: type.medium, fontSize: 13 },
  score: { width: 64, height: 64, alignItems: 'center', justifyContent: 'center' },
  scoreValue: { color: palette.white, fontFamily: type.demi, fontSize: 17 },
  scoreLabel: { color: '#AEB9B0', fontFamily: type.demi, fontSize: 8, letterSpacing: 1 },
  heroMeta: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 9 },
  heroMetaText: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 11 },
  metrics: { flexDirection: 'row', gap: 8, marginVertical: 22 },
  mealList: { backgroundColor: palette.paper, borderRadius: radius.md, borderWidth: 1, borderColor: palette.line, paddingHorizontal: 14, marginBottom: 20 },
  mealRow: { minHeight: 70, flexDirection: 'row', alignItems: 'center', gap: 10 },
  mealRowBorder: { borderBottomWidth: 1, borderBottomColor: palette.line },
  slotDot: { width: 10, height: 10, borderRadius: 5, borderWidth: 1.5, borderColor: palette.muted },
  slotDotFilled: { backgroundColor: palette.lime, borderColor: palette.limeDark },
  mealTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 14 },
  mealDetail: { color: palette.muted, fontFamily: type.regular, fontSize: 11, marginTop: 2 },
  mealCalories: { color: palette.ink, fontFamily: type.medium, fontSize: 11 },
  coach: { flexDirection: 'row', gap: 12, padding: 16, backgroundColor: palette.softLime, borderRadius: radius.md },
  coachIcon: { width: 38, height: 38, borderRadius: 12, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center' },
  coachLabel: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2, marginBottom: 4 },
  coachText: { color: palette.ink, fontFamily: type.medium, fontSize: 12, lineHeight: 18 },
});
