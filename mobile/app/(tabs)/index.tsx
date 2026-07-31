import { type Href, useRouter } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { Glyph } from '@/src/components/glyph';
import {
  Card,
  CountUp,
  GhostButton,
  ListRow,
  MacroChip,
  Metric,
  Reveal,
  Ring,
  Screen,
  ScreenHeader,
  SectionTitle,
  Tap,
  VoiceBar,
  Well,
} from '@/src/components/ui';
import { friendlyDay, greeting } from '@/src/lib/date';
import { dayTotals, slotLabels, workoutTotals } from '@/src/lib/stats';
import { bestStreak, loggingStreak, weekActivity, type IntakeStatus } from '@/src/lib/streak';
import { useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { macroColor, palette, space, tabular, text } from '@/src/theme';
import type { MealSlot } from '@/src/types';

const SLOTS: { slot: MealSlot; icon: 'sun' | 'bowl' | 'spark' | 'sleep' }[] = [
  { slot: 'breakfast', icon: 'sun' },
  { slot: 'lunch', icon: 'bowl' },
  { slot: 'snack', icon: 'spark' },
  { slot: 'dinner', icon: 'sleep' },
];

/**
 * The week strip carries the shape of the week, not just its presence: dim
 * where nothing was eaten, lime for a day that landed on target, coral for one
 * that ran well past it.
 */
const DOT_COLOR: Record<IntakeStatus, string> = {
  none: palette.inkFaint,
  on: palette.lime,
  over: palette.danger,
};

/** One forward-looking line — what today can still do for the streak. */
function streakOutlook(streak: number, best: number, todayLogged: boolean) {
  if (!todayLogged) {
    if (streak > 0) return `Log anything today to make it ${streak + 1} in a row.`;
    return best > 0
      ? `One entry today restarts the run — your best is ${best} days.`
      : 'Log one thing today and the streak starts.';
  }
  if (streak >= best) return `Personal best — ${streak} day${streak === 1 ? '' : 's'} and counting.`;
  const gap = best - streak;
  return `${gap} more day${gap === 1 ? '' : 's'} to match your best of ${best}.`;
}

export default function TodayScreen() {
  const router = useRouter();
  const { session } = useAuth();
  const { data, today, addWater, vaultReset, dismissVaultReset } = useApp();

  const totals = dayTotals(today);
  const burned = workoutTotals(today);
  const goal = Math.max(1, data.goals.calories);
  const remaining = Math.round(data.goals.calories - totals.calories);
  const ratio = totals.calories / goal;
  const streak = loggingStreak(data);
  const best = bestStreak(data);
  const week = weekActivity(data);
  const todayLogged = week[week.length - 1].active;
  const over = remaining < 0;

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          action={
            <Tap
              accessibilityLabel="Account and settings"
              onPress={() => router.push('/account' as Href)}
              scaleTo={0.92}>
              <View style={styles.avatar}>
                <Text style={styles.avatarText}>
                  {(session?.user.displayName ?? 'ME').slice(0, 2).toUpperCase()}
                </Text>
              </View>
            </Tap>
          }
          eyebrow={friendlyDay()}
          title={greeting()}
        />

        {/* ---------- Local data this device could not unlock ----------
            The device key is deliberately excluded from backups, but the
            database is not, so a restored phone arrives holding data it can
            never read. Saying so plainly beats appearing to have lost it. */}
        {vaultReset ? (
          <Reveal>
            <Card style={styles.notice}>
              <View style={styles.noticeHead}>
                <Glyph color={palette.warn} name="alert" size={16} />
                <Text style={styles.noticeTitle}>Local history was reset</Text>
              </View>
              <Text style={styles.noticeBody}>
                This device found saved data it could not unlock. That happens when a backup is
                restored onto a new phone, because the encryption key never leaves the device that
                made it.{' '}
                {session
                  ? 'Your account copy is syncing back now.'
                  : 'Signing in restores anything you had backed up.'}
              </Text>
              <GhostButton
                compact
                label="Got it"
                onPress={dismissVaultReset}
                style={styles.noticeButton}
              />
            </Card>
          </Reveal>
        ) : null}

        {/* ---------- Hero ----------
            One number with one meaning. The ring and the figure inside it now
            read the same way round: the ring is how much of the day is spent,
            the number is what is left to spend. The percentage is spoken to
            screen readers rather than printed a third time. */}
        <Reveal index={1}>
          <Card glow raised>
            <View style={styles.hero}>
              <View
                accessible
                accessibilityLabel={`${Math.abs(remaining).toLocaleString()} kcal ${
                  over ? 'over target' : 'left'
                }, ${Math.round(ratio * 100)} percent of a ${data.goals.calories.toLocaleString()} kcal target`}>
                <Ring size={164} thickness={12} value={ratio}>
                  <CountUp numberOfLines={1} style={styles.heroNumber} value={Math.abs(remaining)} />
                  <Text style={[styles.heroLabel, over && styles.heroLabelOver]}>
                    {over ? 'KCAL OVER' : 'KCAL LEFT'}
                  </Text>
                </Ring>
              </View>

              <Text style={styles.heroSub}>of {data.goals.calories.toLocaleString()} kcal</Text>

              {/* `flame` is the streak glyph on this screen and nothing else —
                  energy burned reads as `trend` in `palette.fat`, so one icon
                  never carries two meanings 12pt apart. */}
              {burned.calories > 0 ? (
                <View style={styles.burnRow}>
                  <Glyph color={palette.fat} name="trend" size={12} />
                  <Text style={styles.burnText}>+{Math.round(burned.calories)} burned today</Text>
                </View>
              ) : null}
            </View>

            <View style={styles.macros}>
              <MacroChip color={macroColor.protein} goal={data.goals.protein} label="Protein" value={totals.protein} />
              <MacroChip color={macroColor.carbs} goal={data.goals.carbs} label="Carbs" value={totals.carbs} />
              <MacroChip color={macroColor.fat} goal={data.goals.fat} label="Fat" value={totals.fat} />
            </View>
          </Card>
        </Reveal>

        {/* ---------- Streak ---------- */}
        <Reveal index={2} style={styles.streakBlock}>
          <Well>
            <View style={styles.streakHead}>
              <Glyph color={streak > 0 ? palette.lime : palette.inkLow} name="flame" size={15} />
              <Text style={styles.streakText}>
                {streak > 0 ? `${streak} day${streak === 1 ? '' : 's'} in a row` : 'Start your streak today'}
              </Text>
            </View>
            <View style={styles.week}>
              {week.map((day) => (
                <View
                  accessible
                  accessibilityLabel={`${day.label}: ${
                    day.calories > 0 ? `${Math.round(day.calories)} kcal` : 'nothing logged'
                  }`}
                  key={day.key}
                  style={styles.weekDay}>
                  <View
                    style={[
                      styles.weekDot,
                      { backgroundColor: DOT_COLOR[day.status], borderColor: DOT_COLOR[day.status] },
                      day.isToday && styles.weekDotToday,
                    ]}
                  />
                  <Text style={[styles.weekLabel, day.isToday && styles.weekLabelToday]}>{day.label}</Text>
                </View>
              ))}
            </View>
            <Text style={styles.streakNote}>{streakOutlook(streak, best, todayLogged)}</Text>
          </Well>
        </Reveal>

        {/* ---------- Log ---------- */}
        <Reveal index={3} style={styles.voice}>
          <VoiceBar />
        </Reveal>

        {/* ---------- Metrics ----------
            Deliberately colourless. `macroColor` owns lime / blue / orange for
            protein / carbs / fat, and repeating those three hues 12pt below
            made colour mean two different things on one screen. `Metric` tints
            its icon chip from `accent`, so a neutral accent is what gives the
            inkMid glyph on a quiet chip. */}
        <Reveal index={4} style={styles.metrics}>
          <Metric
            accent={palette.inkMid}
            detail="Tap to add 250 ml"
            icon="water"
            label="Water"
            onPress={() => addWater(250)}
            value={`${(today.waterMl / 1000).toFixed(1)} L`}
          />
          <Metric
            accent={palette.inkMid}
            detail={`${data.goals.steps.toLocaleString()} goal`}
            icon="steps"
            label="Steps"
            value={today.steps.toLocaleString()}
          />
          <Metric
            accent={palette.inkMid}
            detail={today.sleepHours ? 'From Health' : 'Not synced'}
            icon="sleep"
            label="Sleep"
            value={today.sleepHours ? `${today.sleepHours} h` : '—'}
          />
        </Reveal>

        {/* ---------- Meals ---------- */}
        <SectionTitle title="Today's meals" />
        <Reveal index={5}>
          <Card padded={false} style={styles.mealCard}>
            {SLOTS.map(({ slot, icon }, index) => {
              const meals = today.meals.filter((meal) => meal.slot === slot);
              const calories = meals.reduce((sum, meal) => sum + meal.calories, 0);
              // A `~` only where the entries really carry a spread. A gram
              // weight or a package serving is not a guess, and printing one as
              // though it were would invent an uncertainty the data lacks.
              const estimated = meals.some(
                (meal) =>
                  meal.calorieLow != null && meal.calorieHigh != null && meal.calorieLow !== meal.calorieHigh,
              );
              return (
                <ListRow
                  accent={meals.length ? palette.lime : palette.inkLow}
                  detail={meals.length ? meals.map((meal) => meal.name).join(', ') : 'Tap to log'}
                  icon={icon}
                  key={slot}
                  last={index === SLOTS.length - 1}
                  onPress={() => router.push({ pathname: '/quick-log', params: { slot } })}
                  title={slotLabels[slot]}
                  unit="kcal"
                  value={calories ? `${estimated ? '~' : ''}${Math.round(calories)}` : undefined}
                />
              );
            })}
          </Card>
        </Reveal>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  notice: { borderColor: `${palette.warn}44`, marginBottom: space.sm },
  noticeHead: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 7 },
  noticeTitle: { ...text.row, color: palette.ink },
  noticeBody: { ...text.body, fontSize: 12.5, color: palette.inkMid },
  noticeButton: { marginTop: 12, alignSelf: 'flex-start' },

  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: palette.surfaceHi,
    borderWidth: 1,
    borderColor: palette.lineHi,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: { ...text.value, fontSize: 12, color: palette.lime },

  hero: { alignItems: 'center' },
  heroNumber: { ...text.hero, color: palette.ink, textAlign: 'center', ...tabular },
  heroLabel: { ...text.label, color: palette.lime, marginTop: space.xs, textAlign: 'center' },
  heroLabelOver: { color: palette.danger },
  heroSub: { ...text.caption, color: palette.inkMid, marginTop: space.sm },
  burnRow: { flexDirection: 'row', alignItems: 'center', gap: space.xs, marginTop: space.xs },
  burnText: { ...text.caption, color: palette.inkMid },

  macros: { flexDirection: 'row', gap: space.xs, marginTop: space.md },

  streakBlock: { marginTop: space.sm },
  streakHead: { flexDirection: 'row', alignItems: 'center', gap: space.xs, marginBottom: space.sm },
  streakText: { ...text.value, color: palette.ink },
  week: { flexDirection: 'row', justifyContent: 'space-between' },
  weekDay: { alignItems: 'center', gap: space.xs, flex: 1 },
  weekDot: { width: 9, height: 9, borderRadius: 5, borderWidth: 1 },
  weekDotToday: { transform: [{ scale: 1.35 }] },
  weekLabel: { ...text.micro, fontSize: 9, color: palette.inkLow },
  weekLabelToday: { color: palette.ink },
  streakNote: { ...text.caption, color: palette.inkMid, marginTop: space.sm },

  voice: { marginTop: space.md },

  metrics: { flexDirection: 'row', gap: space.xs, marginTop: space.md },

  mealCard: { paddingHorizontal: 14 },
});
