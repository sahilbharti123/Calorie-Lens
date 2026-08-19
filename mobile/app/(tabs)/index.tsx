import { type Href, useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { useMemo, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, View } from 'react-native';

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
import { mealGroupToOperation, recentMealGroups, savedMealToOperation, type MealGroup } from '@/src/lib/meals';
import { syncNativeHealth } from '@/src/lib/health';
import { dayTotals, slotLabels, workoutTotals } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { macroColor, palette, space, tabular, text } from '@/src/theme';
import type { MealSlot, SavedMeal } from '@/src/types';

const SLOTS: { slot: MealSlot; icon: 'sun' | 'bowl' | 'spark' | 'sleep' }[] = [
  { slot: 'breakfast', icon: 'sun' },
  { slot: 'lunch', icon: 'bowl' },
  { slot: 'snack', icon: 'spark' },
  { slot: 'dinner', icon: 'sleep' },
];

export default function TodayScreen() {
  const router = useRouter();
  const { session } = useAuth();
  const {
    data,
    today,
    applyHealthSnapshot,
    applyOperations,
    dismissVaultReset,
    reportHealthSyncError,
    removeMealBatch,
    saveMeal,
    vaultReset,
  } = useApp();
  const [healthBusy, setHealthBusy] = useState(false);
  const [undoBatch, setUndoBatch] = useState<{ id: string; label: string } | null>(null);

  const totals = dayTotals(today);
  const loggedWorkoutEnergy = workoutTotals(today).calories;
  const burnedCalories = today.activeCalories > 0 ? today.activeCalories : loggedWorkoutEnergy;
  const burnedSource = today.activeCalories > 0
    ? (data.healthSync?.source ?? 'connected health')
    : 'logged workouts';
  const goal = Math.max(1, data.goals.calories);
  const remaining = Math.round(data.goals.calories - totals.calories);
  const ratio = totals.calories / goal;
  const over = remaining < 0;
  const recentMeals = useMemo(() => recentMealGroups(data, 3), [data]);
  const nextSlot = slotForTime();
  const firstName = (session?.user.displayName ?? 'there').trim().split(/\s+/)[0];

  async function syncHealth() {
    if (healthBusy) return;
    setHealthBusy(true);
    try {
      applyHealthSnapshot(await syncNativeHealth());
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error) {
      reportHealthSyncError(error instanceof Error ? error.message : 'Health sync failed. Try again.');
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setHealthBusy(false);
    }
  }

  function repeatRecent(group: MealGroup) {
    const batchId = mealBatchId();
    applyOperations([{ ...mealGroupToOperation(group, nextSlot), batchId }]);
    setUndoBatch({ id: batchId, label: group.name });
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }

  function repeatSaved(meal: SavedMeal) {
    const batchId = mealBatchId();
    applyOperations([{ ...savedMealToOperation(meal, nextSlot), batchId }]);
    setUndoBatch({ id: batchId, label: meal.name });
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }

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
          title={`${greeting()}, ${firstName}`}
        />

        <HealthFreshness
          busy={healthBusy}
          onPress={() => void syncHealth()}
          record={data.healthSync}
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
          <Card>
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
              {burnedCalories > 0 ? (
                <View style={styles.burnRow}>
                  <Glyph color={palette.fat} name="trend" size={12} />
                  <Text style={styles.burnText}>
                    +{Math.round(burnedCalories)} active kcal · {burnedSource}
                  </Text>
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

        {/* ---------- Log ---------- */}
        <Reveal index={2} style={styles.voice}>
          <VoiceBar
            label="Speak or type"
            slot={nextSlot}
            title={`Log ${slotLabels[nextSlot].toLowerCase()}`}
          />
        </Reveal>

        {data.savedMeals.length || recentMeals.length ? (
          <View style={styles.repeatSection}>
            <SectionTitle aside="One tap" title="Repeat a meal" />
            <View style={styles.repeatList}>
              {data.savedMeals.slice(0, 2).map((meal) => (
                <RepeatMealRow
                  detail="Saved meal"
                  key={meal.id}
                  name={meal.name}
                  onPress={() => repeatSaved(meal)}
                  totals={meal.items.reduce(
                    (sum, item) => ({ calories: sum.calories + item.calories, protein: sum.protein + item.protein }),
                    { calories: 0, protein: 0 },
                  )}
                />
              ))}
              {recentMeals
                .filter((group) => !data.savedMeals.some((meal) => savedMatchesGroup(meal, group)))
                .slice(0, Math.max(1, 3 - data.savedMeals.length))
                .map((group) => (
                  <RepeatMealRow
                    detail={relativeMealDay(group.date)}
                    key={group.key}
                    name={group.name}
                    onFavorite={() => saveMeal(group)}
                    onPress={() => repeatRecent(group)}
                    totals={{ calories: group.calories, protein: group.protein }}
                  />
                ))}
            </View>
            {undoBatch ? (
              <Well style={styles.repeatUndo}>
                <Text numberOfLines={1} style={styles.repeatUndoText}>{undoBatch.label} logged</Text>
                <Tap
                  accessibilityLabel={`Undo logging ${undoBatch.label}`}
                  onPress={() => {
                    removeMealBatch(undoBatch.id);
                    setUndoBatch(null);
                  }}
                  style={styles.repeatUndoAction}>
                  <Text style={styles.repeatUndoActionText}>UNDO</Text>
                </Tap>
              </Well>
            ) : null}
          </View>
        ) : null}

        {/* ---------- Metrics ----------
            Deliberately colourless. `macroColor` owns lime / blue / orange for
            protein / carbs / fat, and repeating those three hues 12pt below
            made colour mean two different things on one screen. `Metric` tints
            its icon chip from `accent`, so a neutral accent is what gives the
            inkMid glyph on a quiet chip. */}
        <Reveal index={3} style={styles.metrics}>
          {/* Opens the water screen rather than logging on the spot. A tile
              that wrote 250 ml straight into the day gave a mis-tap no way
              back — the number was simply wrong for the rest of the day. */}
          <Metric
            accent={palette.inkMid}
            detail={`${Math.round((today.waterMl / Math.max(1, data.goals.waterMl)) * 100)}% of goal`}
            icon="water"
            label="Water"
            onPress={() => router.push('/water-log' as Href)}
            progress={today.waterMl / Math.max(1, data.goals.waterMl)}
            value={`${(today.waterMl / 1000).toFixed(1)} L`}
          />
          <Metric
            accent={palette.inkMid}
            detail={`${data.goals.steps.toLocaleString()} goal`}
            icon="steps"
            label="Steps"
            progress={today.steps / Math.max(1, data.goals.steps)}
            value={today.steps.toLocaleString()}
          />
          <Metric
            accent={palette.inkMid}
            detail={today.sleepHours ? 'From Health' : 'Not synced'}
            icon="sleep"
            label="Sleep"
            progress={(today.sleepHours ?? 0) / 8}
            value={today.sleepHours ? `${today.sleepHours} h` : '—'}
          />
        </Reveal>

        {/* ---------- Meals ---------- */}
        <SectionTitle title="Today's meals" />
        <Reveal index={4}>
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

  voice: { marginTop: space.md },

  healthRow: {
    minHeight: 52,
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: palette.line,
    paddingVertical: 8,
    marginBottom: space.md,
  },
  healthIcon: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  healthTitle: { ...text.value, color: palette.ink },
  healthDetail: { ...text.caption, color: palette.inkMid, marginTop: 2 },
  repeatSection: { marginTop: space.sm },
  repeatList: { borderTopWidth: 1, borderBottomWidth: 1, borderColor: palette.line },
  repeatUndo: { marginTop: space.sm, minHeight: 48, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  repeatUndoText: { ...text.caption, color: palette.ink, flex: 1 },
  repeatUndoAction: { minWidth: 64, minHeight: 44, alignItems: 'center', justifyContent: 'center' },
  repeatUndoActionText: { ...text.label, color: palette.lime },
  repeatRow: { minHeight: 68, flexDirection: 'row', alignItems: 'center', gap: space.sm },
  repeatCopy: { flex: 1 },
  repeatName: { ...text.row, color: palette.ink },
  repeatDetail: { ...text.caption, color: palette.inkMid, marginTop: 3 },
  repeatAction: { width: 44, height: 44, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },

  metrics: { flexDirection: 'row', gap: space.xs, marginTop: space.md },

  mealCard: { paddingHorizontal: 14 },
});

function HealthFreshness({
  busy,
  onPress,
  record,
}: {
  busy: boolean;
  onPress: () => void;
  record: ReturnType<typeof useApp>['data']['healthSync'];
}) {
  const nativeHealth = Platform.OS === 'ios' || Platform.OS === 'android';
  const title = record?.source ?? (Platform.OS === 'ios' ? 'Apple Health' : 'Health Connect');
  const detail = !nativeHealth
    ? 'Native app only'
    : busy
    ? 'Refreshing…'
    : record?.status === 'error'
      ? 'Tap to retry'
      : record?.status === 'empty'
        ? 'No shared data'
        : record?.latestSampleAt
          ? `Latest sample ${relativeTime(record.latestSampleAt)}`
          : record?.lastSuccessAt
          ? `Checked ${relativeTime(record.lastSuccessAt)}`
          : 'Tap to connect';
  const color = record?.status === 'error'
    ? palette.danger
    : record?.status === 'current'
      ? palette.lime
      : palette.inkMid;
  const content = (
    <View style={styles.healthRow}>
      <View style={[styles.healthIcon, { backgroundColor: `${color}14` }]}>
        <Glyph color={color} name={record?.status === 'error' ? 'alert' : 'heart'} size={17} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.healthTitle}>{title}</Text>
        <Text style={styles.healthDetail}>{detail}</Text>
      </View>
      {nativeHealth ? <Glyph color={color} name="restart" size={17} /> : null}
    </View>
  );
  return nativeHealth ? (
    <Tap accessibilityLabel={`${title}. ${detail}`} onPress={onPress} scaleTo={0.985}>{content}</Tap>
  ) : content;
}

function RepeatMealRow({
  detail,
  name,
  onFavorite,
  onPress,
  totals,
}: {
  detail: string;
  name: string;
  onFavorite?: () => void;
  onPress: () => void;
  totals: { calories: number; protein: number };
}) {
  return (
    <View style={styles.repeatRow}>
      <Tap accessibilityLabel={`Repeat ${name}`} onPress={onPress} scaleTo={0.985} style={styles.repeatCopy}>
        <Text numberOfLines={1} style={styles.repeatName}>{name}</Text>
        <Text style={styles.repeatDetail}>
          {Math.round(totals.calories)} kcal · {Math.round(totals.protein)} g protein · {detail}
        </Text>
      </Tap>
      {onFavorite ? (
        <Tap accessibilityLabel={`Save ${name}`} onPress={onFavorite} scaleTo={0.9} style={styles.repeatAction}>
          <Glyph color={palette.inkMid} name="star" size={18} />
        </Tap>
      ) : null}
      <Tap accessibilityLabel={`Repeat ${name}`} onPress={onPress} scaleTo={0.9} style={styles.repeatAction}>
        <Glyph color={palette.lime} name="restart" size={18} />
      </Tap>
    </View>
  );
}

function slotForTime(): MealSlot {
  const hour = new Date().getHours();
  if (hour < 11) return 'breakfast';
  if (hour < 15) return 'lunch';
  if (hour < 18) return 'snack';
  return 'dinner';
}

function mealBatchId() {
  return `meal-batch-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function relativeTime(iso: string) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60_000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.floor(hours / 24)} d ago`;
}

function relativeMealDay(date: string) {
  const value = new Date(`${date}T12:00:00`);
  const days = Math.round((new Date().setHours(12, 0, 0, 0) - value.getTime()) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  return `${days} days ago`;
}

function savedMatchesGroup(saved: SavedMeal, group: MealGroup) {
  const savedNames = saved.items.map((item) => `${item.name}|${item.quantity}`).sort().join('::');
  const groupNames = group.items.map((item) => `${item.name}|${item.quantity}`).sort().join('::');
  return savedNames === groupNames;
}
