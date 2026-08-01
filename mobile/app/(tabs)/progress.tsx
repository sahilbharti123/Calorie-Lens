import { useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import { TrendChart } from '@/src/components/trend-chart';
import {
  Bar,
  Card,
  CountUp,
  Reveal,
  Screen,
  ScreenHeader,
  SectionTitle,
  Segmented,
  Tap,
  Well,
} from '@/src/components/ui';
import { dateKey } from '@/src/lib/date';
import { healthSetupCopy, syncNativeHealth } from '@/src/lib/health';
import { dayTotals } from '@/src/lib/stats';
import { loggingStreak } from '@/src/lib/streak';
import { completedSessions } from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, tabular, text } from '@/src/theme';

type Range = '7' | '30' | '90';

const RANGES: { value: Range; label: string }[] = [
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
];

/** The result of the last `sync()`; '' before one has been attempted. */
type SyncOutcome = '' | 'synced' | 'empty' | 'failed';

/** Health data only exists in a native build — the same test `sync()` makes. */
const NATIVE_HEALTH = Platform.OS === 'ios' || Platform.OS === 'android';

const HEALTH_CATEGORIES = 'Steps, activity, sleep and weight';

/**
 * What the provider row is allowed to claim. The note panel underneath states
 * the last thing that actually happened, so the header must never outrank it:
 * a card reading "Connected" above a panel reading "connection unavailable"
 * teaches the user to distrust both.
 */
function connectionState(outcome: SyncOutcome, lastSync: string | undefined) {
  if (!NATIVE_HEALTH) return 'Not available in this build';
  if (outcome === 'synced') return 'Synced just now';
  if (outcome === 'empty') return 'No shared samples yet';
  if (outcome === 'failed') return 'Not syncing';
  return lastSync ? 'Connected' : 'Not connected yet';
}

export default function ProgressScreen() {
  const router = useRouter();
  const { data, today, applyHealthSnapshot } = useApp();
  const [syncing, setSyncing] = useState(false);
  const [message, setMessage] = useState('');
  /** What the last sync attempt actually reported. '' until one has run. */
  const [outcome, setOutcome] = useState<SyncOutcome>('');
  const [range, setRange] = useState<Range>('30');
  const provider = Platform.OS === 'ios' ? 'Apple Health + Watch' : 'Health Connect';
  const setup = healthSetupCopy();
  const days = Number(range);
  const rangeLabel = `Last ${range} days`;

  const week = useMemo(() => Array.from({ length: 7 }, (_, index) => {
    const date = new Date();
    date.setDate(date.getDate() - (6 - index));
    const day = data.days[dateKey(date)];
    return {
      label: new Intl.DateTimeFormat(undefined, { weekday: 'narrow' }).format(date),
      calories: day ? dayTotals(day).calories : 0,
      steps: day?.steps ?? 0,
    };
  }), [data.days]);

  /** The same per-day roll-up as `week`, stretched over the selected range. */
  const series = useMemo(() => Array.from({ length: days }, (_, index) => {
    const date = new Date();
    date.setDate(date.getDate() - (days - 1 - index));
    const key = dateKey(date);
    const day = data.days[key];
    return {
      key,
      calories: day ? dayTotals(day).calories : 0,
      steps: day?.steps ?? 0,
    };
  }), [data.days, days]);

  const rangeStart = useMemo(() => {
    const date = new Date();
    date.setDate(date.getDate() - (days - 1));
    return dateKey(date);
  }, [days]);

  const weightPoints = useMemo(
    () => data.weights
      .filter((point) => point.date >= rangeStart)
      .map((point) => ({ date: point.date, value: point.kg })),
    [data.weights, rangeStart],
  );

  const todayKey = dateKey();

  const caloriePoints = useMemo(
    // A day with nothing logged is missing data, not a zero-calorie day. Plotting
    // it as 0 would draw a crash in intake that never happened, so those days are
    // dropped from the line instead.
    //
    // Today goes with them. Before dinner its total is half a day of food, so the
    // last segment dives — and the chart's glowing end marker, the first thing the
    // eye lands on, would park on the least meaningful number on the screen. The
    // line ends on the last complete day; the tiles above still count today.
    () => series
      .filter((day) => day.key !== todayKey && day.calories > 0)
      .map((day) => ({ date: day.key, value: Math.round(day.calories) })),
    [series, todayKey],
  );

  const training = useMemo(() => {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - (days - 1));
    cutoff.setHours(0, 0, 0, 0);
    const since = cutoff.toISOString();
    let sessions = 0;
    let minutes = 0;
    let volumeKg = 0;
    for (const session of completedSessions(data.training)) {
      if (session.startedAt < since) continue;
      sessions += 1;
      minutes += session.durationMin ?? 0;
      volumeKg += session.totalVolumeKg ?? 0;
    }
    return { sessions, minutes, volumeKg: Math.round(volumeKg) };
  }, [data.training, days]);

  async function sync() {
    setSyncing(true);
    setMessage('');
    setOutcome('');
    try {
      const snapshot = await syncNativeHealth();
      applyHealthSnapshot(snapshot);
      const hasSamples = Boolean(
        snapshot.steps
        || snapshot.activeCalories
        || snapshot.sleepHours
        || snapshot.weightKg,
      );
      setOutcome(hasSamples ? 'synced' : 'empty');
      setMessage(
        hasSamples
          ? `Updated from ${snapshot.source}.`
          : `Connected to ${snapshot.source}, but no approved samples were found. Check the Health app’s sharing permissions; use a physical phone for real wearable records.`,
      );
    } catch (error) {
      setOutcome('failed');
      setMessage(error instanceof Error ? error.message : 'Health sync failed.');
    } finally {
      setSyncing(false);
    }
  }

  const currentWeight = data.weights.at(-1)?.kg;
  const mealDays = week.filter((day) => day.calories > 0).length;
  const stepDays = week.filter((day) => day.steps >= data.goals.steps).length;

  const weightDelta = weightPoints.length > 1
    ? weightPoints[weightPoints.length - 1].value - weightPoints[0].value
    : null;
  const loggedDays = series.filter((day) => day.calories > 0);
  const avgCalories = loggedDays.length
    ? loggedDays.reduce((sum, day) => sum + day.calories, 0) / loggedDays.length
    : 0;
  const adherence = data.goals.calories > 0 ? avgCalories / data.goals.calories : 0;
  const streak = loggingStreak(data);
  const hydration = Math.round((today.waterMl / data.goals.waterMl) * 100);
  const weightPrompt = data.weights.length === 0
    ? 'Say “my weight is 82.4 kg” to start your trend.'
    : weightPoints.length === 1
      ? 'One weigh-in in this window. Log another and the line appears.'
      : 'No weigh-ins in this window yet. Step on the scale to restart the line.';

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          eyebrow={`${loggedDays.length} of ${days} days logged`}
          title="Progress"
        />

        <Reveal index={1}>
          <Segmented onChange={setRange} options={RANGES} value={range} />
        </Reveal>

        {/* ---------- Hero: where the scale sits today ---------- */}
        <Reveal index={2} style={{ marginTop: space.md }}>
          <Card glow raised>
            <View style={styles.heroTop}>
              <View style={{ flex: 1 }}>
                <Text style={styles.heroLabel}>CURRENT WEIGHT</Text>
                <View style={styles.heroValueRow}>
                  {currentWeight === undefined ? (
                    <Text style={styles.heroNumber}>—</Text>
                  ) : (
                    <CountUp decimals={1} style={styles.heroNumber} value={currentWeight} />
                  )}
                  <Text style={styles.heroUnit}>kg</Text>
                </View>
              </View>
              <Tap
                accessibilityLabel="Log weight"
                onPress={() => router.push({ pathname: '/quick-log', params: { prefill: 'My weight is ' } })}
                scaleTo={0.94}
                style={styles.action}>
                <Glyph color={palette.lime} name="scale" size={13} />
                <Text style={styles.actionText}>Log weight</Text>
              </Tap>
            </View>

            {weightDelta === null ? null : (
              <View style={styles.delta}>
                <Glyph
                  color={palette.inkMid}
                  name={weightDelta < 0 ? 'arrowDown' : 'arrowUp'}
                  size={12}
                />
                <CountUp
                  decimals={1}
                  prefix={weightDelta > 0 ? '+' : ''}
                  style={styles.deltaValue}
                  suffix=" kg"
                  value={weightDelta}
                />
                <Text style={styles.deltaCaption}>over {rangeLabel.toLowerCase()}</Text>
              </View>
            )}

            <TrendChart emptyBody={weightPrompt} height={152} points={weightPoints} unit=" kg" />
          </Card>
        </Reveal>

        {/* ---------- Range headlines ---------- */}
        <Reveal index={3} style={styles.tiles}>
          <StatTile
            decimals={0}
            detail={`${loggedDays.length} logged ${loggedDays.length === 1 ? 'day' : 'days'}`}
            icon="bowl"
            label="Avg intake"
            unit="kcal"
            value={avgCalories}
          />
          <StatTile
            accent={palette.info}
            detail={`of ${days} days`}
            icon="calendar"
            label="Logged"
            unit=""
            value={loggedDays.length}
          />
          <StatTile
            accent={palette.fat}
            detail={streak === 0 ? 'log today to start' : streak === 1 ? 'day in a row' : 'days in a row'}
            icon="flame"
            label="Streak"
            unit=""
            value={streak}
          />
        </Reveal>

        {/* ---------- Calorie adherence ---------- */}
        <SectionTitle aside={rangeLabel} title="Calorie adherence" />
        <Reveal index={4}>
          <Card>
            <Text style={styles.cardLabel}>AVERAGE AGAINST GOAL</Text>
            <View style={styles.cardValueRow}>
              <CountUp style={styles.cardNumber} suffix="%" value={Math.round(adherence * 100)} />
              <Text style={styles.cardCaption}>
                of {data.goals.calories.toLocaleString()} kcal a day
              </Text>
            </View>
            <Bar delay={220} style={styles.cardBar} value={adherence} />
            <TrendChart
              emptyBody="Log a couple of days of meals and your intake line appears here."
              height={148}
              points={caloriePoints}
              reference={data.goals.calories}
              referenceLabel="Goal"
              unit=" kcal"
            />
            {caloriePoints.length > 1 ? (
              <Text style={styles.chartNote}>
                The line ends on your last complete day — today is still being logged.
              </Text>
            ) : null}
          </Card>
        </Reveal>

        {/* ---------- Training volume ---------- */}
        <SectionTitle aside={rangeLabel} title="Training volume" />
        <Reveal index={5}>
          <Card padded={false} style={styles.rowCard}>
            <StatRow
              detail="Finished workouts"
              icon="dumbbell"
              label="Sessions"
              value={training.sessions}
            />
            <StatRow
              accent={palette.info}
              detail="Time under tension"
              icon="timer"
              label="Minutes"
              unit=" min"
              value={training.minutes}
            />
            <StatRow
              accent={palette.fat}
              detail="Weight × reps, all working sets"
              icon="muscle"
              label="Volume lifted"
              last
              unit=" kg"
              value={training.volumeKg}
            />
          </Card>
        </Reveal>

        {/* ---------- Weekly consistency ---------- */}
        <SectionTitle aside="Last 7 days" title="This week" />
        <Reveal index={6}>
          <Card>
            <MeterRow label="Food logged" progress={mealDays / 7} suffix="/7" value={mealDays} />
            <MeterRow label="Step goal" progress={stepDays / 7} suffix="/7" value={stepDays} />
            <MeterRow
              label="Hydration today"
              last
              progress={today.waterMl / data.goals.waterMl}
              suffix="%"
              value={hydration}
            />
          </Card>
        </Reveal>

        {/* ---------- Connected health ---------- */}
        <SectionTitle title="Connected health" />
        <Reveal index={7}>
          <Card>
            <View style={styles.providerRow}>
              <View style={styles.providerIcon}>
                <Glyph color={palette.lime} name={Platform.OS === 'ios' ? 'heart' : 'watch'} size={20} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.providerName}>{provider}</Text>
                <Text style={styles.providerMeta}>
                  {connectionState(outcome, data.lastHealthSync)} · {HEALTH_CATEGORIES}
                </Text>
              </View>
              <Tap
                accessibilityLabel="Sync health data"
                disabled={syncing}
                haptic="medium"
                onPress={() => void sync()}
                scaleTo={0.94}
                style={styles.action}>
                <Glyph color={palette.lime} name="cloud" size={13} />
                <Text style={styles.actionText}>{syncing ? 'Syncing…' : 'Sync'}</Text>
              </Tap>
            </View>

            {message ? (
              <Well style={styles.message}>
                {/* A failure told in the accent colour reads as a success. */}
                <Glyph color={outcome === 'failed' ? palette.danger : palette.lime} name="info" size={14} />
                <Text style={styles.messageText}>{message}</Text>
              </Well>
            ) : null}

            <Well style={styles.setup}>
              <Text style={styles.setupTitle}>{setup.title}</Text>
              <Text style={styles.setupBody}>{setup.detail}</Text>
            </Well>
          </Card>
        </Reveal>

        <Reveal index={8}>
          <Text style={styles.privacy}>
            Health access is requested by the operating system. Vigorly reads only the categories you approve.
          </Text>
        </Reveal>
      </ScrollView>
    </Screen>
  );
}

/* ------------------------------------------------------------------ *
 * Pieces
 * ------------------------------------------------------------------ */

function StatTile({
  icon,
  label,
  value,
  unit,
  detail,
  decimals = 0,
  accent = palette.lime,
}: {
  icon: GlyphName;
  label: string;
  value: number;
  unit: string;
  detail: string;
  decimals?: number;
  accent?: string;
}) {
  return (
    <Well style={styles.tile}>
      <View style={[styles.tileIcon, { backgroundColor: `${accent}1A` }]}>
        <Glyph color={accent} name={icon} size={14} />
      </View>
      <Text style={styles.tileLabel}>{label.toUpperCase()}</Text>
      <View style={styles.tileValueRow}>
        <CountUp decimals={decimals} style={styles.tileValue} value={value} />
        {unit ? <Text style={styles.tileUnit}>{unit}</Text> : null}
      </View>
      <Text numberOfLines={1} style={styles.tileDetail}>{detail}</Text>
    </Well>
  );
}

function StatRow({
  icon,
  label,
  detail,
  value,
  unit = '',
  accent = palette.lime,
  last,
}: {
  icon: GlyphName;
  label: string;
  detail: string;
  value: number;
  unit?: string;
  accent?: string;
  last?: boolean;
}) {
  return (
    <View style={[styles.statRow, !last && styles.statRowBorder]}>
      <View style={[styles.statRowIcon, { backgroundColor: `${accent}14` }]}>
        <Glyph color={accent} name={icon} size={16} />
      </View>
      <View style={{ flex: 1 }}>
        <Text numberOfLines={1} style={styles.statRowTitle}>{label}</Text>
        <Text numberOfLines={1} style={styles.statRowDetail}>{detail}</Text>
      </View>
      <CountUp style={[styles.statRowValue, { color: accent }]} suffix={unit} value={value} />
    </View>
  );
}

function MeterRow({
  label,
  value,
  suffix,
  progress,
  last,
}: {
  label: string;
  value: number;
  suffix: string;
  progress: number;
  last?: boolean;
}) {
  return (
    <View style={[styles.meterRow, !last && styles.meterRowGap]}>
      <View style={styles.meterHead}>
        <Text style={styles.meterLabel}>{label}</Text>
        <CountUp style={styles.meterValue} suffix={suffix} value={value} />
      </View>
      <Bar delay={180} value={progress} />
    </View>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  heroTop: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm },
  heroLabel: { ...text.label, color: palette.lime, marginBottom: space.xs },
  heroValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: space.xs },
  heroNumber: { ...text.hero, color: palette.ink, ...tabular },
  heroUnit: { ...text.caption, fontSize: 12, color: palette.inkMid },

  action: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.xs,
    height: 34,
    paddingHorizontal: 12,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: `${palette.lime}33`,
    backgroundColor: palette.limeSoft,
  },
  actionText: { ...text.value, fontSize: 11.5, color: palette.lime },

  delta: { flexDirection: 'row', alignItems: 'center', gap: space.xs, marginTop: space.sm },
  deltaValue: { ...text.value, fontSize: 12.5, color: palette.ink, ...tabular },
  deltaCaption: { ...text.caption, fontSize: 11, color: palette.inkLow },

  tiles: { flexDirection: 'row', gap: space.sm, marginTop: space.md },
  tile: { flex: 1, minHeight: 112, padding: 11 },
  tileIcon: {
    width: 28,
    height: 28,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: space.sm,
  },
  tileLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1, color: palette.inkLow },
  tileValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: space.xs, marginTop: space.xs },
  tileValue: { ...text.headline, fontSize: 18, color: palette.ink, ...tabular },
  tileUnit: { ...text.caption, fontSize: 9.5, color: palette.inkLow },
  tileDetail: { ...text.caption, fontSize: 10, color: palette.inkLow, marginTop: space.xs },

  cardLabel: { ...text.label, color: palette.inkLow },
  cardValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: space.sm, marginTop: space.xs },
  cardNumber: { ...text.headline, color: palette.ink, ...tabular },
  cardCaption: { ...text.caption, fontSize: 11.5, color: palette.inkMid, flex: 1 },
  cardBar: { marginTop: space.sm },
  chartNote: { ...text.caption, fontSize: 10.5, color: palette.inkLow, marginTop: space.xs },

  rowCard: { paddingHorizontal: 14 },
  statRow: { minHeight: 62, flexDirection: 'row', alignItems: 'center', gap: space.sm, paddingVertical: 10 },
  statRowBorder: { borderBottomWidth: 1, borderBottomColor: palette.line },
  statRowIcon: { width: 34, height: 34, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  statRowTitle: { ...text.row, color: palette.ink },
  statRowDetail: { ...text.caption, color: palette.inkLow, marginTop: space.xs },
  statRowValue: { ...text.value, ...tabular, textAlign: 'right' },

  meterRow: { gap: space.sm },
  meterRowGap: { marginBottom: space.md },
  meterHead: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between' },
  meterLabel: { ...text.row, fontSize: 13.5, color: palette.ink },
  meterValue: { ...text.value, color: palette.lime, ...tabular, textAlign: 'right' },

  providerRow: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  providerIcon: {
    width: 40,
    height: 40,
    borderRadius: 14,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  providerName: { ...text.row, color: palette.ink },
  providerMeta: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: space.xs },

  message: { flexDirection: 'row', gap: space.sm, alignItems: 'flex-start', marginTop: space.md },
  messageText: { ...text.caption, color: palette.inkMid, flex: 1 },

  setup: { marginTop: space.sm },
  setupTitle: { ...text.value, fontSize: 12, color: palette.ink },
  setupBody: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: space.xs },

  privacy: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: space.md },
});
