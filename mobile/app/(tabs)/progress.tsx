import { type Href, useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import { TrendChart } from '@/src/components/trend-chart';
import {
  Bar,
  Card,
  CountUp,
  Reveal,
  ListRow,
  Ring,
  Screen,
  ScreenHeader,
  SectionTitle,
  Segmented,
  Tap,
  Well,
} from '@/src/components/ui';
import { dateKey } from '@/src/lib/date';
import { healthSnapshotHasSamples, syncNativeHealth } from '@/src/lib/health';
import { buildInsights, type Insight } from '@/src/lib/insights';
import { personalDailyNudge } from '@/src/lib/personalization';
import { dayTotals } from '@/src/lib/stats';
import { loggingStreak } from '@/src/lib/streak';
import { completedSessions } from '@/src/lib/training';
import { weightToTargetCopy } from '@/src/lib/weight';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, tabular, text } from '@/src/theme';
import type { HealthSnapshot } from '@/src/types';

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

/**
 * Amber is the only colour that carries meaning in the insight list — lime is
 * the app's accent and grey is simply quiet, so a warm chip always means "look".
 */
const TONE_COLOR: Record<Insight['tone'], string> = {
  good: palette.lime,
  watch: palette.warn,
  neutral: palette.inkMid,
};

/**
 * What the provider row is allowed to claim. The note panel underneath states
 * the last thing that actually happened, so the header must never outrank it:
 * a card reading "Connected" above a panel reading "connection unavailable"
 * teaches the user to distrust both.
 */
function connectionState(outcome: SyncOutcome, record: ReturnType<typeof useApp>['data']['healthSync']) {
  if (!NATIVE_HEALTH) return 'Not available in this build';
  // The stored record outranks this screen's own last result. `outcome` is
  // local state that survives tab switches, so a sync that failed on the Today
  // tab used to leave this row still reading "Synced just now".
  if (record?.status === 'error') return 'Needs attention';
  if (outcome === 'synced') return 'Synced just now';
  if (outcome === 'empty') return 'No shared samples yet';
  if (outcome === 'failed') return 'Not syncing';
  if (record?.status === 'empty') return 'No shared samples yet';
  return record?.lastSuccessAt ? `Updated ${relativeTime(record.lastSuccessAt)}` : 'Not connected yet';
}

function relativeTime(value: string) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60_000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export default function ProgressScreen() {
  const router = useRouter();
  const { data, today, applyHealthSnapshot, reportHealthSyncError } = useApp();
  const [syncing, setSyncing] = useState(false);
  const [message, setMessage] = useState('');
  /** What the last sync attempt actually reported. '' until one has run. */
  const [outcome, setOutcome] = useState<SyncOutcome>('');
  const [range, setRange] = useState<Range>('30');
  /** Which insight row is expanded, or null when all are collapsed. */
  const [openInsight, setOpenInsight] = useState<string | null>(null);
  const provider = data.healthSync?.source
    ?? (Platform.OS === 'ios' ? 'Apple Health' : 'Health Connect');
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
      const hasSamples = healthSnapshotHasSamples(snapshot);
      setOutcome(hasSamples ? 'synced' : 'empty');
      setMessage(
        hasSamples
          ? `Updated from ${snapshot.source}.`
          : `Connected to ${snapshot.source}, but no approved samples were found. Check the Health app’s sharing permissions; use a physical phone for real wearable records.`,
      );
    } catch (error) {
      setOutcome('failed');
      const nextMessage = error instanceof Error ? error.message : 'Health sync failed.';
      reportHealthSyncError(nextMessage);
      setMessage(nextMessage);
    } finally {
      setSyncing(false);
    }
  }

  const currentWeight = data.weights.at(-1)?.kg;
  const targetWeight = data.profile.targetWeightKg;
  const targetCopy = weightToTargetCopy(currentWeight, targetWeight);
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
    ? 'Log your first weigh-in.'
    : weightPoints.length === 1
      ? 'One weigh-in in this window. Log another and the line appears.'
      : 'No weigh-ins in this window yet. Step on the scale to restart the line.';

  // Moved here from the Coach tab: a reading of the user's own numbers belongs
  // next to the charts those numbers came from.
  const insights = buildInsights(data, today);
  const nudge = personalDailyNudge(data, today);

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
                onPress={() => router.push('/weight-log' as Href)}
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

            <Well style={styles.targetWell}>
              <View style={styles.targetRow}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.targetLabel}>TARGET WEIGHT</Text>
                  <Text style={styles.targetText}>
                    {targetWeight ? `${targetWeight.toFixed(1)} kg${targetCopy ? ` · ${targetCopy}` : ''}` : 'Not set yet'}
                  </Text>
                </View>
                <Tap
                  accessibilityLabel={targetWeight ? 'Edit target weight' : 'Add target weight'}
                  onPress={() => router.push('/settings')}
                  scaleTo={0.95}>
                  <Text style={styles.targetAction}>{targetWeight ? 'Edit' : 'Add target'}</Text>
                </Tap>
              </View>
            </Well>

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
            progress={adherence}
            unit="kcal"
            value={avgCalories}
          />
          <StatTile
            accent={palette.info}
            detail={`of ${days} days`}
            icon="calendar"
            label="Logged"
            progress={loggedDays.length / days}
            unit=""
            value={loggedDays.length}
          />
          <StatTile
            accent={palette.fat}
            detail={streak === 0 ? 'log today to start' : streak === 1 ? 'day in a row' : 'days in a row'}
            icon="flame"
            label="Streak"
            progress={streak / 7}
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
              <Text style={styles.cardCaption}>target {data.goals.calories.toLocaleString()}</Text>
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

        {/* ---------- What the logs say ---------- */}
        <SectionTitle title="What your logs show" />
        <Reveal index={7}>
          <Card>
            <View style={styles.focusHead}>
              <View style={styles.focusIcon}>
                <Glyph color={palette.lime} name="spark" size={15} />
              </View>
              <Text style={styles.focusLabel}>TODAY&apos;S FOCUS</Text>
            </View>
            <Text style={styles.focusTitle}>{nudge.title}</Text>
            <Text style={styles.focusBody}>{nudge.body}</Text>
          </Card>
        </Reveal>

        {insights.length ? (
          <Reveal index={8} style={{ marginTop: space.sm }}>
            <Card padded={false} style={styles.insightCard}>
              {insights.map((insight, index) => {
                const open = openInsight === insight.id;
                return (
                  <View
                    key={insight.id}
                    style={index < insights.length - 1 ? styles.insightRow : undefined}>
                    <ListRow
                      accent={TONE_COLOR[insight.tone]}
                      detail={insight.summary}
                      icon={insight.icon}
                      last
                      onPress={() => setOpenInsight(open ? null : insight.id)}
                      right={
                        <View style={open ? styles.chevronOpen : undefined}>
                          <Glyph color={palette.inkLow} name="chevronDown" size={15} />
                        </View>
                      }
                      title={insight.title}
                    />
                    {open ? (
                      <Reveal from={6}>
                        <Text style={styles.insightBody}>{insight.body}</Text>
                      </Reveal>
                    ) : null}
                  </View>
                );
              })}
            </Card>
          </Reveal>
        ) : (
          <Reveal index={8} style={{ marginTop: space.sm }}>
            <Well style={styles.patternEmpty}>
              <View style={styles.patternDots}>
                <View style={styles.patternDot} />
                <View style={[styles.patternDot, styles.patternDotMid]} />
                <View style={styles.patternDot} />
              </View>
              <Text style={styles.movedText}>Patterns appear after 3 logged days.</Text>
            </Well>
          </Reveal>
        )}

        {/* ---------- The plan ---------- */}
        <SectionTitle title="Your plan" />
        <Reveal index={9}>
          <Card>
            <View style={styles.planRow}>
              <View style={styles.planCell}>
                <Text style={styles.planLabel}>DAILY CALORIES</Text>
                <CountUp style={styles.planValue} value={data.goals.calories} />
              </View>
              <View style={styles.planDivider} />
              <View style={styles.planCell}>
                <Text style={styles.planLabel}>PROTEIN</Text>
                <CountUp style={styles.planValue} suffix=" g" value={data.goals.protein} />
              </View>
              <View style={styles.planDivider} />
              <View style={styles.planCell}>
                <Text style={styles.planLabel}>TRAINING</Text>
                <CountUp
                  style={styles.planValue}
                  suffix=" min"
                  value={data.goals.weeklyWorkoutMinutes}
                />
              </View>
            </View>

            <View style={styles.planRows}>
              <ListRow
                detail="Recalculates from your profile"
                icon="target"
                last
                onPress={() => router.push('/settings' as Href)}
                title="Adjust goals and targets"
              />
            </View>
          </Card>
        </Reveal>

        {/* ---------- Connected health ---------- */}
        <SectionTitle title="Connected health" />
        <Reveal index={10}>
          <Card>
            <View style={styles.providerRow}>
              <View style={styles.providerIcon}>
                <Glyph color={palette.lime} name={Platform.OS === 'ios' ? 'heart' : 'watch'} size={20} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.providerName}>{provider}</Text>
                <Text style={styles.providerMeta}>
                  {connectionState(outcome, data.healthSync)}
                </Text>
              </View>
              <Tap
                accessibilityLabel="Sync health data"
                disabled={syncing || !NATIVE_HEALTH}
                haptic="medium"
                onPress={() => void sync()}
                scaleTo={0.94}
                style={styles.action}>
                <Glyph color={palette.lime} name="cloud" size={13} />
                <Text style={styles.actionText}>
                  {syncing ? 'Syncing…' : NATIVE_HEALTH ? 'Sync' : 'Unavailable'}
                </Text>
              </Tap>
            </View>

            {message || data.healthSync?.message ? (
              <Well style={styles.message}>
                {/* A failure told in the accent colour reads as a success. */}
                <Glyph
                  color={outcome === 'failed' || data.healthSync?.status === 'error' ? palette.danger : palette.lime}
                  name="info"
                  size={14}
                />
                <Text style={styles.messageText}>{message || data.healthSync?.message}</Text>
              </Well>
            ) : null}

            {data.healthSync?.sampleCounts ? (
              <View
                accessible
                accessibilityLabel={healthCoverageCopy(data.healthSync.sampleCounts)}
                style={styles.healthCoverage}>
                <Text style={styles.healthCoverageTitle}>SHARED DATA</Text>
                <Text style={styles.healthCoverageText}>
                  {healthCoverageCopy(data.healthSync.sampleCounts)}
                </Text>
                {data.healthSync.latestSampleAt ? (
                  <Text style={styles.healthCoverageMeta}>
                    Latest source sample {relativeTime(data.healthSync.latestSampleAt)}
                  </Text>
                ) : null}
              </View>
            ) : null}

          </Card>
        </Reveal>
      </ScrollView>
    </Screen>
  );
}

function healthCoverageCopy(counts: NonNullable<HealthSnapshot['sampleCounts']>) {
  const labels: [keyof typeof counts, string][] = [
    ['steps', 'steps'],
    ['activeCalories', 'active calories'],
    ['sleep', 'sleep'],
    ['weight', 'weight'],
  ];
  const shared = labels.filter(([key]) => counts[key] > 0).map(([, label]) => label);
  const empty = labels.filter(([key]) => counts[key] === 0).map(([, label]) => label);
  if (!shared.length) return 'No approved samples were returned for steps, active calories, sleep, or weight.';
  return `${shared.join(', ')} available${empty.length ? ` · no samples returned for ${empty.join(', ')}` : ''}.`;
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
  progress = 0,
}: {
  icon: GlyphName;
  label: string;
  value: number;
  unit: string;
  detail: string;
  decimals?: number;
  accent?: string;
  progress?: number;
}) {
  return (
    <Well accessibilityLabel={`${label}, ${Math.round(value)} ${unit}. ${detail}`} style={styles.tile}>
      <Ring colors={[accent, accent]} size={46} thickness={4} track={`${accent}1F`} value={progress}>
        <View style={[styles.tileIcon, { backgroundColor: `${accent}14` }]}>
          <Glyph color={accent} name={icon} size={14} />
        </View>
      </Ring>
      <Text style={styles.tileLabel}>{label.toUpperCase()}</Text>
      <View style={styles.tileValueRow}>
        <CountUp decimals={decimals} style={styles.tileValue} value={value} />
        {unit ? <Text style={styles.tileUnit}>{unit}</Text> : null}
      </View>
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
    <View
      accessibilityLabel={`${label}, ${value}${unit}. ${detail}`}
      style={[styles.statRow, !last && styles.statRowBorder]}>
      <View style={[styles.statRowIcon, { backgroundColor: `${accent}14` }]}>
        <Glyph color={accent} name={icon} size={16} />
      </View>
      <View style={{ flex: 1 }}>
        <Text numberOfLines={1} style={styles.statRowTitle}>{label}</Text>
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

const insightStyles = {
  /* today's focus */
  focusHead: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: space.sm },
  focusIcon: {
    width: 30,
    height: 30,
    borderRadius: radius.sm,
    backgroundColor: palette.limeSoft,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  },
  focusLabel: { ...text.label, color: palette.lime },
  focusTitle: { ...text.value, color: palette.ink, marginTop: space.sm },
  focusBody: { ...text.caption, color: palette.inkMid, marginTop: space.xs },

  /* insight list */
  insightCard: { paddingHorizontal: 14 },
  insightRow: { borderBottomWidth: 1, borderBottomColor: palette.line },
  insightBody: { ...text.body, color: palette.inkMid, paddingBottom: space.sm },
  chevronOpen: { transform: [{ rotate: '180deg' }] },
  moved: { flexDirection: 'row' as const, alignItems: 'flex-start' as const, gap: space.sm, padding: 14 },
  movedText: { ...text.caption, color: palette.inkMid, flex: 1 },
  patternEmpty: { alignItems: 'center' as const, gap: space.sm, paddingVertical: space.lg },
  patternDots: { flexDirection: 'row' as const, alignItems: 'flex-end' as const, gap: 7, height: 28 },
  patternDot: { width: 8, height: 12, borderRadius: 4, backgroundColor: palette.lineHi },
  patternDotMid: { height: 28, backgroundColor: palette.lime },

  /* plan */
  planRow: { flexDirection: 'row' as const, alignItems: 'center' as const },
  planCell: { flex: 1, alignItems: 'center' as const },
  planDivider: { width: 1, alignSelf: 'stretch' as const, backgroundColor: palette.line },
  planLabel: {
    ...text.label,
    fontSize: 8.5,
    letterSpacing: 1,
    color: palette.inkLow,
    marginBottom: space.xs,
  },
  planValue: { ...text.headline, fontSize: 17, color: palette.ink, textAlign: 'center' as const, ...tabular },
  planRows: { marginTop: space.sm },
};

const styles = StyleSheet.create({
  ...insightStyles,
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
  targetWell: { marginTop: space.md },
  targetRow: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  targetLabel: { ...text.label, fontSize: 8.5, color: palette.inkLow },
  targetText: { ...text.caption, color: palette.inkMid, marginTop: 4 },
  targetAction: { ...text.value, fontSize: 11.5, color: palette.lime, paddingVertical: 8 },

  tiles: { flexDirection: 'row', gap: space.sm, marginTop: space.md },
  tile: { flex: 1, minHeight: 124, padding: 11 },
  tileIcon: {
    width: 32,
    height: 32,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tileLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1, color: palette.inkLow, marginTop: 8 },
  tileValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: space.xs, marginTop: space.xs },
  tileValue: { ...text.headline, fontSize: 18, color: palette.ink, ...tabular },
  tileUnit: { ...text.caption, fontSize: 9.5, color: palette.inkLow },

  cardLabel: { ...text.label, color: palette.inkLow },
  cardValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: space.sm, marginTop: space.xs },
  cardNumber: { ...text.headline, color: palette.ink, ...tabular },
  cardCaption: { ...text.caption, fontSize: 11.5, color: palette.inkMid, flex: 1 },
  cardBar: { marginTop: space.sm },

  rowCard: { paddingHorizontal: 14 },
  statRow: { minHeight: 62, flexDirection: 'row', alignItems: 'center', gap: space.sm, paddingVertical: 10 },
  statRowBorder: { borderBottomWidth: 1, borderBottomColor: palette.line },
  statRowIcon: { width: 34, height: 34, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  statRowTitle: { ...text.row, color: palette.ink },
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
  healthCoverage: { marginTop: space.md, paddingTop: space.sm, borderTopWidth: 1, borderTopColor: palette.line },
  healthCoverageTitle: { ...text.label, color: palette.inkLow },
  healthCoverageText: { ...text.caption, color: palette.ink, marginTop: space.xs },
  healthCoverageMeta: { ...text.micro, color: palette.inkLow, marginTop: space.xs },

  setup: { marginTop: space.sm },
  setupTitle: { ...text.value, fontSize: 12, color: palette.ink },
  setupBody: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: space.xs },

  privacy: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: space.md },
});
