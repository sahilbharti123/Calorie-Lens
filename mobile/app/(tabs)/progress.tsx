import { useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Circle, Line, Polyline } from 'react-native-svg';

import { Glyph } from '@/src/components/glyph';
import { ScreenHeader, SectionTitle } from '@/src/components/ui';
import { dateKey } from '@/src/lib/date';
import { healthSetupCopy, syncNativeHealth } from '@/src/lib/health';
import { dayTotals } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';

export default function ProgressScreen() {
  const router = useRouter();
  const { data, today, applyHealthSnapshot } = useApp();
  const [syncing, setSyncing] = useState(false);
  const [message, setMessage] = useState('');
  const provider = Platform.OS === 'ios' ? 'Apple Health + Watch' : 'Health Connect';
  const setup = healthSetupCopy();

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

  async function sync() {
    setSyncing(true);
    setMessage('');
    try {
      const snapshot = await syncNativeHealth();
      applyHealthSnapshot(snapshot);
      const hasSamples = Boolean(
        snapshot.steps
        || snapshot.activeCalories
        || snapshot.sleepHours
        || snapshot.weightKg,
      );
      setMessage(
        hasSamples
          ? `Updated from ${snapshot.source}.`
          : `Connected to ${snapshot.source}, but no approved samples were found. Check the Health app’s sharing permissions; use a physical phone for real wearable records.`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Health sync failed.');
    } finally {
      setSyncing(false);
    }
  }

  const currentWeight = data.weights.at(-1)?.kg;
  const mealDays = week.filter((day) => day.calories > 0).length;
  const stepDays = week.filter((day) => day.steps >= data.goals.steps).length;

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader eyebrow="Your trend" title="Progress" />

        <View style={styles.weightCard}>
          <View style={styles.weightTop}>
            <View>
              <Text style={styles.cardEyebrow}>CURRENT WEIGHT</Text>
              <Text style={styles.weightValue}>{currentWeight ? currentWeight.toFixed(1) : '—'} <Text style={styles.weightUnit}>kg</Text></Text>
            </View>
            <Pressable
              onPress={() => router.push({ pathname: '/quick-log', params: { prefill: 'My weight is ' } })}
              accessibilityLabel="Log weight"
              style={styles.periodPill}>
              <Text style={styles.periodText}>Log weight</Text>
            </Pressable>
          </View>
          <WeightChart weights={data.weights} />
          {!data.weights.length ? <Text style={styles.chartEmpty}>Say “my weight is 82.4 kg” to start your trend.</Text> : null}
        </View>

        <SectionTitle title="This week" aside="Last 7 days" />
        <View style={styles.consistency}>
          <ConsistencyRow label="Food logged" value={`${mealDays}/7`} progress={mealDays / 7} />
          <ConsistencyRow label="Step goal" value={`${stepDays}/7`} progress={stepDays / 7} />
          <ConsistencyRow
            label="Hydration today"
            value={`${Math.round((today.waterMl / data.goals.waterMl) * 100)}%`}
            progress={today.waterMl / data.goals.waterMl}
          />
        </View>

        <SectionTitle title="Connected health" />
        <View style={styles.healthCard}>
          <View style={styles.providerIcon}>
            <Glyph name={Platform.OS === 'ios' ? 'heart' : 'watch'} color={palette.forest} size={24} />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.providerName}>{provider}</Text>
            <Text style={styles.providerMeta}>
              {data.lastHealthSync ? 'Connected · ' : ''}Steps, activity, sleep and weight
            </Text>
          </View>
          <Pressable disabled={syncing} onPress={sync} style={styles.syncButton}>
            <Text style={styles.syncText}>{syncing ? 'Syncing…' : 'Sync'}</Text>
          </Pressable>
        </View>
        {message ? <Text style={styles.syncMessage}>{message}</Text> : null}
        <View style={styles.setupNote}>
          <Text style={styles.setupTitle}>{setup.title}</Text>
          <Text style={styles.setupBody}>{setup.detail}</Text>
        </View>
        <Text style={styles.privacy}>
          Health access is requested by the operating system. Calorie Lens reads only the categories you approve.
        </Text>

        <View style={styles.weekBars}>
          {week.map((day) => (
            <View key={`${day.label}-${day.calories}`} style={styles.weekBarItem}>
              <View style={styles.weekBarTrack}>
                <View style={[styles.weekBarFill, { height: `${Math.min(100, (day.calories / data.goals.calories) * 100)}%` }]} />
              </View>
              <Text style={styles.weekLabel}>{day.label}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function WeightChart({ weights }: { weights: { date: string; kg: number }[] }) {
  const points = weights.slice(-10);
  if (points.length < 2) {
    return (
      <Svg width="100%" height={92} viewBox="0 0 320 92">
        <Line x1="0" y1="70" x2="320" y2="70" stroke={palette.line} strokeWidth="1" />
      </Svg>
    );
  }
  const values = points.map((point) => point.kg);
  const min = Math.min(...values) - 0.5;
  const max = Math.max(...values) + 0.5;
  const coordinates = points.map((point, index) => {
    const x = (index / (points.length - 1)) * 300 + 10;
    const y = 75 - ((point.kg - min) / (max - min || 1)) * 55;
    return { x, y };
  });
  return (
    <Svg width="100%" height={92} viewBox="0 0 320 92">
      <Line x1="0" y1="75" x2="320" y2="75" stroke={palette.line} strokeWidth="1" />
      <Polyline
        points={coordinates.map(({ x, y }) => `${x},${y}`).join(' ')}
        fill="none"
        stroke={palette.limeDark}
        strokeWidth="3"
        strokeLinejoin="round"
      />
      {coordinates.map(({ x, y }, index) => (
        <Circle key={`${x}-${y}`} cx={x} cy={y} r={index === coordinates.length - 1 ? 4 : 2.5} fill={palette.limeDark} />
      ))}
    </Svg>
  );
}

function ConsistencyRow({ label, value, progress }: { label: string; value: string; progress: number }) {
  return (
    <View style={styles.consistencyRow}>
      <Text style={styles.consistencyLabel}>{label}</Text>
      <View style={styles.consistencyTrack}>
        <View style={[styles.consistencyFill, { width: `${Math.min(100, progress * 100)}%` }]} />
      </View>
      <Text style={styles.consistencyValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { paddingHorizontal: space.md, paddingBottom: 30 },
  weightCard: { backgroundColor: palette.forest, borderRadius: radius.lg, padding: 19, marginBottom: 24 },
  weightTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  cardEyebrow: { color: palette.lime, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.3 },
  weightValue: { color: palette.white, fontFamily: type.demi, fontSize: 34, letterSpacing: -1.4, marginTop: 3 },
  weightUnit: { color: '#AEB9B0', fontFamily: type.medium, fontSize: 12 },
  periodPill: { paddingVertical: 7, paddingHorizontal: 10, borderRadius: radius.pill, backgroundColor: '#2B382F' },
  periodText: { color: '#CBD2CC', fontFamily: type.medium, fontSize: 10 },
  chartEmpty: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 10, marginTop: -7 },
  consistency: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 15, gap: 16, marginBottom: 24 },
  consistencyRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  consistencyLabel: { width: 94, color: palette.ink, fontFamily: type.medium, fontSize: 11 },
  consistencyTrack: { flex: 1, height: 5, backgroundColor: palette.line, borderRadius: 3, overflow: 'hidden' },
  consistencyFill: { height: '100%', backgroundColor: palette.limeDark },
  consistencyValue: { width: 34, color: palette.muted, fontFamily: type.demi, fontSize: 10, textAlign: 'right' },
  healthCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14, flexDirection: 'row', alignItems: 'center', gap: 11 },
  providerIcon: { width: 42, height: 42, borderRadius: 13, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  providerName: { color: palette.ink, fontFamily: type.demi, fontSize: 13 },
  providerMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 14, marginTop: 2 },
  syncButton: { backgroundColor: palette.forest, borderRadius: radius.pill, paddingHorizontal: 14, paddingVertical: 9 },
  syncText: { color: palette.lime, fontFamily: type.demi, fontSize: 10 },
  syncMessage: { color: palette.ink, backgroundColor: palette.softLime, borderRadius: radius.sm, padding: 11, fontFamily: type.medium, fontSize: 10, lineHeight: 15, marginTop: 8 },
  setupNote: { marginTop: 10, paddingHorizontal: 3 },
  setupTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 10.5 },
  setupBody: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 15, marginTop: 3 },
  privacy: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 15, marginTop: 10 },
  weekBars: { height: 120, flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-around', marginTop: 25, paddingHorizontal: 18 },
  weekBarItem: { alignItems: 'center', gap: 6 },
  weekBarTrack: { width: 13, height: 88, borderRadius: 7, backgroundColor: palette.line, overflow: 'hidden', justifyContent: 'flex-end' },
  weekBarFill: { width: '100%', minHeight: 2, backgroundColor: palette.limeDark, borderRadius: 7 },
  weekLabel: { color: palette.muted, fontFamily: type.medium, fontSize: 9 },
});
