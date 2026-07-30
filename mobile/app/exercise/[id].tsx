import { Stack, useLocalSearchParams } from 'expo-router';
import { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ExerciseFigure } from '@/src/components/exercise-figure';
import { TrendChart } from '@/src/components/trend-chart';
import { findExercise } from '@/src/lib/exercises';
import {
  exerciseInfo,
  exerciseRecords,
  exerciseTrend,
  formatDuration,
  formatSet,
  sessionsWithExercise,
  type TrendMetric,
} from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';

type Tab = 'about' | 'history' | 'charts' | 'records';

const METRIC_OPTIONS: { key: TrendMetric; label: string; unit: string }[] = [
  { key: 'heaviest', label: 'Heaviest weight', unit: ' kg' },
  { key: 'oneRm', label: 'Est. 1RM', unit: ' kg' },
  { key: 'setVolume', label: 'Best set volume', unit: ' kg' },
  { key: 'sessionVolume', label: 'Session volume', unit: ' kg' },
  { key: 'reps', label: 'Most reps', unit: '' },
  { key: 'duration', label: 'Longest hold', unit: ' s' },
];

export default function ExerciseDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { data } = useApp();
  const training = data.training;
  const [tab, setTab] = useState<Tab>('about');

  const info = exerciseInfo(training, id ?? '');
  const builtIn = findExercise(id ?? '');
  const records = useMemo(() => exerciseRecords(training, id ?? ''), [training, id]);
  const history = useMemo(() => sessionsWithExercise(training, id ?? ''), [training, id]);
  const isDuration = info.kind === 'duration';
  const isRepsOnly = info.kind === 'reps-only';
  const defaultMetric: TrendMetric = isDuration ? 'duration' : isRepsOnly ? 'reps' : 'heaviest';
  const [metric, setMetric] = useState<TrendMetric>(defaultMetric);
  const trend = useMemo(() => exerciseTrend(training, id ?? '', metric), [training, id, metric]);
  const metricOptions = METRIC_OPTIONS.filter((option) => {
    if (isDuration) return option.key === 'duration';
    if (isRepsOnly) return option.key === 'reps';
    return option.key !== 'duration' && option.key !== 'reps';
  });

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <Stack.Screen options={{ title: info.name }} />
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.stage}>
          <ExerciseFigure template={info.template} gear={info.gear} size={210} />
          <View style={styles.badges}>
            <Badge label={info.primaryMuscle} strong />
            {info.secondaryMuscles.map((muscle) => <Badge key={muscle} label={muscle} />)}
            <Badge label={info.equipment} />
            <Badge label={isDuration ? 'timed' : isRepsOnly ? 'bodyweight reps' : 'weight × reps'} />
          </View>
        </View>

        <View style={styles.tabs}>
          {(['about', 'history', 'charts', 'records'] as Tab[]).map((candidate) => (
            <Pressable key={candidate} onPress={() => setTab(candidate)} style={[styles.tab, tab === candidate && styles.tabActive]}>
              <Text style={[styles.tabText, tab === candidate && styles.tabTextActive]}>
                {candidate === 'about' ? 'How to' : candidate[0].toUpperCase() + candidate.slice(1)}
              </Text>
            </Pressable>
          ))}
        </View>

        {tab === 'about' ? (
          <View style={styles.card}>
            {builtIn ? (
              <>
                <Text style={styles.sectionLabel}>STEP BY STEP</Text>
                {builtIn.instructions.map((step, index) => (
                  <View key={step} style={styles.step}>
                    <View style={styles.stepIndex}><Text style={styles.stepIndexText}>{index + 1}</Text></View>
                    <Text style={styles.stepText}>{step}</Text>
                  </View>
                ))}
                {builtIn.tips.length ? (
                  <>
                    <Text style={[styles.sectionLabel, { marginTop: 14 }]}>FORM TIPS</Text>
                    {builtIn.tips.map((tip) => (
                      <View key={tip} style={styles.tipRow}>
                        <Text style={styles.tipBullet}>•</Text>
                        <Text style={styles.tipText}>{tip}</Text>
                      </View>
                    ))}
                  </>
                ) : null}
              </>
            ) : (
              <Text style={styles.emptyText}>This is a custom exercise — instructions are up to you.</Text>
            )}
          </View>
        ) : null}

        {tab === 'history' ? (
          history.length ? (
            <View style={{ gap: 9 }}>
              {history.map((session) => {
                const entries = session.exercises.filter((entry) => entry.exerciseId === id);
                return (
                  <View key={session.id} style={styles.card}>
                    <Text style={styles.historyDate}>
                      {new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}
                      {' · '}{session.name}
                    </Text>
                    {entries.flatMap((entry) => entry.sets).map((set, index) => (
                      <View key={set.id} style={styles.historySet}>
                        <Text style={styles.historySetIndex}>{index + 1}</Text>
                        <Text style={styles.historySetText}>{formatSet(set, info.kind)}</Text>
                        {set.rpe ? <Text style={styles.historyRpe}>RPE {set.rpe}</Text> : null}
                        {set.prFlags?.length ? <Text style={styles.historyPr}>PR</Text> : null}
                      </View>
                    ))}
                  </View>
                );
              })}
            </View>
          ) : (
            <EmptyCard text="No sessions with this exercise yet. It appears here after your first logged set." />
          )
        ) : null}

        {tab === 'charts' ? (
          <View style={styles.card}>
            <View style={styles.metricRow}>
              {metricOptions.map((option) => (
                <Pressable
                  key={option.key}
                  onPress={() => setMetric(option.key)}
                  style={[styles.metricChip, metric === option.key && styles.metricChipActive]}>
                  <Text style={[styles.metricChipText, metric === option.key && styles.metricChipTextActive]}>
                    {option.label}
                  </Text>
                </Pressable>
              ))}
            </View>
            <TrendChart points={trend} unit={metricOptions.find((option) => option.key === metric)?.unit ?? ''} />
          </View>
        ) : null}

        {tab === 'records' ? (
          <View style={styles.card}>
            {records.totalSessions ? (
              <>
                {!isDuration && !isRepsOnly ? (
                  <>
                    <RecordRow label="Heaviest weight" value={records.heaviestKg ? `${records.heaviestKg} kg` : '—'} />
                    <RecordRow label="Best est. 1RM (Epley)" value={records.best1Rm ? `${Math.round(records.best1Rm * 10) / 10} kg` : '—'} />
                    <RecordRow label="Best set volume" value={records.bestSetVolume ? `${Math.round(records.bestSetVolume)} kg` : '—'} />
                    <RecordRow label="Best session volume" value={records.bestSessionVolume ? `${Math.round(records.bestSessionVolume)} kg` : '—'} />
                  </>
                ) : null}
                {isRepsOnly ? <RecordRow label="Most reps in a set" value={records.bestReps ? `${records.bestReps} reps` : '—'} /> : null}
                {isDuration ? <RecordRow label="Longest hold" value={records.bestDurationSec ? formatDuration(records.bestDurationSec) : '—'} /> : null}
                <RecordRow label="Sessions logged" value={String(records.totalSessions)} />
                {records.heaviestKg && records.best1Rm ? (
                  <Text style={styles.recordNote}>
                    Est. 1RM uses the Epley formula (weight × (1 + reps ÷ 30)) — an estimate, not a test.
                  </Text>
                ) : null}
              </>
            ) : (
              <Text style={styles.emptyText}>Records appear after your first completed working set.</Text>
            )}
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function Badge({ label, strong }: { label: string; strong?: boolean }) {
  return (
    <View style={[styles.badge, strong && styles.badgeStrong]}>
      <Text style={[styles.badgeText, strong && styles.badgeTextStrong]}>{label}</Text>
    </View>
  );
}

function RecordRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.recordRow}>
      <Text style={styles.recordLabel}>{label}</Text>
      <Text style={styles.recordValue}>{value}</Text>
    </View>
  );
}

function EmptyCard({ text }: { text: string }) {
  return (
    <View style={styles.card}>
      <Text style={styles.emptyText}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { padding: space.md, paddingBottom: 30 },
  stage: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.lg, alignItems: 'center', paddingVertical: 14 },
  badges: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 6, paddingHorizontal: 14, marginTop: 4 },
  badge: { backgroundColor: palette.canvas, borderWidth: 1, borderColor: palette.line, borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 4 },
  badgeStrong: { backgroundColor: palette.forest, borderColor: palette.forest },
  badgeText: { color: palette.ink, fontFamily: type.medium, fontSize: 10, textTransform: 'capitalize' },
  badgeTextStrong: { color: palette.lime, fontFamily: type.demi },
  tabs: { flexDirection: 'row', backgroundColor: '#E8ECE3', borderRadius: radius.pill, padding: 4, marginVertical: 12 },
  tab: { flex: 1, height: 34, borderRadius: radius.pill, alignItems: 'center', justifyContent: 'center' },
  tabActive: { backgroundColor: palette.paper },
  tabText: { color: palette.muted, fontFamily: type.medium, fontSize: 11 },
  tabTextActive: { color: palette.ink, fontFamily: type.demi },
  card: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 15 },
  sectionLabel: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2, marginBottom: 9 },
  step: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  stepIndex: { width: 22, height: 22, borderRadius: 8, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center', marginTop: 1 },
  stepIndexText: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10.5 },
  stepText: { flex: 1, color: palette.ink, fontFamily: type.regular, fontSize: 12.5, lineHeight: 19 },
  tipRow: { flexDirection: 'row', gap: 8, marginBottom: 6 },
  tipBullet: { color: palette.limeDark, fontFamily: type.demi, fontSize: 12 },
  tipText: { flex: 1, color: palette.muted, fontFamily: type.regular, fontSize: 11.5, lineHeight: 17 },
  historyDate: { color: palette.ink, fontFamily: type.demi, fontSize: 12, marginBottom: 8 },
  historySet: { flexDirection: 'row', alignItems: 'center', gap: 9, paddingVertical: 4 },
  historySetIndex: { width: 18, color: palette.muted, fontFamily: type.demi, fontSize: 10.5 },
  historySetText: { flex: 1, color: palette.ink, fontFamily: type.medium, fontSize: 12 },
  historyRpe: { color: palette.muted, fontFamily: type.medium, fontSize: 10 },
  historyPr: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10 },
  metricRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 12 },
  metricChip: { height: 30, paddingHorizontal: 11, borderRadius: radius.pill, borderWidth: 1, borderColor: palette.line, backgroundColor: palette.canvas, alignItems: 'center', justifyContent: 'center' },
  metricChipActive: { backgroundColor: palette.forest, borderColor: palette.forest },
  metricChipText: { color: palette.ink, fontFamily: type.medium, fontSize: 10.5 },
  metricChipTextActive: { color: palette.lime, fontFamily: type.demi },
  recordRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 9, borderBottomWidth: 1, borderBottomColor: palette.line },
  recordLabel: { color: palette.muted, fontFamily: type.medium, fontSize: 12 },
  recordValue: { color: palette.ink, fontFamily: type.demi, fontSize: 13 },
  recordNote: { color: palette.muted, fontFamily: type.regular, fontSize: 10, lineHeight: 15, marginTop: 10 },
  emptyText: { color: palette.muted, fontFamily: type.regular, fontSize: 12, lineHeight: 18 },
});
