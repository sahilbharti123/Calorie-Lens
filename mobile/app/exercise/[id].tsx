import { Stack, useLocalSearchParams } from 'expo-router';
import { useMemo, useState } from 'react';
import { Linking, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ExerciseFigure } from '@/src/components/exercise-figure';
import { Glyph } from '@/src/components/glyph';
import { PhotoDemo } from '@/src/components/photo-demo';
import { TrendChart } from '@/src/components/trend-chart';
import { findExercise } from '@/src/lib/exercises';
import { guideFor } from '@/src/lib/exercise-guides';
import { photosFor } from '@/src/lib/exercise-photos';
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
  const [showPattern, setShowPattern] = useState(false);

  const info = exerciseInfo(training, id ?? '');
  const builtIn = findExercise(id ?? '');
  const guide = guideFor(id ?? '');
  const photos = photosFor(id ?? '');
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
        {/* Demo: real photos when available, stylized figure otherwise */}
        {photos ? (
          <PhotoDemo exerciseId={id ?? ''} />
        ) : (
          <View style={styles.figureStage}>
            <ExerciseFigure template={info.template} gear={info.gear} size={210} />
          </View>
        )}
        <View style={styles.badges}>
          <Badge label={info.primaryMuscle} strong />
          {info.secondaryMuscles.map((muscle) => <Badge key={muscle} label={muscle} />)}
          <Badge label={info.equipment} />
          <Badge label={isDuration ? 'timed' : isRepsOnly ? 'bodyweight reps' : 'weight × reps'} />
        </View>

        {guide?.video ? (
          <Pressable
            accessibilityRole="link"
            onPress={() => void Linking.openURL(guide.video!.url)}
            style={({ pressed }) => [styles.videoCard, pressed && styles.pressed]}>
            <View style={styles.videoPlay}><Text style={styles.videoPlayIcon}>▶</Text></View>
            <View style={{ flex: 1 }}>
              <Text numberOfLines={2} style={styles.videoTitle}>{guide.video.title}</Text>
              <Text style={styles.videoMeta}>{guide.video.channel} · opens YouTube</Text>
            </View>
            <Glyph name="chevron" color={palette.muted} size={16} />
          </Pressable>
        ) : null}

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
          guide ? (
            <View style={{ gap: 10 }}>
              <View style={styles.card}>
                <Text style={styles.sectionLabel}>SET UP</Text>
                {guide.setup.map((step, index) => (
                  <StepRow key={step} index={index + 1} text={step} muted />
                ))}
                <Text style={[styles.sectionLabel, { marginTop: 14 }]}>EXECUTION</Text>
                {guide.execution.map((step, index) => (
                  <StepRow key={step} index={index + 1} text={step} />
                ))}
                <View style={styles.breathRow}>
                  <View style={styles.breathCell}>
                    <Text style={styles.breathLabel}>BREATHING</Text>
                    <Text style={styles.breathText}>{guide.breathing}</Text>
                  </View>
                  {guide.tempo ? (
                    <View style={styles.breathCell}>
                      <Text style={styles.breathLabel}>TEMPO</Text>
                      <Text style={styles.breathText}>{guide.tempo}</Text>
                    </View>
                  ) : null}
                </View>
              </View>

              <View style={styles.card}>
                <Text style={styles.sectionLabel}>COMMON MISTAKES</Text>
                {guide.mistakes.map((entry) => {
                  const [mistake, ...rest] = entry.split('—');
                  return (
                    <View key={entry} style={styles.mistakeRow}>
                      <Text style={styles.mistakeMark}>✕</Text>
                      <View style={{ flex: 1 }}>
                        <Text style={styles.mistakeTitle}>{mistake.trim()}</Text>
                        {rest.length ? <Text style={styles.mistakeFix}>{rest.join('—').trim()}</Text> : null}
                      </View>
                    </View>
                  );
                })}
              </View>

              {guide.safety ? (
                <View style={styles.safety}>
                  <Text style={styles.safetyLabel}>SAFETY</Text>
                  <Text style={styles.safetyText}>{guide.safety}</Text>
                </View>
              ) : null}

              <Pressable onPress={() => setShowPattern((value) => !value)} style={styles.patternToggle}>
                <Text style={styles.patternToggleText}>
                  {showPattern ? 'Hide movement path' : 'Show movement path'}
                </Text>
              </Pressable>
              {showPattern ? (
                <View style={styles.patternCard}>
                  <ExerciseFigure template={info.template} gear={info.gear} size={150} />
                  <Text style={styles.patternNote}>
                    Stylized joint path — use the photos and video above for real form.
                  </Text>
                </View>
              ) : null}

              {photos ? (
                <Text style={styles.credit}>Demo photos: free-exercise-db (public domain).</Text>
              ) : null}
            </View>
          ) : (
            <View style={styles.card}>
              {builtIn ? (
                <>
                  {builtIn.instructions.map((step, index) => (
                    <StepRow key={step} index={index + 1} text={step} />
                  ))}
                </>
              ) : (
                <Text style={styles.emptyText}>This is a custom exercise — instructions are up to you.</Text>
              )}
            </View>
          )
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

function StepRow({ index, text, muted }: { index: number; text: string; muted?: boolean }) {
  return (
    <View style={styles.step}>
      <View style={[styles.stepIndex, muted && styles.stepIndexMuted]}>
        <Text style={[styles.stepIndexText, muted && styles.stepIndexTextMuted]}>{index}</Text>
      </View>
      <Text style={styles.stepText}>{text}</Text>
    </View>
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
  pressed: { opacity: 0.85 },
  figureStage: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.lg, alignItems: 'center', paddingVertical: 14 },
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 10 },
  badge: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 4 },
  badgeStrong: { backgroundColor: palette.forest, borderColor: palette.forest },
  badgeText: { color: palette.ink, fontFamily: type.medium, fontSize: 10, textTransform: 'capitalize' },
  badgeTextStrong: { color: palette.lime, fontFamily: type.demi },
  videoCard: { flexDirection: 'row', alignItems: 'center', gap: 11, backgroundColor: palette.forest, borderRadius: radius.md, padding: 13, marginTop: 10 },
  videoPlay: { width: 40, height: 40, borderRadius: 20, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center' },
  videoPlayIcon: { color: palette.forest, fontSize: 15 },
  videoTitle: { color: palette.white, fontFamily: type.demi, fontSize: 12.5, lineHeight: 17 },
  videoMeta: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 10, marginTop: 3 },
  tabs: { flexDirection: 'row', backgroundColor: '#E8ECE3', borderRadius: radius.pill, padding: 4, marginVertical: 12 },
  tab: { flex: 1, height: 34, borderRadius: radius.pill, alignItems: 'center', justifyContent: 'center' },
  tabActive: { backgroundColor: palette.paper },
  tabText: { color: palette.muted, fontFamily: type.medium, fontSize: 11 },
  tabTextActive: { color: palette.ink, fontFamily: type.demi },
  card: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 15 },
  sectionLabel: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2, marginBottom: 9 },
  step: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  stepIndex: { width: 22, height: 22, borderRadius: 8, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center', marginTop: 1 },
  stepIndexMuted: { backgroundColor: palette.softLime },
  stepIndexText: { color: palette.forest, fontFamily: type.demi, fontSize: 10.5 },
  stepIndexTextMuted: { color: palette.limeDark },
  stepText: { flex: 1, color: palette.ink, fontFamily: type.regular, fontSize: 12.5, lineHeight: 19 },
  breathRow: { flexDirection: 'row', gap: 8, marginTop: 6 },
  breathCell: { flex: 1, backgroundColor: palette.canvas, borderRadius: radius.sm, padding: 11 },
  breathLabel: { color: palette.limeDark, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 1, marginBottom: 4 },
  breathText: { color: palette.ink, fontFamily: type.regular, fontSize: 11, lineHeight: 16 },
  mistakeRow: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  mistakeMark: { color: '#B64B45', fontFamily: type.demi, fontSize: 12, marginTop: 1 },
  mistakeTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 12, lineHeight: 17 },
  mistakeFix: { color: palette.muted, fontFamily: type.regular, fontSize: 11, lineHeight: 16, marginTop: 2 },
  safety: { backgroundColor: palette.softCoral, borderRadius: radius.md, padding: 13 },
  safetyLabel: { color: palette.coral, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2, marginBottom: 4 },
  safetyText: { color: palette.ink, fontFamily: type.regular, fontSize: 11.5, lineHeight: 17 },
  patternToggle: { alignItems: 'center', paddingVertical: 4 },
  patternToggleText: { color: palette.coral, fontFamily: type.demi, fontSize: 11 },
  patternCard: { alignItems: 'center', backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingVertical: 10 },
  patternNote: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, textAlign: 'center', paddingHorizontal: 20, marginTop: 2 },
  credit: { color: palette.muted, fontFamily: type.regular, fontSize: 9, textAlign: 'center' },
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
