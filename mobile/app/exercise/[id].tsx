import { LinearGradient } from 'expo-linear-gradient';
import { Stack, useLocalSearchParams, useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import { Linking, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ExerciseFigure } from '@/src/components/exercise-figure';
import { Glyph } from '@/src/components/glyph';
import { PhotoDemo } from '@/src/components/photo-demo';
import { TrendChart } from '@/src/components/trend-chart';
import {
  Card,
  Chip,
  EmptyState,
  ListRow,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  Segmented,
  Tap,
  Well,
} from '@/src/components/ui';
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
import { alpha, gradient, palette, radius, shadow, space, tabular, text } from '@/src/theme';

type Tab = 'about' | 'history' | 'charts' | 'records';

const TABS: { value: Tab; label: string }[] = [
  { value: 'about', label: 'How to' },
  { value: 'history', label: 'History' },
  { value: 'charts', label: 'Charts' },
  { value: 'records', label: 'Records' },
];

const METRIC_OPTIONS: { key: TrendMetric; label: string; unit: string }[] = [
  { key: 'heaviest', label: 'Heaviest weight', unit: ' kg' },
  { key: 'oneRm', label: 'Est. 1RM', unit: ' kg' },
  { key: 'setVolume', label: 'Best set volume', unit: ' kg' },
  { key: 'sessionVolume', label: 'Session volume', unit: ' kg' },
  { key: 'reps', label: 'Most reps', unit: '' },
  { key: 'duration', label: 'Longest hold', unit: ' s' },
];

/** Tall enough for a 3:2 demo photo to breathe under the status bar. */
const HERO_HEIGHT = 326;

/** 'full body' → 'Full Body'. The taxonomy data is lower-case; the chips are not. */
function titleCase(value: string) {
  return value.replace(/(^|\s)\S/g, (character) => character.toUpperCase());
}

export default function ExerciseDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const insets = useSafeAreaInsets();
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

  /** The headline record for this exercise kind, plus the supporting rows. */
  const headlineRecord = isDuration
    ? { label: 'Longest hold', value: records.bestDurationSec ? formatDuration(records.bestDurationSec) : '—' }
    : isRepsOnly
      ? { label: 'Most reps in a set', value: records.bestReps ? `${records.bestReps} reps` : '—' }
      : { label: 'Heaviest weight', value: records.heaviestKg ? `${records.heaviestKg} kg` : '—' };

  const recordRows: { label: string; value: string }[] = [];
  if (!isDuration && !isRepsOnly) {
    recordRows.push({
      label: 'Best est. 1RM (Epley)',
      value: records.best1Rm ? `${Math.round(records.best1Rm * 10) / 10} kg` : '—',
    });
    recordRows.push({
      label: 'Best set volume',
      value: records.bestSetVolume ? `${Math.round(records.bestSetVolume)} kg` : '—',
    });
    recordRows.push({
      label: 'Best session volume',
      value: records.bestSessionVolume ? `${Math.round(records.bestSessionVolume)} kg` : '—',
    });
  }
  recordRows.push({ label: 'Sessions logged', value: String(records.totalSessions) });

  return (
    <Screen edges={[]}>
      <Stack.Screen options={{ title: info.name }} />

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {/* ---------- Hero: real photos when available, stylized figure otherwise ---------- */}
        <Reveal from={0} style={styles.hero}>
          {photos ? (
            <PhotoDemo exerciseId={id ?? ''} height={HERO_HEIGHT} style={styles.heroMedia} />
          ) : (
            <View style={styles.figureStage}>
              <View pointerEvents="none" style={styles.figureGlow} />
              <ExerciseFigure
                accent={palette.lime}
                gear={info.gear}
                size={232}
                template={info.template}
                tint={palette.ink}
              />
            </View>
          )}

          <LinearGradient
            colors={[alpha.scrim, gradient.fade[0]]}
            pointerEvents="none"
            style={[styles.heroTopScrim, { height: insets.top + 62 }]}
          />

          <LinearGradient
            colors={[gradient.fade[0], alpha.scrim, palette.bg]}
            locations={[0, 0.62, 1]}
            pointerEvents="none"
            style={styles.heroScrim}
          />

          <View pointerEvents="none" style={[styles.heroCopy, !photos && styles.heroCopyBare]}>
            <Text numberOfLines={1} style={styles.heroMeta}>
              {`${info.primaryMuscle} · ${info.equipment}`.toUpperCase()}
            </Text>
            <Text numberOfLines={2} style={styles.heroTitle}>{info.name}</Text>
          </View>
        </Reveal>

        <View style={styles.body}>
          {/* ---------- Tag strip ---------- */}
          <Reveal index={1}>
            <View style={styles.pills}>
              <Pill icon="muscle" label={titleCase(info.primaryMuscle)} tone="accent" />
              {info.secondaryMuscles.map((muscle) => <Pill key={muscle} label={titleCase(muscle)} />)}
              <Pill icon="dumbbell" label={titleCase(info.equipment)} />
              <Pill
                icon={isDuration ? 'timer' : 'target'}
                label={titleCase(isDuration ? 'timed' : isRepsOnly ? 'bodyweight reps' : 'weight × reps')}
              />
            </View>
          </Reveal>

          {/* ---------- Video ---------- */}
          {guide?.video ? (
            <Reveal index={2} style={styles.block}>
              <Tap
                accessibilityLabel={`Play ${guide.video.title} on YouTube`}
                accessibilityRole="link"
                haptic="medium"
                onPress={() => void Linking.openURL(guide.video!.url)}
                scaleTo={0.985}>
                <Card style={styles.video}>
                  <View style={[styles.videoPlayWrap, shadow.glow]}>
                    <LinearGradient
                      colors={[...gradient.lime]}
                      end={{ x: 1, y: 1 }}
                      start={{ x: 0, y: 0 }}
                      style={styles.videoPlay}>
                      <Glyph color={palette.onLime} name="play" size={16} />
                    </LinearGradient>
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.videoLabel}>VIDEO GUIDE</Text>
                    <Text numberOfLines={2} style={styles.videoTitle}>{guide.video.title}</Text>
                    <Text numberOfLines={1} style={styles.videoMeta}>
                      {guide.video.channel} · opens YouTube
                    </Text>
                  </View>
                  <Glyph color={palette.inkLow} name="chevron" size={16} />
                </Card>
              </Tap>
            </Reveal>
          ) : null}

          {/* ---------- Tabs ---------- */}
          <Reveal index={3} style={styles.block}>
            <Segmented onChange={setTab} options={TABS} value={tab} />
          </Reveal>

          {/* ---------- How to ---------- */}
          {tab === 'about' ? (
            <Reveal index={4} key="about" style={styles.block}>
              {guide ? (
                <View style={styles.stack}>
                  <Card>
                    <Text style={styles.sectionLabel}>SET UP</Text>
                    {guide.setup.map((step, index) => (
                      <StepRow key={step} index={index + 1} muted text={step} />
                    ))}

                    <Text style={[styles.sectionLabel, styles.sectionLabelSpaced]}>EXECUTION</Text>
                    {guide.execution.map((step, index) => (
                      <StepRow key={step} index={index + 1} text={step} />
                    ))}

                    <View style={styles.breathRow}>
                      <Well style={styles.breathCell}>
                        <Text style={styles.microLabel}>BREATHING</Text>
                        <Text style={styles.breathText}>{guide.breathing}</Text>
                      </Well>
                      {guide.tempo ? (
                        <Well style={styles.breathCell}>
                          <Text style={styles.microLabel}>TEMPO</Text>
                          <Text style={styles.breathText}>{guide.tempo}</Text>
                        </Well>
                      ) : null}
                    </View>
                  </Card>

                  <Card>
                    <Text style={styles.sectionLabel}>COMMON MISTAKES</Text>
                    <View style={styles.mistakes}>
                      {guide.mistakes.map((entry) => {
                        const [mistake, ...rest] = entry.split('—');
                        return (
                          <Well key={entry} style={styles.mistake}>
                            <View style={styles.mistakeHead}>
                              <View style={styles.mistakeMark}>
                                <Glyph color={palette.danger} name="close" size={10} strokeWidth={2.8} />
                              </View>
                              <Text style={styles.mistakeTitle}>{mistake.trim()}</Text>
                            </View>
                            {rest.length ? (
                              <Text style={styles.mistakeFix}>{rest.join('—').trim()}</Text>
                            ) : null}
                          </Well>
                        );
                      })}
                    </View>
                  </Card>

                  {guide.safety ? (
                    <View style={styles.safety}>
                      <View style={styles.safetyHead}>
                        <Glyph color={palette.warn} name="shield" size={13} />
                        <Text style={styles.safetyLabel}>SAFETY</Text>
                      </View>
                      <Text style={styles.safetyText}>{guide.safety}</Text>
                    </View>
                  ) : null}

                  <Tap
                    accessibilityLabel={showPattern ? 'Hide movement path' : 'Show movement path'}
                    onPress={() => setShowPattern((value) => !value)}
                    scaleTo={0.985}>
                    <Well style={styles.pathToggle}>
                      <Glyph color={palette.lime} name="spark" size={14} />
                      <Text style={styles.pathToggleText}>
                        {showPattern ? 'Hide movement path' : 'Show movement path'}
                      </Text>
                      <Glyph
                        color={palette.inkLow}
                        name={showPattern ? 'chevronDown' : 'chevron'}
                        size={14}
                      />
                    </Well>
                  </Tap>

                  {showPattern ? (
                    <Reveal from={8}>
                      <Card style={styles.patternCard}>
                        <View pointerEvents="none" style={styles.patternGlow} />
                        <ExerciseFigure
                          accent={palette.lime}
                          gear={info.gear}
                          size={168}
                          template={info.template}
                          tint={palette.inkMid}
                        />
                        <Text style={styles.patternNote}>
                          Stylized joint path — use the photos and video above for real form.
                        </Text>
                      </Card>
                    </Reveal>
                  ) : null}

                  {photos ? (
                    <Text style={styles.credit}>Demo photos: free-exercise-db (public domain).</Text>
                  ) : null}
                </View>
              ) : builtIn ? (
                <Card>
                  <Text style={styles.sectionLabel}>HOW TO</Text>
                  {builtIn.instructions.map((step, index) => (
                    <StepRow key={step} index={index + 1} text={step} />
                  ))}
                </Card>
              ) : (
                <EmptyState
                  body="This is a custom exercise — instructions are up to you."
                  icon="edit"
                  title="Your movement, your rules"
                />
              )}
            </Reveal>
          ) : null}

          {/* ---------- History ---------- */}
          {tab === 'history' ? (
            <Reveal index={4} key="history" style={styles.block}>
              {history.length ? (
                <View style={styles.stack}>
                  {history.map((session) => {
                    const entries = session.exercises.filter((entry) => entry.exerciseId === id);
                    return (
                      <Card key={session.id}>
                        <View style={styles.historyHead}>
                          <View style={styles.historyIcon}>
                            <Glyph color={palette.lime} name="calendar" size={14} />
                          </View>
                          <View style={{ flex: 1 }}>
                            <Text style={styles.historyDate}>
                              {new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}
                            </Text>
                            <Text numberOfLines={1} style={styles.historyName}>{session.name}</Text>
                          </View>
                        </View>
                        <View style={styles.historySets}>
                          {entries.flatMap((entry) => entry.sets).map((set, index) => (
                            <View key={set.id} style={styles.historySet}>
                              <Text style={styles.historySetIndex}>{index + 1}</Text>
                              <Text style={styles.historySetText}>{formatSet(set, info.kind)}</Text>
                              {set.rpe ? <Text style={styles.historyRpe}>RPE {set.rpe}</Text> : null}
                              {set.prFlags?.length ? <Pill icon="trophy" label="PR" tone="accent" /> : null}
                            </View>
                          ))}
                        </View>
                      </Card>
                    );
                  })}
                </View>
              ) : (
                <EmptyState
                  action={<PrimaryButton icon="book" label="Read the how-to" onPress={() => setTab('about')} />}
                  body="No sessions with this exercise yet. It appears here after your first logged set."
                  icon="calendar"
                  title="The logbook starts with rep one"
                />
              )}
            </Reveal>
          ) : null}

          {/* ---------- Charts ---------- */}
          {tab === 'charts' ? (
            <Reveal index={4} key="charts" style={styles.block}>
              <View style={styles.stack}>
                <View style={styles.metricRow}>
                  {metricOptions.map((option) => (
                    <Chip
                      active={metric === option.key}
                      key={option.key}
                      label={option.label}
                      onPress={() => setMetric(option.key)}
                    />
                  ))}
                </View>
                <Card>
                  <TrendChart
                    points={trend}
                    unit={metricOptions.find((option) => option.key === metric)?.unit ?? ''}
                  />
                </Card>
              </View>
            </Reveal>
          ) : null}

          {/* ---------- Records ---------- */}
          {tab === 'records' ? (
            <Reveal index={4} key="records" style={styles.block}>
              {records.totalSessions ? (
                <View style={styles.stack}>
                  <Card glow raised>
                    <Text style={styles.recordHeroLabel}>{headlineRecord.label.toUpperCase()}</Text>
                    <Text numberOfLines={1} style={styles.recordHeroValue}>{headlineRecord.value}</Text>
                    <Text style={styles.recordHeroHint}>Your best working set to date</Text>
                  </Card>

                  <Card padded={false} style={styles.recordCard}>
                    {recordRows.map((row, index) => (
                      <ListRow
                        key={row.label}
                        last={index === recordRows.length - 1}
                        title={row.label}
                        value={row.value}
                      />
                    ))}
                  </Card>

                  {records.heaviestKg && records.best1Rm ? (
                    <View style={styles.note}>
                      <Glyph color={palette.inkLow} name="info" size={13} />
                      <Text style={styles.noteText}>
                        Est. 1RM uses the Epley formula (weight × (1 + reps ÷ 30)) — an estimate, not a test.
                      </Text>
                    </View>
                  ) : null}
                </View>
              ) : (
                <EmptyState
                  action={<PrimaryButton icon="book" label="Read the how-to" onPress={() => setTab('about')} />}
                  body="Records appear after your first completed working set."
                  icon="trophy"
                  title="No personal bests yet"
                />
              )}
            </Reveal>
          ) : null}
        </View>
      </ScrollView>

      {/* ---------- Floating back control ---------- */}
      <View pointerEvents="box-none" style={[styles.backFloat, { top: insets.top + 6 }]}>
        <Tap accessibilityLabel="Go back" haptic="light" onPress={() => router.back()} scaleTo={0.9}>
          <View style={styles.backChip}>
            <Glyph color={palette.ink} name="chevronLeft" size={18} />
          </View>
        </Tap>
      </View>
    </Screen>
  );
}

/** A numbered instruction. `muted` marks set-up steps so execution reads louder. */
function StepRow({ index, text: copy, muted }: { index: number; text: string; muted?: boolean }) {
  return (
    <View style={styles.step}>
      <View style={[styles.stepIndex, muted && styles.stepIndexMuted]}>
        <Text style={[styles.stepIndexText, muted && styles.stepIndexTextMuted]}>{index}</Text>
      </View>
      <Text style={styles.stepText}>{copy}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  content: { paddingBottom: space.tabClearance },
  body: { paddingHorizontal: space.md },
  block: { marginTop: space.md },
  stack: { gap: 10 },

  /* ---------- hero ---------- */
  hero: { height: HERO_HEIGHT, backgroundColor: palette.surface },
  heroMedia: { height: HERO_HEIGHT, borderRadius: 0, borderWidth: 0, backgroundColor: palette.surface },
  figureStage: {
    height: HERO_HEIGHT,
    backgroundColor: palette.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  figureGlow: {
    position: 'absolute',
    top: 26,
    width: 260,
    height: 260,
    borderRadius: 130,
    backgroundColor: alpha.limeFaint,
  },
  heroTopScrim: { position: 'absolute', left: 0, right: 0, top: 0 },
  heroScrim: { position: 'absolute', left: 0, right: 0, bottom: 0, height: HERO_HEIGHT * 0.66 },
  heroCopy: { position: 'absolute', left: space.md, right: space.md, bottom: 54 },
  heroCopyBare: { bottom: space.md },
  heroMeta: {
    ...text.label,
    color: palette.lime,
    marginBottom: 7,
  },
  heroTitle: {
    ...text.title,
    color: palette.ink,
    textShadowColor: alpha.scrim,
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 14,
  },

  backFloat: { position: 'absolute', left: space.md },
  backChip: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: alpha.scrim,
    borderWidth: 1,
    borderColor: alpha.white12,
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* ---------- tag strip ---------- */
  pills: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: space.md },

  /* ---------- video ---------- */
  video: { flexDirection: 'row', alignItems: 'center', gap: 13 },
  videoPlayWrap: { borderRadius: 22 },
  videoPlay: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    paddingLeft: 2,
  },
  videoLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1.2, color: palette.lime, marginBottom: 4 },
  videoTitle: { ...text.row, color: palette.ink },
  videoMeta: { ...text.caption, fontSize: 10.5, color: palette.inkLow, marginTop: 3 },

  /* ---------- how to ---------- */
  sectionLabel: { ...text.label, color: palette.lime, marginBottom: 12 },
  sectionLabelSpaced: { marginTop: 10 },
  microLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1.1, color: palette.inkLow, marginBottom: 5 },

  step: { flexDirection: 'row', gap: 11, marginBottom: 11 },
  stepIndex: {
    width: 23,
    height: 23,
    borderRadius: 9,
    backgroundColor: palette.lime,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  stepIndexMuted: { backgroundColor: palette.limeSoft },
  stepIndexText: { ...text.value, fontSize: 11, color: palette.onLime, ...tabular },
  stepIndexTextMuted: { color: palette.lime },
  stepText: { flex: 1, ...text.body, color: palette.inkMid },

  breathRow: { flexDirection: 'row', gap: 8, marginTop: 5 },
  breathCell: { flex: 1 },
  breathText: { ...text.caption, color: palette.ink },

  mistakes: { gap: 8 },
  mistake: { gap: 5 },
  mistakeHead: { flexDirection: 'row', alignItems: 'flex-start', gap: 9 },
  mistakeMark: {
    width: 18,
    height: 18,
    borderRadius: 6,
    backgroundColor: `${palette.danger}1A`,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  mistakeTitle: { flex: 1, ...text.value, color: palette.danger, lineHeight: 18 },
  mistakeFix: { ...text.caption, color: palette.inkMid, paddingLeft: 27 },

  safety: {
    backgroundColor: `${palette.warn}12`,
    borderWidth: 1,
    borderColor: `${palette.warn}33`,
    borderRadius: radius.lg,
    padding: 15,
  },
  safetyHead: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 7 },
  safetyLabel: { ...text.label, color: palette.warn },
  safetyText: { ...text.body, color: palette.ink },

  pathToggle: { flexDirection: 'row', alignItems: 'center', gap: 9, paddingVertical: 14 },
  pathToggleText: { flex: 1, ...text.value, color: palette.ink },
  patternCard: { alignItems: 'center', overflow: 'hidden' },
  patternGlow: {
    position: 'absolute',
    top: -60,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: alpha.limeFaint,
  },
  patternNote: {
    ...text.caption,
    fontSize: 10.5,
    color: palette.inkLow,
    textAlign: 'center',
    paddingHorizontal: 18,
    marginTop: 4,
  },
  credit: { ...text.caption, fontSize: 10, color: palette.inkLow, textAlign: 'center', marginTop: 4 },

  /* ---------- history ---------- */
  historyHead: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  historyIcon: {
    width: 32,
    height: 32,
    borderRadius: 11,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  historyDate: { ...text.row, color: palette.ink },
  historyName: { ...text.caption, color: palette.inkLow, marginTop: 2 },
  historySets: { marginTop: 12, gap: 2 },
  historySet: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 6 },
  historySetIndex: { width: 18, ...text.label, fontSize: 10, letterSpacing: 0, color: palette.inkLow, ...tabular },
  historySetText: { flex: 1, ...text.value, color: palette.ink, ...tabular },
  historyRpe: { ...text.caption, fontSize: 10.5, color: palette.inkLow, ...tabular },

  /* ---------- charts ---------- */
  metricRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },

  /* ---------- records ---------- */
  recordHeroLabel: { ...text.label, color: palette.lime },
  recordHeroValue: { ...text.hero, color: palette.ink, marginTop: 8, ...tabular },
  recordHeroHint: { ...text.caption, color: palette.inkMid, marginTop: 6 },
  recordCard: { paddingHorizontal: 15 },
  note: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, paddingHorizontal: 4, marginTop: 2 },
  noteText: { flex: 1, ...text.caption, fontSize: 10.5, color: palette.inkLow },
});
