import { useRouter } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';

import { Glyph } from '@/src/components/glyph';
import { useReducedMotion } from '@/src/lib/accessibility';
import {
  Bar,
  Card,
  Chip,
  CountUp,
  EmptyState,
  GhostButton,
  ListRow,
  Metric,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  ScreenHeader,
  SectionTitle,
  Segmented,
  Tap,
  Well,
} from '@/src/components/ui';
import { TEMPLATE_ROUTINE_SEEDS, findExercise } from '@/src/lib/exercises';
import {
  REST_CHOICES,
  completedSessions,
  exerciseInfo,
  formatDuration,
  sessionTotals,
  weeklyMuscleSets,
  weeklyTrainingSummary,
} from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';
import { palette, radius, space, tabular, text } from '@/src/theme';
import type { Routine } from '@/src/types';

/**
 * Lower bound of the hypertrophy guideline printed under the weekly chart.
 * A muscle below it is the only thing on that chart worth colouring differently.
 */
const WEEKLY_MIN_SETS = 10;

/** Rough per-set working time used only for the routine duration estimate. */
const WORK_SEC_PER_SET = 45;

function sentenceCase(value: string) {
  return value ? `${value[0].toUpperCase()}${value.slice(1)}` : value;
}

export default function TrainScreen() {
  const router = useRouter();
  const { data } = useApp();
  const workouts = useWorkouts();
  const training = data.training;
  const [expandedRoutineId, setExpandedRoutineId] = useState<string | null>(null);
  const [showTemplates, setShowTemplates] = useState(false);

  const history = useMemo(() => completedSessions(training), [training]);
  const week = useMemo(() => weeklyTrainingSummary(training), [training]);
  const muscles = useMemo(() => weeklyMuscleSets(training), [training]);
  const routines = useMemo(
    () => [...training.routines].sort((a, b) => (b.lastPerformedAt ?? b.updatedAt).localeCompare(a.lastPerformedAt ?? a.updatedAt)),
    [training.routines],
  );
  const active = training.activeSession;
  const importedNames = new Set(training.routines.map((routine) => routine.name));
  // `routines` is already ordered by last performed (falling back to last
  // edited), so the head of the list is the session the user is most likely to
  // repeat — that is what the hero offers.
  const suggested: Routine | undefined = routines[0];

  // Live elapsed clock — only ticks while a workout is actually running.
  const isActive = Boolean(active);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!isActive) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [isActive]);

  const elapsedSec = active
    ? Math.max(0, Math.floor((now - new Date(active.startedAt).getTime()) / 1000))
    : 0;
  const liveTotals = active ? sessionTotals(active) : null;

  function startEmpty() {
    if (active) {
      router.push('/workout-session');
      return;
    }
    workouts.startEmptyWorkout();
    router.push('/workout-session');
  }

  function startRoutine(routine: Routine) {
    if (active) {
      Alert.alert(
        'Workout in progress',
        'Finish or discard the current workout before starting another one.',
        [
          { text: 'Open current workout', onPress: () => router.push('/workout-session') },
          { text: 'Cancel', style: 'cancel' },
        ],
      );
      return;
    }
    workouts.startRoutine(routine.id);
    router.push('/workout-session');
  }

  function editRoutine(routineId?: string) {
    workouts.beginRoutineDraft(routineId);
    router.push('/routine-editor');
  }

  function confirmDelete(routine: Routine) {
    Alert.alert('Delete routine', `Delete “${routine.name}”? Workout history is kept.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => workouts.deleteRoutine(routine.id) },
    ]);
  }

  /** Exercise count, set count, estimated duration and the muscles a routine hits. */
  function routineShape(routine: Routine) {
    let sets = 0;
    let seconds = 0;
    const hits: string[] = [];
    for (const entry of routine.exercises) {
      sets += entry.sets.length;
      seconds += entry.sets.length * (entry.restSec + WORK_SEC_PER_SET);
      const muscle = exerciseInfo(training, entry.exerciseId).primaryMuscle;
      if (!hits.includes(muscle)) hits.push(muscle);
    }
    return { sets, minutes: Math.max(1, Math.round(seconds / 60)), hits };
  }

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          eyebrow={
            week.workouts
              ? `${week.workouts} session${week.workouts === 1 ? '' : 's'} this week`
              : 'Ready when you are'
          }
          title="Train"
        />

        {/* ---------- Hero: the training state ---------- */}
        <Reveal index={1}>
          {active ? (
            <Card glow raised>
              <View style={styles.liveTop}>
                <LivePulse />
                <Text style={styles.liveLabel}>WORKOUT IN PROGRESS</Text>
              </View>
              <Text style={styles.liveTime}>{formatDuration(elapsedSec)}</Text>
              <Text numberOfLines={1} style={styles.liveName}>{active.name.trim() || 'Workout'}</Text>
              <Text style={styles.liveMeta}>
                Started {new Date(active.startedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </Text>
              <Well style={styles.liveWell}>
                <HeroStat label="Sets done" value={String(liveTotals?.sets ?? 0)} />
                <View style={styles.heroDivider} />
                <HeroStat label="Volume" value={`${(liveTotals?.volumeKg ?? 0).toLocaleString()} kg`} />
              </Well>
              <PrimaryButton
                icon="play"
                label="Resume workout"
                onPress={() => router.push('/workout-session')}
                style={styles.heroButton}
              />
            </Card>
          ) : (
            <Card glow raised>
              <View style={styles.heroTop}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.heroLabel}>VOLUME THIS WEEK</Text>
                  <CountUp style={styles.heroNumber} value={week.volumeKg} />
                  <Text style={styles.heroSub}>
                    {week.workouts
                      ? `kg · ${week.workouts} session${week.workouts === 1 ? '' : 's'}`
                      : 'kg · start your first session'}
                  </Text>
                </View>
                <View style={styles.heroBadge}>
                  <Glyph color={palette.lime} name="dumbbell" size={22} />
                </View>
              </View>
              <Well style={styles.liveWell}>
                <HeroStat label="Workouts" value={String(week.workouts)} />
                <View style={styles.heroDivider} />
                <HeroStat label="Time" value={week.minutes ? formatDuration(week.minutes * 60) : '0m'} />
              </Well>
              {suggested ? (
                <>
                  <PrimaryButton
                    icon="play"
                    label={`Start ${suggested.name}`}
                    onPress={() => startRoutine(suggested)}
                    style={styles.heroButton}
                  />
                  <GhostButton
                    compact
                    icon="plus"
                    label="Empty workout"
                    onPress={startEmpty}
                    style={styles.heroGhost}
                  />
                </>
              ) : (
                <PrimaryButton
                  icon="play"
                  label="Start empty workout"
                  onPress={startEmpty}
                  style={styles.heroButton}
                />
              )}
            </Card>
          )}
        </Reveal>

        {active ? (
          <Reveal index={2} style={styles.weekRow}>
            <Metric icon="dumbbell" label="Workouts" progress={week.workouts / 3} value={String(week.workouts)} />
            <Metric
              accent={palette.info}
              icon="timer"
              label="Time"
              progress={week.minutes / Math.max(1, data.goals.weeklyWorkoutMinutes)}
              value={week.minutes ? formatDuration(week.minutes * 60) : '0m'}
            />
            <Metric
              accent={palette.fat}
              icon="chart"
              label="Volume"
              value={`${week.volumeKg.toLocaleString()} kg`}
            />
          </Reveal>
        ) : null}

        {/* ---------- Routine library ---------- */}
        <SectionTitle aside={`${routines.length} saved`} title="Routines" />
        <Reveal index={3}>
          {routines.length ? (
            <View style={styles.routineList}>
              {routines.map((routine) => {
                const expanded = expandedRoutineId === routine.id;
                const shape = routineShape(routine);
                return (
                  <Card key={routine.id} style={styles.routineCard}>
                    <View style={styles.routineHead}>
                      <View style={{ flex: 1 }}>
                        <Tap
                          accessibilityLabel={`${routine.name}, routine options`}
                          onPress={() => setExpandedRoutineId(expanded ? null : routine.id)}
                          scaleTo={0.99}>
                          {/*
                            One line of name, one line of meta — so every card
                            heads at the same height and every Start pill lands
                            on the same baseline. The folder line was dropped: it
                            repeated "PUSH · PULL · LEGS" down the list, and is
                            still shown and edited in the routine editor.
                          */}
                          <View>
                            <Text numberOfLines={1} style={styles.routineName}>{routine.name}</Text>
                            <Text numberOfLines={1} style={styles.routineMeta}>
                              {routine.exercises.length
                                ? `${routine.exercises.length} exercises · ${shape.sets} sets · ${shape.minutes}m`
                                : 'Add exercises'}
                            </Text>
                          </View>
                        </Tap>
                      </View>
                      <Tap
                        accessibilityLabel={`Start ${routine.name}`}
                        haptic="medium"
                        onPress={() => startRoutine(routine)}
                        scaleTo={0.94}>
                        <View style={styles.startPill}>
                          <Glyph color={palette.onLime} name="play" size={13} />
                          <Text style={styles.startPillText}>Start</Text>
                        </View>
                      </Tap>
                    </View>

                    {shape.hits.length ? (
                      <View style={styles.muscleTags}>
                        {shape.hits.slice(0, 4).map((muscle) => (
                          <Pill key={muscle} label={sentenceCase(muscle)} />
                        ))}
                        {shape.hits.length > 4 ? <Pill label={`+${shape.hits.length - 4}`} /> : null}
                      </View>
                    ) : null}

                    {expanded ? (
                      <>
                        {routine.exercises.length ? (
                          <Well style={styles.routineWell}>
                            {routine.exercises.map((entry, index) => (
                              <View
                                key={entry.id}
                                style={[styles.routineLine, index === routine.exercises.length - 1 && styles.routineLineLast]}>
                                <Text numberOfLines={1} style={styles.routineLineName}>
                                  {exerciseInfo(training, entry.exerciseId).name}
                                </Text>
                                <Text style={styles.routineLineSets}>
                                  {entry.sets.length} set{entry.sets.length === 1 ? '' : 's'}
                                </Text>
                              </View>
                            ))}
                          </Well>
                        ) : null}
                        <View style={styles.routineActions}>
                          <RoutineAction icon="edit" label="Edit" onPress={() => editRoutine(routine.id)} />
                          <RoutineAction
                            icon="copy"
                            label="Duplicate"
                            onPress={() => workouts.duplicateRoutine(routine.id)}
                          />
                          <RoutineAction
                            destructive
                            icon="trash"
                            label="Delete"
                            onPress={() => confirmDelete(routine)}
                          />
                        </View>
                      </>
                    ) : null}
                  </Card>
                );
              })}
            </View>
          ) : (
            <EmptyState
              action={<PrimaryButton icon="plus" label="Build a routine" onPress={() => editRoutine()} />}
              body="Build your own program or import a template below. Routines remember your targets and last weights."
              icon="dumbbell"
              title="No routines yet"
            />
          )}
        </Reveal>

        <Reveal index={4}>
          {routines.length ? (
            <GhostButton
              compact
              icon="plus"
              label="New routine"
              onPress={() => editRoutine()}
              style={styles.newRoutine}
            />
          ) : null}

          <SectionTitle
            aside={showTemplates ? 'Hide' : 'Explore'}
            onPressAside={() => setShowTemplates((value) => !value)}
            title="Template routines"
          />
          {showTemplates ? (
            <Card padded={false} style={styles.listCard}>
              {TEMPLATE_ROUTINE_SEEDS.map((seed, index) => {
                const added = importedNames.has(seed.name);
                return (
                  <View
                    key={seed.name}
                    style={[
                      styles.templateRow,
                      index < TEMPLATE_ROUTINE_SEEDS.length - 1 && styles.rowBorder,
                    ]}>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.templateFolder}>{seed.folder.toUpperCase()}</Text>
                      <Text style={styles.templateName}>{seed.name}</Text>
                      <Text numberOfLines={1} style={styles.templateMeta}>
                        {seed.items.map((item) => findExercise(item.exerciseId)?.name ?? item.exerciseId).slice(0, 3).join(' · ')}
                        {seed.items.length > 3 ? ` +${seed.items.length - 3}` : ''}
                      </Text>
                    </View>
                    <Tap
                      accessibilityLabel={added ? `${seed.name} already added` : `Add ${seed.name}`}
                      disabled={added}
                      onPress={() => workouts.importTemplateRoutine(seed.name)}
                      scaleTo={0.93}>
                      <View style={[styles.addPill, added && styles.addPillDone]}>
                        <Glyph
                          color={added ? palette.inkLow : palette.lime}
                          name={added ? 'check' : 'plus'}
                          size={13}
                        />
                        <Text style={[styles.addPillText, added && styles.addPillTextDone]}>
                          {added ? 'Added' : 'Add'}
                        </Text>
                      </View>
                    </Tap>
                  </View>
                );
              })}
            </Card>
          ) : (
            <View style={styles.templatePreview}>
              {TEMPLATE_ROUTINE_SEEDS.slice(0, 3).map((seed) => (
                <View key={seed.name} style={styles.templateDot} />
              ))}
              <Text style={styles.templateHint}>Starter programs</Text>
            </View>
          )}
        </Reveal>

        {/* ---------- Weekly sets per muscle ---------- */}
        {muscles.length ? (
          <>
            <SectionTitle aside="per muscle" title="Sets this week" />
            <Reveal index={5}>
              <Card>
                {muscles.map((entry, index) => {
                  // One measure, one colour — amber only where the reading is
                  // under the guideline printed below the chart.
                  const color = entry.sets < WEEKLY_MIN_SETS ? palette.warn : palette.lime;
                  return (
                    <View key={entry.muscle} style={styles.muscleRow}>
                      <View style={styles.muscleHead}>
                        <Text numberOfLines={1} style={styles.muscleName}>{entry.muscle}</Text>
                        <Text style={[styles.muscleSets, { color }]}>{entry.sets}</Text>
                      </View>
                      <Bar
                        color={color}
                        delay={120 + index * 45}
                        height={6}
                        value={Math.min(1, entry.sets / 20)}
                      />
                    </View>
                  );
                })}
                <View style={styles.muscleFoot}>
                  <Glyph color={palette.inkLow} name="info" size={13} />
                  <Text style={styles.muscleHint}>
                    10–20 sets / muscle · amber is under {WEEKLY_MIN_SETS}
                  </Text>
                </View>
              </Card>
            </Reveal>
          </>
        ) : null}

        {/* ---------- Recent workouts ---------- */}
        <SectionTitle aside={history.length ? `${history.length} workouts` : undefined} title="History" />
        <Reveal index={6}>
          {history.length ? (
            <>
              <Card padded={false} style={styles.listCard}>
                {history.slice(0, 3).map((session, index, shown) => (
                  <ListRow
                    accent={session.records ? palette.lime : palette.inkMid}
                    detail={`${new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })} · ${formatDuration((session.durationMin ?? 0) * 60)}`}
                    icon={session.records ? 'trophy' : 'dumbbell'}
                    key={session.id}
                    last={index === shown.length - 1}
                    onPress={() => router.push({ pathname: '/workout/[id]', params: { id: session.id } })}
                    right={
                      session.records ? (
                        <Pill icon="trophy" label={`${session.records} PR`} tone="accent" />
                      ) : undefined
                    }
                    title={session.name}
                    value={`${(session.totalVolumeKg ?? 0).toLocaleString()} kg`}
                  />
                ))}
              </Card>
              {history.length > 3 ? (
                <GhostButton
                  compact
                  icon="calendar"
                  label="See all workouts"
                  onPress={() => router.push('/workout-history')}
                  style={styles.seeAll}
                />
              ) : null}
            </>
          ) : (
            <EmptyState
              action={
                <PrimaryButton
                  icon="play"
                  label={active ? 'Resume workout' : 'Start empty workout'}
                  onPress={startEmpty}
                />
              }
              body="Finish a session to reveal history and records."
              icon="chart"
              title="No workouts logged"
            />
          )}
        </Reveal>

        {/* ---------- Workout settings ---------- */}
        <SectionTitle title="Workout settings" />
        <Reveal index={7}>
          <Card>
            <View style={styles.settingRow}>
              <View style={{ flex: 1 }}>
                <Text style={styles.settingTitle}>Default rest</Text>
                <Text style={styles.settingDetail}>Rest timer preset for new exercises</Text>
              </View>
              <Chip
                active={training.defaultRestSec > 0}
                icon="timer"
                label={training.defaultRestSec ? `${training.defaultRestSec}s` : 'Off'}
                onPress={() => {
                  const index = REST_CHOICES.indexOf(training.defaultRestSec);
                  workouts.setDefaultRest(REST_CHOICES[(index + 1) % REST_CHOICES.length]);
                }}
              />
            </View>
            <View style={[styles.settingRow, styles.settingRowLast]}>
              <View style={{ flex: 1 }}>
                <Text style={styles.settingTitle}>RPE</Text>
                <Text style={styles.settingDetail}>Log an effort rating with each set</Text>
              </View>
              <Segmented
                onChange={(value) => workouts.setRpeEnabled(value === 'on')}
                options={[
                  { value: 'off', label: 'Off' },
                  { value: 'on', label: 'On' },
                ]}
                style={styles.settingSegment}
                value={training.rpeEnabled ? 'on' : 'off'}
              />
            </View>
          </Card>
        </Reveal>
      </ScrollView>
    </Screen>
  );
}

/** Breathing lime dot that marks the live session card. */
function LivePulse() {
  const reducedMotion = useReducedMotion();
  const pulse = useSharedValue(0);

  useEffect(() => {
    if (reducedMotion) {
      pulse.value = 0;
      return;
    }
    pulse.value = withRepeat(
      withTiming(1, { duration: 1100, easing: Easing.inOut(Easing.quad) }),
      -1,
      true,
    );
  }, [pulse, reducedMotion]);

  const halo = useAnimatedStyle(() => ({
    opacity: 0.55 - pulse.value * 0.42,
    transform: [{ scale: 1 + pulse.value * 1.6 }],
  }));

  return (
    <View style={styles.pulse}>
      <Animated.View pointerEvents="none" style={[styles.pulseHalo, halo]} />
      <View style={styles.pulseDot} />
    </View>
  );
}

function HeroStat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.heroStat}>
      <Text style={styles.heroStatLabel}>{label.toUpperCase()}</Text>
      <Text numberOfLines={1} style={styles.heroStatValue}>{value}</Text>
    </View>
  );
}

function RoutineAction({
  label,
  icon,
  destructive,
  onPress,
}: {
  label: string;
  icon: 'edit' | 'copy' | 'trash';
  destructive?: boolean;
  onPress: () => void;
}) {
  const color = destructive ? palette.danger : palette.inkMid;
  return (
    <View style={{ flex: 1 }}>
      <Tap accessibilityLabel={label} onPress={onPress} scaleTo={0.95}>
        <View style={[styles.routineAction, destructive && styles.routineActionDanger]}>
          <Glyph color={color} name={icon} size={13} />
          <Text numberOfLines={1} style={[styles.routineActionText, { color }]}>{label}</Text>
        </View>
      </Tap>
    </View>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  /* hero — live */
  liveTop: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  liveLabel: { ...text.label, color: palette.lime },
  liveTime: { ...text.hero, color: palette.ink, marginTop: space.sm, ...tabular },
  liveName: { ...text.row, fontSize: 15, color: palette.ink, marginTop: space.xs },
  liveMeta: { ...text.caption, color: palette.inkMid, marginTop: space.xs },
  liveWell: { flexDirection: 'row', alignItems: 'center', marginTop: space.md },
  heroButton: { marginTop: space.sm },
  heroGhost: { marginTop: space.xs },

  pulse: { width: 9, height: 9, alignItems: 'center', justifyContent: 'center' },
  pulseHalo: {
    position: 'absolute',
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: palette.lime,
  },
  pulseDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: palette.lime },

  /* hero — idle */
  heroTop: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm },
  heroLabel: { ...text.label, color: palette.lime, marginBottom: space.xs },
  heroNumber: { ...text.hero, color: palette.ink, ...tabular },
  heroSub: { ...text.caption, color: palette.inkMid, marginTop: space.xs },
  heroBadge: {
    width: 44,
    height: 44,
    borderRadius: radius.md,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroStat: { flex: 1 },
  heroStatLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1, color: palette.inkLow },
  heroStatValue: { ...text.value, fontSize: 14, color: palette.ink, marginTop: space.xs, ...tabular },
  heroDivider: {
    width: 1,
    alignSelf: 'stretch',
    backgroundColor: palette.line,
    marginHorizontal: space.sm,
  },

  weekRow: { flexDirection: 'row', gap: space.sm, marginTop: space.sm },

  /* routines */
  routineList: { gap: space.sm },
  routineCard: { padding: 14 },
  routineHead: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  routineName: { ...text.section, color: palette.ink },
  routineMeta: { ...text.caption, fontSize: 11, color: palette.inkMid, marginTop: space.xs, ...tabular },
  startPill: {
    minHeight: 40,
    paddingHorizontal: 15,
    borderRadius: radius.pill,
    backgroundColor: palette.lime,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.xs,
  },
  startPillText: { ...text.value, fontSize: 12.5, color: palette.onLime },
  muscleTags: { flexDirection: 'row', flexWrap: 'wrap', gap: space.xs, marginTop: space.sm },
  routineWell: { marginTop: space.sm, paddingVertical: 4 },
  routineLine: {
    minHeight: 34,
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    borderBottomWidth: 1,
    borderBottomColor: palette.line,
  },
  routineLineLast: { borderBottomWidth: 0 },
  routineLineName: { ...text.caption, fontSize: 12, color: palette.inkMid, flex: 1 },
  routineLineSets: { ...text.value, fontSize: 12, color: palette.ink, ...tabular },
  routineActions: { flexDirection: 'row', gap: space.xs, marginTop: space.sm },
  routineAction: {
    minHeight: 38,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceLo,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.xs,
    paddingHorizontal: 6,
  },
  routineActionDanger: { borderColor: `${palette.danger}33`, backgroundColor: `${palette.danger}10` },
  routineActionText: { ...text.micro, fontSize: 11 },
  newRoutine: { marginTop: space.sm },

  /* templates */
  listCard: { paddingHorizontal: 14 },
  templatePreview: { flexDirection: 'row', alignItems: 'center', gap: 7, minHeight: 34 },
  templateDot: { width: 26, height: 5, borderRadius: 3, backgroundColor: palette.lineHi },
  templateHint: { ...text.caption, color: palette.inkLow, marginLeft: 2 },
  templateRow: {
    minHeight: 66,
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    paddingVertical: 11,
  },
  rowBorder: { borderBottomWidth: 1, borderBottomColor: palette.line },
  templateFolder: { ...text.label, fontSize: 8.5, letterSpacing: 1.1, color: palette.inkLow },
  templateName: { ...text.row, color: palette.ink, marginTop: space.xs },
  templateMeta: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: space.xs },
  addPill: {
    minHeight: 36,
    paddingHorizontal: 13,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: `${palette.lime}3D`,
    backgroundColor: palette.limeSoft,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.xs,
  },
  addPillDone: { borderColor: palette.line, backgroundColor: palette.surfaceLo },
  addPillText: { ...text.value, fontSize: 12, color: palette.lime },
  addPillTextDone: { color: palette.inkLow },

  /* weekly muscle chart */
  muscleRow: { marginBottom: space.sm },
  muscleHead: { flexDirection: 'row', alignItems: 'baseline', gap: space.sm, marginBottom: space.xs },
  muscleName: { ...text.value, fontSize: 12.5, color: palette.ink, textTransform: 'capitalize', flex: 1 },
  muscleSets: { ...text.value, fontSize: 12.5, ...tabular },
  muscleFoot: { flexDirection: 'row', alignItems: 'flex-start', gap: space.xs, marginTop: space.xs },
  muscleHint: { ...text.caption, fontSize: 10.5, color: palette.inkLow, flex: 1 },

  /* history */
  seeAll: { marginTop: space.sm },

  /* settings */
  settingRow: {
    minHeight: 62,
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: palette.line,
  },
  settingRowLast: { borderBottomWidth: 0 },
  settingTitle: { ...text.row, color: palette.ink },
  settingDetail: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: space.xs },
  settingSegment: { width: 118 },
});
