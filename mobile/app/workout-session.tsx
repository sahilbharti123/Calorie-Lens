import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withSpring,
  withTiming,
} from 'react-native-reanimated';

import { Glyph } from '@/src/components/glyph';
import {
  Card,
  CountUp,
  EmptyState,
  GhostButton,
  GlassFooter,
  Pill,
  PrimaryButton,
  Reveal,
  Ring,
  Screen,
  Tap,
  Well,
} from '@/src/components/ui';
import { onNextExercisePick } from '@/src/lib/exercise-pick-bus';
import {
  RPE_CHOICES,
  REST_CHOICES,
  SET_TYPE_LABEL,
  exerciseInfo,
  formatDuration,
  formatSet,
  makeSessionExercise,
  newId,
  nextSetType,
  previousPerformance,
  sessionTotals,
} from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';
import { alpha, motion, palette, radius, space, tabular, text } from '@/src/theme';
import type { SessionExercise, WorkoutSet } from '@/src/types';

type ExerciseInfo = ReturnType<typeof exerciseInfo>;

export default function WorkoutSessionScreen() {
  const router = useRouter();
  const { data } = useApp();
  const workouts = useWorkouts();
  const training = data.training;
  const session = training.activeSession;

  const [now, setNow] = useState(Date.now());
  const [rest, setRest] = useState<{ endsAt: number; totalSec: number } | null>(null);
  const [prToast, setPrToast] = useState<string | null>(null);

  // Live elapsed clock — only ticks while a workout is actually running, so the
  // empty state never re-renders the screen once a second.
  const hasSession = Boolean(session);
  useEffect(() => {
    if (!hasSession) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [hasSession]);

  // Rest countdown hit zero: buzz, then clear the dock after a short beat.
  // The dismissal handle belongs to this effect and is cleared on cleanup, and
  // the deps are the rest object plus the expiry flag rather than `now`: a plain
  // clock tick no longer re-runs (so it cannot cancel an armed dismissal), while
  // a +15s tap replaces `rest` and unsets `restExpired`, which cancels the
  // pending dismissal instead of letting it tear down a live timer.
  const restExpired = rest ? now >= rest.endsAt : false;
  useEffect(() => {
    if (!rest || !restExpired) return;
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    const timer = setTimeout(() => setRest(null), 900);
    return () => clearTimeout(timer);
  }, [rest, restExpired]);

  if (!session) {
    return (
      <Screen edges={['top', 'bottom']}>
        <View style={styles.emptyWrap}>
          <EmptyState
            action={<PrimaryButton icon="chevronLeft" label="Go back" onPress={() => router.back()} />}
            body="Start a routine or an empty session from the Train tab and it will show up right here."
            icon="dumbbell"
            title="No workout in progress"
          />
        </View>
      </Screen>
    );
  }

  const elapsedSec = Math.max(0, Math.floor((now - new Date(session.startedAt).getTime()) / 1000));
  const totals = sessionTotals(session);
  const restRemaining = rest ? Math.max(0, Math.ceil((rest.endsAt - now) / 1000)) : 0;

  function patchSet(exerciseEntryId: string, setId: string, patch: Partial<WorkoutSet>) {
    workouts.updateActiveSession((current) => ({
      ...current,
      exercises: current.exercises.map((entry) => entry.id !== exerciseEntryId ? entry : {
        ...entry,
        sets: entry.sets.map((set) => set.id === setId ? { ...set, ...patch } : set),
      }),
    }));
  }

  function toggleComplete(entry: SessionExercise, set: WorkoutSet) {
    if (set.completed) {
      patchSet(entry.id, set.id, { completed: false, prFlags: undefined });
      return;
    }
    const flags = workouts.completeSet(entry.id, set.id);
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (flags.length) {
      setPrToast(`PR! ${flags.join(' · ')}`);
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setTimeout(() => setPrToast(null), 2600);
    }
    if (entry.restSec > 0) {
      setRest({ endsAt: Date.now() + entry.restSec * 1000, totalSec: entry.restSec });
    }
  }

  function addExercisesMidWorkout() {
    onNextExercisePick((exerciseIds) => {
      workouts.updateActiveSession((current) => ({
        ...current,
        exercises: [
          ...current.exercises,
          ...exerciseIds.map((exerciseId) => makeSessionExercise(exerciseId, training.defaultRestSec)),
        ],
      }));
    });
    router.push('/exercise-picker');
  }

  function replaceExercise(entryId: string) {
    onNextExercisePick(([exerciseId]) => {
      if (!exerciseId) return;
      workouts.updateActiveSession((current) => ({
        ...current,
        exercises: current.exercises.map((entry) => entry.id === entryId
          ? { ...makeSessionExercise(exerciseId, entry.restSec), id: entry.id, supersetId: entry.supersetId }
          : entry),
      }));
    });
    router.push('/exercise-picker');
  }

  function removeExercise(entryId: string) {
    workouts.updateActiveSession((current) => ({
      ...current,
      exercises: current.exercises.filter((entry) => entry.id !== entryId),
    }));
  }

  function discard() {
    Alert.alert('Discard workout?', 'All sets from this session will be lost.', [
      { text: 'Keep training', style: 'cancel' },
      {
        text: 'Discard',
        style: 'destructive',
        onPress: () => {
          workouts.discardActiveWorkout();
          router.back();
        },
      },
    ]);
  }

  function finish() {
    const hasIncomplete = session!.exercises.some((entry) => entry.sets.some((set) => !set.completed));
    const complete = () => {
      const finishedId = workouts.finishActiveWorkout();
      if (finishedId) {
        router.replace({ pathname: '/workout/[id]', params: { id: finishedId, celebrate: '1' } });
      } else {
        router.back();
      }
    };
    if (!totals.sets) {
      Alert.alert('Nothing logged yet', 'Complete at least one set, or discard the workout.');
      return;
    }
    if (hasIncomplete) {
      Alert.alert('Finish workout?', 'Unchecked sets will be removed.', [
        { text: 'Keep training', style: 'cancel' },
        { text: 'Finish', onPress: complete },
      ]);
    } else {
      complete();
    }
  }

  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        {/* ---------- Header ---------- */}
        <Reveal>
          <View style={styles.header}>
            <View style={styles.headerCopy}>
              <Text style={styles.headerEyebrow}>IN PROGRESS</Text>
              <TextInput
                accessibilityLabel="Workout name"
                // Store exactly what was typed — including "" — so clearing the
                // field stays cleared. The 'Workout' fallback lives where the
                // name is read: this placeholder, the Train tab live card, and
                // finishActiveWorkout, which is what actually persists it.
                onChangeText={(name) => workouts.updateActiveSession((current) => ({ ...current, name }))}
                placeholder="Workout"
                placeholderTextColor={palette.inkLow}
                selectionColor={palette.lime}
                style={styles.title}
                value={session.name}
              />
            </View>
            <PrimaryButton compact icon="check" label="Finish" onPress={finish} />
          </View>
        </Reveal>

        {prToast ? <PrBanner label={prToast} /> : null}

        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          {/* ---------- Hero ---------- */}
          <Reveal index={1}>
            <Card glow raised>
              <View style={styles.heroTop}>
                <Text style={styles.heroLabel}>ELAPSED</Text>
                <View style={styles.liveRow}>
                  <LivePulse />
                  <Text style={styles.liveLabel}>LIVE</Text>
                </View>
              </View>
              <Text numberOfLines={1} style={styles.heroTime}>{formatDuration(elapsedSec)}</Text>
              <View style={styles.heroStats}>
                <Well style={styles.heroStat}>
                  <Text style={styles.statLabel}>SETS</Text>
                  <CountUp style={styles.statValue} value={totals.sets} />
                </Well>
                <Well style={styles.heroStat}>
                  <Text style={styles.statLabel}>VOLUME</Text>
                  <View style={styles.statValueRow}>
                    <CountUp style={styles.statValue} value={totals.volumeKg} />
                    <Text style={styles.statUnit}>kg</Text>
                  </View>
                </Well>
              </View>
            </Card>
          </Reveal>

          {/* ---------- Exercises ---------- */}
          {session.exercises.map((entry, index) => {
            const info = exerciseInfo(training, entry.exerciseId);
            const previous = previousPerformance(training, entry.exerciseId);
            const linkedWithAbove = index > 0 && Boolean(entry.supersetId)
              && session.exercises[index - 1].supersetId === entry.supersetId;
            const showRpe = training.rpeEnabled && info.kind !== 'duration';
            return (
              <Reveal
                index={Math.min(2 + index, 6)}
                key={entry.id}
                style={[styles.exerciseSlot, linkedWithAbove && styles.exerciseSlotLinked]}>
                {linkedWithAbove ? (
                  <View style={styles.supersetLink}>
                    <View style={styles.supersetLinkLine} />
                    <Text style={styles.supersetLinkText}>SUPERSET</Text>
                    <View style={styles.supersetLinkLine} />
                  </View>
                ) : null}
                <Card padded={false} style={[styles.exerciseCard, linkedWithAbove && styles.exerciseCardLinked]}>
                  <View style={styles.exerciseHead}>
                    <View style={styles.exerciseHeadCopy}>
                      <Tap
                        accessibilityLabel={`How to do ${info.name}`}
                        onPress={() => router.push({ pathname: '/exercise/[id]', params: { id: entry.exerciseId } })}
                        scaleTo={0.99}>
                        <Text numberOfLines={1} style={styles.exerciseName}>{info.name}</Text>
                        <Text numberOfLines={1} style={styles.exerciseMeta}>
                          {info.primaryMuscle}{entry.supersetId ? ' · superset' : ''} · rest {entry.restSec ? `${entry.restSec}s` : 'off'}
                        </Text>
                      </Tap>
                    </View>
                    <Tap
                      accessibilityLabel="Change rest time"
                      hitSlop={6}
                      onPress={() => {
                        const currentIndex = REST_CHOICES.indexOf(entry.restSec);
                        const nextRest = REST_CHOICES[(currentIndex + 1) % REST_CHOICES.length];
                        workouts.updateActiveSession((current) => ({
                          ...current,
                          exercises: current.exercises.map((candidate) => candidate.id === entry.id
                            ? { ...candidate, restSec: nextRest }
                            : candidate),
                        }));
                      }}
                      scaleTo={0.92}
                      style={styles.headChip}>
                      <Glyph color={entry.restSec ? palette.lime : palette.inkLow} name="timer" size={13} />
                      <Text style={styles.headChipText}>{entry.restSec ? `${entry.restSec}s` : 'rest off'}</Text>
                    </Tap>
                    <Tap
                      accessibilityLabel={`Options for ${info.name}`}
                      hitSlop={6}
                      onPress={() => Alert.alert(info.name, undefined, [
                        { text: 'Replace exercise', onPress: () => replaceExercise(entry.id) },
                        { text: 'Remove exercise', style: 'destructive', onPress: () => removeExercise(entry.id) },
                        { text: 'Cancel', style: 'cancel' },
                      ])}
                      scaleTo={0.9}
                      style={styles.headIcon}>
                      <Glyph color={palette.inkMid} name="more" size={16} />
                    </Tap>
                  </View>

                  {entry.note ? (
                    <Well style={styles.note}>
                      <Glyph color={palette.lime} name="info" size={13} />
                      <Text style={styles.noteText}>{entry.note}</Text>
                    </Well>
                  ) : null}

                  <View style={styles.setHeader}>
                    <Text style={[styles.colHead, styles.colSet]}>SET</Text>
                    <Text style={[styles.colHead, styles.colPrev]}>PREVIOUS</Text>
                    {info.kind === 'weight-reps' ? <Text style={[styles.colHead, styles.colField]}>KG</Text> : null}
                    <Text style={[styles.colHead, styles.colField]}>
                      {info.kind === 'duration' ? 'SEC' : 'REPS'}
                    </Text>
                    {showRpe ? <Text style={[styles.colHead, styles.colRpe]}>RPE</Text> : null}
                    <View style={styles.colCheck} />
                  </View>

                  {entry.sets.map((set, setIndex) => (
                    <SetRow
                      fallback={previous[setIndex] ?? previous.at(-1)}
                      index={setIndex}
                      info={info}
                      key={set.id}
                      onCycleType={() => patchSet(entry.id, set.id, { type: nextSetType(set.type) })}
                      onCycleRpe={() => {
                        const currentIndex = set.rpe != null ? RPE_CHOICES.indexOf(set.rpe) : -1;
                        const nextRpe = currentIndex >= RPE_CHOICES.length - 1
                          ? undefined
                          : RPE_CHOICES[currentIndex + 1];
                        patchSet(entry.id, set.id, { rpe: nextRpe });
                      }}
                      onDelete={() => workouts.updateActiveSession((current) => ({
                        ...current,
                        exercises: current.exercises.map((candidate) => candidate.id === entry.id
                          ? { ...candidate, sets: candidate.sets.filter((existing) => existing.id !== set.id) }
                          : candidate),
                      }))}
                      onPatch={(patch) => patchSet(entry.id, set.id, patch)}
                      onToggle={() => toggleComplete(entry, set)}
                      set={set}
                      showRpe={showRpe}
                    />
                  ))}

                  <View style={styles.addSetSlot}>
                    <Tap
                      accessibilityLabel={`Add a set to ${info.name}`}
                      onPress={() => workouts.updateActiveSession((current) => ({
                        ...current,
                        exercises: current.exercises.map((candidate) => candidate.id === entry.id
                          ? {
                              ...candidate,
                              sets: [...candidate.sets, {
                                id: newId('set'),
                                type: 'normal' as const,
                                weightKg: candidate.sets.at(-1)?.weightKg,
                                reps: candidate.sets.at(-1)?.reps,
                                durationSec: candidate.sets.at(-1)?.durationSec,
                                completed: false,
                              }],
                            }
                          : candidate),
                      }))}
                      scaleTo={0.96}
                      style={styles.addSet}>
                      <Glyph color={palette.lime} name="plus" size={14} />
                      <Text style={styles.addSetText}>Add set</Text>
                    </Tap>
                  </View>
                </Card>
              </Reveal>
            );
          })}

          {/* ---------- Add / discard ---------- */}
          {session.exercises.length ? (
            <Reveal index={7} style={styles.addExerciseSlot}>
              <Tap
                accessibilityLabel="Add exercises"
                haptic="medium"
                onPress={addExercisesMidWorkout}
                scaleTo={0.98}
                style={styles.addExercise}>
                <Glyph color={palette.lime} name="plus" size={18} />
                <Text style={styles.addExerciseText}>Add exercises</Text>
              </Tap>
            </Reveal>
          ) : (
            <Reveal index={2} style={styles.addExerciseSlot}>
              <EmptyState
                action={<PrimaryButton icon="plus" label="Add exercises" onPress={addExercisesMidWorkout} />}
                body="Pick your first movement and every set, rep and PR from here on is logged automatically."
                icon="dumbbell"
                title="An empty bar is still a start"
              />
            </Reveal>
          )}

          <Reveal index={8} style={styles.discardSlot}>
            <GhostButton icon="trash" label="Discard workout" onPress={discard} tone="danger" />
          </Reveal>
        </ScrollView>

        {rest ? (
          <RestDock
            onMinus={() => setRest((current) => current ? { ...current, endsAt: current.endsAt - 15_000 } : null)}
            onPlus={() => setRest((current) => current
              ? { ...current, endsAt: current.endsAt + 15_000, totalSec: current.totalSec + 15 }
              : null)}
            onSkip={() => setRest(null)}
            remaining={restRemaining}
            total={rest.totalSec}
          />
        ) : null}
      </KeyboardAvoidingView>
    </Screen>
  );
}

/* ------------------------------------------------------------------ *
 * The set row — the core object of this screen.
 * ------------------------------------------------------------------ */

function SetRow({
  set,
  index,
  info,
  fallback,
  showRpe,
  onCycleType,
  onCycleRpe,
  onDelete,
  onPatch,
  onToggle,
}: {
  set: WorkoutSet;
  index: number;
  info: ExerciseInfo;
  fallback?: WorkoutSet;
  showRpe: boolean;
  onCycleType: () => void;
  onCycleRpe: () => void;
  onDelete: () => void;
  onPatch: (patch: Partial<WorkoutSet>) => void;
  onToggle: () => void;
}) {
  const pop = useSharedValue(0);
  const mounted = useRef(false);

  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true;
      return;
    }
    if (set.completed) {
      pop.value = withSequence(
        withSpring(1, motion.bouncy),
        withTiming(0, { duration: motion.base, easing: Easing.out(Easing.cubic) }),
      );
    }
  }, [set.completed, pop]);

  const rowAnimated = useAnimatedStyle(() => ({ transform: [{ scale: 1 + pop.value * 0.02 }] }));
  const checkAnimated = useAnimatedStyle(() => ({ transform: [{ scale: 1 + pop.value * 0.16 }] }));

  const special = set.type !== 'normal';

  return (
    <View style={styles.setSlot}>
      <Animated.View style={[styles.setRow, set.completed && styles.setRowDone, rowAnimated]}>
        <Tap
          accessibilityLabel={`Set ${index + 1}. Tap to change set type, hold to remove.`}
          hitSlop={4}
          onLongPress={onDelete}
          onPress={onCycleType}
          scaleTo={0.9}
          style={[styles.setBadge, special && styles.setBadgeSpecial, set.completed && styles.setBadgeDone]}>
          <Text style={[styles.setBadgeText, special && styles.setBadgeTextSpecial]}>
            {SET_TYPE_LABEL[set.type] || String(index + 1)}
          </Text>
        </Tap>

        <Text numberOfLines={1} style={styles.prevText}>
          {fallback ? formatSet(fallback, info.kind) : '—'}
        </Text>

        {info.kind === 'weight-reps' ? (
          <TextInput
            // defaultValue + a key that changes on completion keeps
            // decimal typing free ("62." stays visible) while still
            // showing auto-prefilled values after check-off.
            key={`kg-${set.id}-${set.completed ? 'done' : 'open'}`}
            editable={!set.completed}
            keyboardType="decimal-pad"
            placeholder={fallback?.weightKg != null ? String(fallback.weightKg) : '0'}
            placeholderTextColor={palette.inkLow}
            selectionColor={palette.lime}
            style={[styles.setInput, set.completed && styles.setInputDone]}
            defaultValue={set.weightKg != null ? String(set.weightKg) : ''}
            onChangeText={(value) => {
              const weightKg = Number.parseFloat(value.replace(',', '.'));
              onPatch({ weightKg: Number.isFinite(weightKg) ? weightKg : undefined });
            }}
          />
        ) : null}

        <TextInput
          key={`reps-${set.id}-${set.completed ? 'done' : 'open'}`}
          editable={!set.completed}
          keyboardType="number-pad"
          placeholder={info.kind === 'duration'
            ? (fallback?.durationSec != null ? String(fallback.durationSec) : '30')
            : (fallback?.reps != null ? String(fallback.reps) : '0')}
          placeholderTextColor={palette.inkLow}
          selectionColor={palette.lime}
          style={[styles.setInput, set.completed && styles.setInputDone]}
          defaultValue={info.kind === 'duration'
            ? (set.durationSec != null ? String(set.durationSec) : '')
            : (set.reps != null ? String(set.reps) : '')}
          onChangeText={(value) => {
            const numeric = Number.parseInt(value, 10);
            const parsed = Number.isFinite(numeric) ? numeric : undefined;
            onPatch(info.kind === 'duration' ? { durationSec: parsed } : { reps: parsed });
          }}
        />

        {showRpe ? (
          <Tap
            accessibilityLabel={`Rate of perceived exertion${set.rpe != null ? `, currently ${set.rpe}` : ''}`}
            hitSlop={4}
            onPress={onCycleRpe}
            scaleTo={0.9}
            style={[styles.rpeChip, set.rpe != null && styles.rpeChipOn]}>
            <Text style={[styles.rpeText, set.rpe != null && styles.rpeTextOn]}>{set.rpe ?? '—'}</Text>
          </Tap>
        ) : null}

        <Tap
          accessibilityLabel={set.completed ? `Set ${index + 1} done, tap to reopen` : `Complete set ${index + 1}`}
          haptic="none"
          onPress={onToggle}
          scaleTo={0.88}>
          <Animated.View style={[styles.checkButton, set.completed && styles.checkButtonOn, checkAnimated]}>
            <Glyph
              color={set.completed ? palette.onLime : palette.inkLow}
              name="check"
              size={18}
              strokeWidth={set.completed ? 2.6 : 1.9}
            />
          </Animated.View>
        </Tap>
      </Animated.View>

      {set.prFlags?.length ? (
        <View pointerEvents="none" style={styles.prChip}>
          <Pill icon="trophy" label="PR" tone="accent" />
        </View>
      ) : null}
    </View>
  );
}

/* ------------------------------------------------------------------ *
 * Rest timer
 * ------------------------------------------------------------------ */

function RestDock({
  remaining,
  total,
  onMinus,
  onPlus,
  onSkip,
}: {
  remaining: number;
  total: number;
  onMinus: () => void;
  onPlus: () => void;
  onSkip: () => void;
}) {
  const enter = useSharedValue(0);

  useEffect(() => {
    enter.value = withSpring(1, motion.enter);
  }, [enter]);

  const animated = useAnimatedStyle(() => ({
    opacity: enter.value,
    transform: [{ translateY: (1 - enter.value) * 60 }],
  }));

  const ratio = total > 0 ? remaining / total : 0;

  return (
    <Animated.View style={animated}>
      <GlassFooter>
        <View style={styles.restRow}>
          <Ring delay={0} size={86} thickness={8} value={ratio}>
            <Text style={styles.restTime}>{formatDuration(remaining)}</Text>
            <Text style={styles.restRingLabel}>LEFT</Text>
          </Ring>

          <View style={styles.restCopy}>
            <Text style={styles.restLabel}>REST</Text>
            <View style={styles.restButtons}>
              <View style={styles.restButtonSlot}>
                <Tap
                  accessibilityLabel="Take 15 seconds off the rest timer"
                  onPress={onMinus}
                  scaleTo={0.92}
                  style={styles.restButton}>
                  <Text style={styles.restButtonText}>−15s</Text>
                </Tap>
              </View>
              <View style={styles.restButtonSlot}>
                <Tap
                  accessibilityLabel="Add 15 seconds to the rest timer"
                  onPress={onPlus}
                  scaleTo={0.92}
                  style={styles.restButton}>
                  <Text style={styles.restButtonText}>+15s</Text>
                </Tap>
              </View>
              <View style={styles.restButtonSlot}>
                <Tap
                  accessibilityLabel="Skip rest"
                  haptic="medium"
                  onPress={onSkip}
                  scaleTo={0.92}
                  style={[styles.restButton, styles.restButtonSkip]}>
                  <Text style={[styles.restButtonText, styles.restButtonTextSkip]}>Skip</Text>
                </Tap>
              </View>
            </View>
          </View>
        </View>
      </GlassFooter>
    </Animated.View>
  );
}

/* ------------------------------------------------------------------ *
 * Celebration + live state
 * ------------------------------------------------------------------ */

function PrBanner({ label }: { label: string }) {
  const enter = useSharedValue(0);

  useEffect(() => {
    enter.value = withSpring(1, motion.bouncy);
  }, [enter]);

  const animated = useAnimatedStyle(() => ({
    opacity: Math.min(1, enter.value * 1.4),
    transform: [
      { scale: 0.88 + enter.value * 0.12 },
      { translateY: (1 - enter.value) * -12 },
    ],
  }));

  return (
    <Animated.View style={[styles.prBanner, animated]}>
      <View style={styles.prBannerIcon}>
        <Glyph color={palette.lime} name="trophy" size={17} />
      </View>
      <Text numberOfLines={2} style={styles.prBannerText}>{label}</Text>
      <Pill icon="spark" label="New best" tone="accent" />
    </Animated.View>
  );
}

function LivePulse() {
  const pulse = useSharedValue(0);

  useEffect(() => {
    pulse.value = withRepeat(
      withTiming(1, { duration: 1100, easing: Easing.inOut(Easing.quad) }),
      -1,
      true,
    );
  }, [pulse]);

  const animated = useAnimatedStyle(() => ({
    opacity: 0.4 + pulse.value * 0.6,
    transform: [{ scale: 0.8 + pulse.value * 0.35 }],
  }));

  return <Animated.View style={[styles.liveDot, animated]} />;
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  emptyWrap: { flex: 1, justifyContent: 'center', paddingHorizontal: space.md },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: space.md,
    paddingTop: 4,
    paddingBottom: 12,
  },
  headerCopy: { flex: 1 },
  headerEyebrow: { ...text.label, color: palette.lime, marginBottom: 4 },
  title: { ...text.headline, color: palette.ink, padding: 0, marginTop: -1 },

  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  heroTop: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  heroLabel: { ...text.label, color: palette.inkLow },
  liveRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  liveDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: palette.lime },
  liveLabel: { ...text.label, color: palette.lime },
  heroTime: { ...text.hero, color: palette.ink, marginTop: 6, ...tabular },
  heroStats: { flexDirection: 'row', gap: 8, marginTop: 16 },
  heroStat: { flex: 1, paddingVertical: 10 },
  statLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1, color: palette.inkLow },
  statValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 3 },
  statValue: { ...text.headline, fontSize: 19, color: palette.ink, marginTop: 3, ...tabular },
  statUnit: { ...text.caption, fontSize: 10.5, color: palette.inkLow },

  exerciseSlot: { marginTop: 12 },
  exerciseSlotLinked: { marginTop: 4 },
  supersetLink: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingBottom: 6, paddingHorizontal: 4 },
  supersetLinkLine: { flex: 1, height: 1, backgroundColor: `${palette.lime}33` },
  supersetLinkText: { ...text.label, fontSize: 8, color: palette.lime },

  exerciseCard: { paddingHorizontal: 12, paddingTop: 12, paddingBottom: 6 },
  exerciseCardLinked: { borderColor: `${palette.lime}33` },
  exerciseHead: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  exerciseHeadCopy: { flex: 1 },
  exerciseName: { ...text.row, fontSize: 15, color: palette.ink },
  exerciseMeta: { ...text.caption, fontSize: 10.5, color: palette.inkLow, marginTop: 3, textTransform: 'capitalize' },
  headChip: {
    height: 32,
    paddingHorizontal: 10,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceLo,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  headChipText: { ...text.micro, fontFamily: text.value.fontFamily, fontSize: 11, color: palette.ink, ...tabular },
  headIcon: {
    width: 32,
    height: 32,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceLo,
    alignItems: 'center',
    justifyContent: 'center',
  },

  note: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, marginTop: 10, padding: 10 },
  noteText: { ...text.caption, flex: 1, color: palette.inkMid },

  setHeader: { flexDirection: 'row', alignItems: 'center', gap: 5, marginTop: 14, marginBottom: 7, paddingHorizontal: 4 },
  colHead: { ...text.label, fontSize: 8, letterSpacing: 1, color: palette.inkLow },
  colSet: { width: 32, textAlign: 'center' },
  colPrev: { width: 60 },
  colField: { flex: 1, textAlign: 'center' },
  colRpe: { width: 36, textAlign: 'center' },
  colCheck: { width: 42 },

  setSlot: { position: 'relative' },
  setRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginBottom: 8,
    padding: 4,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: `${palette.lime}00`,
  },
  setRowDone: { backgroundColor: alpha.limeFaint, borderColor: `${palette.lime}2E` },

  setBadge: {
    width: 32,
    height: 42,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceLo,
    borderWidth: 1,
    borderColor: palette.line,
    alignItems: 'center',
    justifyContent: 'center',
  },
  setBadgeSpecial: { backgroundColor: palette.limeSoft, borderColor: `${palette.lime}44` },
  setBadgeDone: { borderColor: `${palette.lime}33` },
  setBadgeText: { ...text.value, fontSize: 13, color: palette.inkMid, ...tabular },
  setBadgeTextSpecial: { color: palette.lime },

  prevText: { width: 60, ...text.caption, fontSize: 10, color: palette.inkLow, ...tabular },

  setInput: {
    flex: 1,
    height: 42,
    borderWidth: 1,
    borderColor: palette.line,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceLo,
    color: palette.ink,
    ...text.value,
    fontSize: 14,
    paddingHorizontal: 4,
    textAlign: 'center',
    ...tabular,
  },
  setInputDone: {
    backgroundColor: palette.limeSoft,
    borderColor: `${palette.lime}33`,
    color: palette.lime,
  },

  rpeChip: {
    width: 36,
    height: 42,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surfaceLo,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rpeChipOn: { borderColor: `${palette.lime}44`, backgroundColor: palette.limeSoft },
  rpeText: { ...text.value, fontSize: 11.5, color: palette.inkLow, ...tabular },
  rpeTextOn: { color: palette.lime },

  checkButton: {
    width: 42,
    height: 42,
    borderRadius: radius.sm,
    borderWidth: 1.5,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceLo,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkButtonOn: { backgroundColor: palette.lime, borderColor: palette.lime },

  prChip: { position: 'absolute', right: 44, top: -12, zIndex: 3 },

  addSetSlot: { alignSelf: 'flex-start' },
  addSet: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    minHeight: 40,
    paddingRight: 12,
    paddingLeft: 4,
  },
  addSetText: { ...text.value, fontSize: 12.5, color: palette.lime },

  addExerciseSlot: { marginTop: 14 },
  addExercise: {
    minHeight: 56,
    borderRadius: radius.md,
    borderWidth: 1.5,
    borderColor: `${palette.lime}55`,
    borderStyle: 'dashed',
    backgroundColor: alpha.limeFaint,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  addExerciseText: { ...text.row, fontSize: 14, color: palette.lime },

  discardSlot: { marginTop: 12 },

  prBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginHorizontal: space.md,
    marginBottom: 10,
    paddingHorizontal: 12,
    paddingVertical: 11,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: `${palette.lime}55`,
    backgroundColor: palette.limeSoft,
  },
  prBannerIcon: {
    width: 32,
    height: 32,
    borderRadius: 11,
    backgroundColor: alpha.limeGlowSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  prBannerText: { ...text.value, flex: 1, fontSize: 13, color: palette.ink },

  restRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  restTime: { ...text.headline, fontSize: 17, color: palette.ink, ...tabular },
  restRingLabel: { ...text.label, fontSize: 7.5, letterSpacing: 1.1, color: palette.inkLow, marginTop: 2 },
  restCopy: { flex: 1, gap: 9 },
  restLabel: { ...text.label, color: palette.lime },
  restButtons: { flexDirection: 'row', gap: 7 },
  restButtonSlot: { flex: 1 },
  restButton: {
    minHeight: 42,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceHi,
    alignItems: 'center',
    justifyContent: 'center',
  },
  restButtonSkip: { borderColor: `${palette.lime}55`, backgroundColor: palette.limeSoft },
  restButtonText: { ...text.value, fontSize: 12.5, color: palette.ink, ...tabular },
  restButtonTextSkip: { color: palette.lime },
});
