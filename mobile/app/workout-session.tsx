import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
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
import { palette, radius, space, type } from '@/src/theme';
import type { SessionExercise, WorkoutSet } from '@/src/types';

export default function WorkoutSessionScreen() {
  const router = useRouter();
  const { data } = useApp();
  const workouts = useWorkouts();
  const training = data.training;
  const session = training.activeSession;

  const [now, setNow] = useState(Date.now());
  const [rest, setRest] = useState<{ endsAt: number; totalSec: number } | null>(null);
  const [prToast, setPrToast] = useState<string | null>(null);
  const restDoneRef = useRef(false);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!rest) return;
    const remaining = rest.endsAt - now;
    if (remaining <= 0 && !restDoneRef.current) {
      restDoneRef.current = true;
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setTimeout(() => setRest(null), 900);
    }
  }, [now, rest]);

  if (!session) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.emptyWrap}>
          <Text style={styles.emptyTitle}>No workout in progress</Text>
          <Pressable onPress={() => router.back()} style={styles.emptyButton}>
            <Text style={styles.emptyButtonText}>Go back</Text>
          </Pressable>
        </View>
      </SafeAreaView>
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
      restDoneRef.current = false;
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
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        {/* Header */}
        <View style={styles.header}>
          <View style={{ flex: 1 }}>
            <TextInput
              style={styles.title}
              value={session.name}
              onChangeText={(name) => workouts.updateActiveSession((current) => ({ ...current, name: name || 'Workout' }))}
            />
            <Text style={styles.headerMeta}>
              {formatDuration(elapsedSec)} · {totals.sets} sets · {totals.volumeKg.toLocaleString()} kg volume
            </Text>
          </View>
          <Pressable onPress={finish} style={({ pressed }) => [styles.finishButton, pressed && styles.pressed]}>
            <Text style={styles.finishText}>Finish</Text>
          </Pressable>
        </View>

        {prToast ? (
          <View style={styles.prToast}>
            <Glyph name="spark" color={palette.forest} size={16} />
            <Text style={styles.prToastText}>{prToast}</Text>
          </View>
        ) : null}

        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          {session.exercises.map((entry, index) => {
            const info = exerciseInfo(training, entry.exerciseId);
            const previous = previousPerformance(training, entry.exerciseId);
            const linkedWithAbove = index > 0 && Boolean(entry.supersetId)
              && session.exercises[index - 1].supersetId === entry.supersetId;
            return (
              <View key={entry.id} style={[styles.exercise, linkedWithAbove && styles.exerciseLinked]}>
                <View style={styles.exerciseHead}>
                  <Pressable style={{ flex: 1 }} onPress={() => router.push({ pathname: '/exercise/[id]', params: { id: entry.exerciseId } })}>
                    <Text style={styles.exerciseName}>{info.name}</Text>
                    <Text style={styles.exerciseMeta}>
                      {info.primaryMuscle}{entry.supersetId ? ' · superset' : ''} · rest {entry.restSec ? `${entry.restSec}s` : 'off'}
                    </Text>
                  </Pressable>
                  <Pressable
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
                    style={styles.smallChip}>
                    <Text style={styles.smallChipText}>{entry.restSec ? `${entry.restSec}s` : 'rest off'}</Text>
                  </Pressable>
                  <Pressable
                    hitSlop={6}
                    onPress={() => Alert.alert(info.name, undefined, [
                      { text: 'Replace exercise', onPress: () => replaceExercise(entry.id) },
                      { text: 'Remove exercise', style: 'destructive', onPress: () => removeExercise(entry.id) },
                      { text: 'Cancel', style: 'cancel' },
                    ])}
                    style={styles.smallChip}>
                    <Text style={styles.smallChipText}>⋯</Text>
                  </Pressable>
                </View>

                {entry.note ? <Text style={styles.note}>{entry.note}</Text> : null}

                <View style={styles.setHeader}>
                  <Text style={[styles.colHead, styles.colSet]}>SET</Text>
                  <Text style={[styles.colHead, styles.colPrev]}>PREVIOUS</Text>
                  {info.kind === 'weight-reps' ? <Text style={[styles.colHead, styles.colField]}>KG</Text> : null}
                  <Text style={[styles.colHead, styles.colField]}>
                    {info.kind === 'duration' ? 'SEC' : 'REPS'}
                  </Text>
                  {training.rpeEnabled && info.kind !== 'duration' ? (
                    <Text style={[styles.colHead, styles.colRpe]}>RPE</Text>
                  ) : null}
                  <View style={styles.colCheck} />
                </View>

                {entry.sets.map((set, setIndex) => {
                  const fallback = previous[setIndex] ?? previous.at(-1);
                  return (
                    <View key={set.id} style={[styles.setRow, set.completed && styles.setRowDone]}>
                      <Pressable
                        onLongPress={() => workouts.updateActiveSession((current) => ({
                          ...current,
                          exercises: current.exercises.map((candidate) => candidate.id === entry.id
                            ? { ...candidate, sets: candidate.sets.filter((existing) => existing.id !== set.id) }
                            : candidate),
                        }))}
                        onPress={() => patchSet(entry.id, set.id, { type: nextSetType(set.type) })}
                        style={[styles.setBadge, set.type !== 'normal' && styles.setBadgeSpecial]}>
                        <Text style={[styles.setBadgeText, set.type !== 'normal' && styles.setBadgeTextSpecial]}>
                          {SET_TYPE_LABEL[set.type] || String(setIndex + 1)}
                        </Text>
                      </Pressable>
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
                          placeholderTextColor="#A8B0A6"
                          style={[styles.setInput, set.completed && styles.setInputDone]}
                          defaultValue={set.weightKg != null ? String(set.weightKg) : ''}
                          onChangeText={(text) => {
                            const weightKg = Number.parseFloat(text.replace(',', '.'));
                            patchSet(entry.id, set.id, { weightKg: Number.isFinite(weightKg) ? weightKg : undefined });
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
                        placeholderTextColor="#A8B0A6"
                        style={[styles.setInput, set.completed && styles.setInputDone]}
                        defaultValue={info.kind === 'duration'
                          ? (set.durationSec != null ? String(set.durationSec) : '')
                          : (set.reps != null ? String(set.reps) : '')}
                        onChangeText={(text) => {
                          const numeric = Number.parseInt(text, 10);
                          const value = Number.isFinite(numeric) ? numeric : undefined;
                          patchSet(entry.id, set.id, info.kind === 'duration' ? { durationSec: value } : { reps: value });
                        }}
                      />
                      {training.rpeEnabled && info.kind !== 'duration' ? (
                        <Pressable
                          onPress={() => {
                            const currentIndex = set.rpe != null ? RPE_CHOICES.indexOf(set.rpe) : -1;
                            const nextRpe = currentIndex >= RPE_CHOICES.length - 1
                              ? undefined
                              : RPE_CHOICES[currentIndex + 1];
                            patchSet(entry.id, set.id, { rpe: nextRpe });
                          }}
                          style={styles.rpeChip}>
                          <Text style={styles.rpeText}>{set.rpe ?? '—'}</Text>
                        </Pressable>
                      ) : null}
                      <Pressable
                        onPress={() => toggleComplete(entry, set)}
                        style={[styles.checkButton, set.completed && styles.checkButtonOn]}>
                        <Text style={[styles.checkText, set.completed && styles.checkTextOn]}>✓</Text>
                      </Pressable>
                      {set.prFlags?.length ? (
                        <View style={styles.prChip}><Text style={styles.prChipText}>PR</Text></View>
                      ) : null}
                    </View>
                  );
                })}

                <Pressable
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
                  style={styles.addSet}>
                  <Text style={styles.addSetText}>+ Add set</Text>
                </Pressable>
              </View>
            );
          })}

          <Pressable onPress={addExercisesMidWorkout} style={({ pressed }) => [styles.addExercise, pressed && styles.pressed]}>
            <Glyph name="plus" color={palette.forest} size={18} />
            <Text style={styles.addExerciseText}>Add exercises</Text>
          </Pressable>

          <Pressable onPress={discard} style={styles.discard}>
            <Text style={styles.discardText}>Discard workout</Text>
          </Pressable>
        </ScrollView>

        {rest ? (
          <View style={styles.restBar}>
            <View style={styles.restTrack}>
              <View style={[styles.restFill, { width: `${Math.min(100, (restRemaining / rest.totalSec) * 100)}%` }]} />
            </View>
            <View style={styles.restRow}>
              <Text style={styles.restLabel}>REST</Text>
              <Text style={styles.restTime}>{formatDuration(restRemaining)}</Text>
              <Pressable onPress={() => setRest((current) => current ? { ...current, endsAt: current.endsAt - 15_000 } : null)} style={styles.restButton}>
                <Text style={styles.restButtonText}>−15s</Text>
              </Pressable>
              <Pressable onPress={() => setRest((current) => current ? { ...current, endsAt: current.endsAt + 15_000, totalSec: current.totalSec + 15 } : null)} style={styles.restButton}>
                <Text style={styles.restButtonText}>+15s</Text>
              </Pressable>
              <Pressable onPress={() => setRest(null)} style={styles.restButton}>
                <Text style={styles.restButtonText}>Skip</Text>
              </Pressable>
            </View>
          </View>
        ) : null}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  fill: { flex: 1 },
  pressed: { opacity: 0.85 },
  emptyWrap: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 14 },
  emptyTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 16 },
  emptyButton: { height: 44, paddingHorizontal: 22, borderRadius: radius.md, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  emptyButtonText: { color: palette.lime, fontFamily: type.demi, fontSize: 13 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: space.md, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: palette.line, backgroundColor: palette.canvas },
  title: { color: palette.ink, fontFamily: type.demi, fontSize: 19, padding: 0 },
  headerMeta: { color: palette.muted, fontFamily: type.medium, fontSize: 11, marginTop: 3 },
  finishButton: { height: 40, paddingHorizontal: 18, borderRadius: radius.pill, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  finishText: { color: palette.lime, fontFamily: type.demi, fontSize: 13 },
  prToast: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: palette.lime, marginHorizontal: space.md, marginTop: 8, borderRadius: radius.sm, paddingHorizontal: 12, paddingVertical: 9 },
  prToastText: { color: palette.forest, fontFamily: type.demi, fontSize: 12 },
  content: { padding: space.md, paddingBottom: 30, gap: 10 },
  exercise: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 13 },
  exerciseLinked: { borderColor: palette.limeDark, marginTop: -4 },
  exerciseHead: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  exerciseName: { color: palette.ink, fontFamily: type.demi, fontSize: 14.5 },
  exerciseMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 2, textTransform: 'capitalize' },
  smallChip: { minWidth: 34, height: 30, paddingHorizontal: 8, borderRadius: radius.pill, borderWidth: 1, borderColor: palette.line, backgroundColor: palette.canvas, alignItems: 'center', justifyContent: 'center' },
  smallChipText: { color: palette.ink, fontFamily: type.medium, fontSize: 10.5 },
  note: { color: palette.limeDark, fontFamily: type.medium, fontSize: 10.5, lineHeight: 15, marginTop: 7, backgroundColor: palette.softLime, borderRadius: radius.sm, paddingHorizontal: 10, paddingVertical: 7 },
  setHeader: { flexDirection: 'row', alignItems: 'center', gap: 7, marginTop: 12, marginBottom: 5 },
  colHead: { color: palette.muted, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 0.7 },
  colSet: { width: 32 },
  colPrev: { width: 76 },
  colField: { flex: 1 },
  colRpe: { width: 36 },
  colCheck: { width: 34 },
  setRow: { flexDirection: 'row', alignItems: 'center', gap: 7, marginBottom: 7 },
  setRowDone: { opacity: 0.92 },
  setBadge: { width: 32, height: 34, borderRadius: 10, backgroundColor: palette.canvas, borderWidth: 1, borderColor: palette.line, alignItems: 'center', justifyContent: 'center' },
  setBadgeSpecial: { backgroundColor: palette.softLime, borderColor: palette.limeDark },
  setBadgeText: { color: palette.ink, fontFamily: type.demi, fontSize: 11.5 },
  setBadgeTextSpecial: { color: palette.limeDark },
  prevText: { width: 76, color: palette.muted, fontFamily: type.regular, fontSize: 10 },
  setInput: { flex: 1, height: 36, borderWidth: 1, borderColor: palette.line, borderRadius: radius.sm, backgroundColor: palette.canvas, color: palette.ink, fontFamily: type.demi, fontSize: 13, paddingHorizontal: 8, textAlign: 'center' },
  setInputDone: { backgroundColor: '#EAF6D8', borderColor: '#CFE8A8' },
  rpeChip: { width: 36, height: 34, borderRadius: 10, borderWidth: 1, borderColor: palette.line, backgroundColor: palette.canvas, alignItems: 'center', justifyContent: 'center' },
  rpeText: { color: palette.ink, fontFamily: type.demi, fontSize: 10.5 },
  checkButton: { width: 34, height: 34, borderRadius: 11, borderWidth: 1.5, borderColor: palette.line, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.canvas },
  checkButtonOn: { backgroundColor: palette.forest, borderColor: palette.forest },
  checkText: { color: palette.muted, fontFamily: type.demi, fontSize: 14 },
  checkTextOn: { color: palette.lime },
  prChip: { position: 'absolute', right: 40, top: -5, backgroundColor: palette.lime, borderRadius: radius.pill, paddingHorizontal: 6, paddingVertical: 2 },
  prChipText: { color: palette.forest, fontFamily: type.demi, fontSize: 8.5 },
  addSet: { alignSelf: 'flex-start', paddingVertical: 6 },
  addSetText: { color: palette.forest, fontFamily: type.demi, fontSize: 11.5 },
  addExercise: { height: 52, borderRadius: radius.md, borderWidth: 1.5, borderColor: palette.forest, borderStyle: 'dashed', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  addExerciseText: { color: palette.forest, fontFamily: type.demi, fontSize: 13 },
  discard: { alignItems: 'center', paddingVertical: 12 },
  discardText: { color: '#B64B45', fontFamily: type.demi, fontSize: 12 },
  restBar: { borderTopWidth: 1, borderTopColor: palette.line, backgroundColor: palette.forest, paddingBottom: 4 },
  restTrack: { height: 4, backgroundColor: '#2B382F' },
  restFill: { height: '100%', backgroundColor: palette.lime },
  restRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: space.md, paddingVertical: 10 },
  restLabel: { color: '#AEB9B0', fontFamily: type.demi, fontSize: 10, letterSpacing: 1.2 },
  restTime: { flex: 1, color: palette.white, fontFamily: type.demi, fontSize: 17 },
  restButton: { height: 32, paddingHorizontal: 12, borderRadius: radius.pill, backgroundColor: '#2B382F', alignItems: 'center', justifyContent: 'center' },
  restButtonText: { color: palette.lime, fontFamily: type.demi, fontSize: 11 },
});
