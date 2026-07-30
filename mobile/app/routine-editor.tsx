import { useRouter } from 'expo-router';
import { useState } from 'react';
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

import { Glyph, type GlyphName } from '@/src/components/glyph';
import {
  Card,
  GhostButton,
  GlassFooter,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  SectionTitle,
  Tap,
  Well,
} from '@/src/components/ui';
import { onNextExercisePick } from '@/src/lib/exercise-pick-bus';
import {
  REST_CHOICES,
  SET_TYPE_LABEL,
  exerciseInfo,
  makeRoutineExercise,
  makeRoutineSet,
  newId,
  nextSetType,
} from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';
import { alpha, palette, radius, space, tabular, text } from '@/src/theme';
import type { Routine, RoutineExercise } from '@/src/types';

export default function RoutineEditorScreen() {
  const router = useRouter();
  const { data } = useApp();
  const workouts = useWorkouts();
  const [draft, setDraft] = useState<Routine>(() => workouts.draftRef.current ?? workouts.beginRoutineDraft());

  function patchExercise(id: string, recipe: (entry: RoutineExercise) => RoutineExercise) {
    setDraft((current) => ({
      ...current,
      exercises: current.exercises.map((entry) => entry.id === id ? recipe(entry) : entry),
    }));
  }

  function addExercises() {
    onNextExercisePick((exerciseIds) => {
      setDraft((current) => ({
        ...current,
        exercises: [
          ...current.exercises,
          ...exerciseIds.map((exerciseId) => makeRoutineExercise(exerciseId, data.training.defaultRestSec)),
        ],
      }));
    });
    router.push('/exercise-picker');
  }

  function removeExercise(id: string) {
    setDraft((current) => ({
      ...current,
      exercises: current.exercises.filter((entry) => entry.id !== id),
    }));
  }

  function move(id: string, delta: -1 | 1) {
    setDraft((current) => {
      const index = current.exercises.findIndex((entry) => entry.id === id);
      const target = index + delta;
      if (index < 0 || target < 0 || target >= current.exercises.length) return current;
      const exercises = [...current.exercises];
      const [entry] = exercises.splice(index, 1);
      exercises.splice(target, 0, entry);
      return { ...current, exercises };
    });
  }

  function toggleSuperset(id: string) {
    setDraft((current) => {
      const index = current.exercises.findIndex((entry) => entry.id === id);
      if (index <= 0) return current;
      const exercises = [...current.exercises];
      const entry = exercises[index];
      const above = exercises[index - 1];
      if (entry.supersetId && entry.supersetId === above.supersetId) {
        exercises[index] = { ...entry, supersetId: undefined };
      } else {
        const groupId = above.supersetId ?? newId('ss');
        exercises[index - 1] = { ...above, supersetId: groupId };
        exercises[index] = { ...entry, supersetId: groupId };
      }
      return { ...current, exercises };
    });
  }

  function save() {
    if (!draft.exercises.length) {
      Alert.alert('Add exercises', 'A routine needs at least one exercise.');
      return;
    }
    workouts.saveRoutineDraft(draft);
    router.back();
  }

  return (
    <Screen edges={['bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          {/* ---------- Identity ---------- */}
          <Reveal>
            <Card glow raised>
              <Text style={styles.label}>ROUTINE NAME</Text>
              <TextInput
                accessibilityLabel="Routine name"
                onChangeText={(name) => setDraft((current) => ({ ...current, name }))}
                placeholder="e.g. Push Day"
                placeholderTextColor={palette.inkLow}
                selectionColor={palette.lime}
                style={styles.nameInput}
                value={draft.name}
              />
              <View style={styles.folderBlock}>
                <Text style={styles.label}>FOLDER (OPTIONAL)</Text>
                <View style={styles.folderRow}>
                  <Glyph color={palette.inkLow} name="folder" size={15} />
                  <TextInput
                    accessibilityLabel="Routine folder"
                    onChangeText={(folder) => setDraft((current) => ({ ...current, folder: folder || undefined }))}
                    placeholder="e.g. Push · Pull · Legs"
                    placeholderTextColor={palette.inkLow}
                    selectionColor={palette.lime}
                    style={styles.folderInput}
                    value={draft.folder ?? ''}
                  />
                </View>
              </View>
            </Card>
          </Reveal>

          <SectionTitle title="Exercises" />

          {draft.exercises.map((entry, index) => {
            const info = exerciseInfo(data.training, entry.exerciseId);
            const inSuperset = Boolean(entry.supersetId);
            const linkedWithAbove = index > 0 && inSuperset && draft.exercises[index - 1].supersetId === entry.supersetId;
            return (
              <Reveal
                index={Math.min(1 + index, 5)}
                key={entry.id}
                style={[styles.exerciseSlot, linkedWithAbove && styles.exerciseSlotLinked]}>
                <Card padded={false} style={[styles.exerciseCard, linkedWithAbove && styles.exerciseCardLinked]}>
                  <View style={styles.exerciseHead}>
                    <View style={styles.exerciseHeadCopy}>
                      <Tap
                        accessibilityLabel={`How to do ${info.name}`}
                        onPress={() => router.push({ pathname: '/exercise/[id]', params: { id: entry.exerciseId } })}
                        scaleTo={0.99}>
                        <Text numberOfLines={1} style={styles.exerciseName}>{info.name}</Text>
                        <Text numberOfLines={1} style={styles.exerciseMeta}>{info.primaryMuscle} · {info.equipment}</Text>
                      </Tap>
                    </View>
                    <View style={styles.headActions}>
                      <HeadButton
                        icon="arrowUp"
                        label={`Move ${info.name} up`}
                        onPress={() => move(entry.id, -1)}
                      />
                      <HeadButton
                        icon="arrowDown"
                        label={`Move ${info.name} down`}
                        onPress={() => move(entry.id, 1)}
                      />
                      <HeadButton
                        danger
                        icon="close"
                        label={`Remove ${info.name}`}
                        onPress={() => removeExercise(entry.id)}
                      />
                    </View>
                  </View>

                  {inSuperset ? (
                    <View style={styles.supersetTag}>
                      <Pill icon="link" label="Superset" tone="accent" />
                    </View>
                  ) : null}

                  <View style={styles.setHeader}>
                    <Text style={[styles.setHeaderText, styles.colSet]}>SET</Text>
                    {info.kind === 'weight-reps' ? <Text style={[styles.setHeaderText, styles.colInput]}>KG</Text> : null}
                    <Text style={[styles.setHeaderText, styles.colInput]}>
                      {info.kind === 'duration' ? 'SECONDS' : 'REPS (OR RANGE)'}
                    </Text>
                    <View style={styles.colRemove} />
                  </View>

                  {entry.sets.map((set, setIndex) => (
                    <View key={set.id} style={styles.setRow}>
                      <Tap
                        accessibilityLabel={`Set ${setIndex + 1}, change set type`}
                        hitSlop={4}
                        onPress={() => patchExercise(entry.id, (current) => ({
                          ...current,
                          sets: current.sets.map((candidate) => candidate.id === set.id
                            ? { ...candidate, type: nextSetType(candidate.type) }
                            : candidate),
                        }))}
                        scaleTo={0.9}
                        style={[styles.setBadge, set.type !== 'normal' && styles.setBadgeSpecial]}>
                        <Text style={[styles.setBadgeText, set.type !== 'normal' && styles.setBadgeTextSpecial]}>
                          {SET_TYPE_LABEL[set.type] || String(setIndex + 1)}
                        </Text>
                      </Tap>
                      {info.kind === 'weight-reps' ? (
                        <TextInput
                          // defaultValue keeps decimal typing intact ("62." stays visible).
                          key={`kg-${set.id}`}
                          keyboardType="decimal-pad"
                          placeholder="—"
                          placeholderTextColor={palette.inkLow}
                          selectionColor={palette.lime}
                          style={styles.setInput}
                          defaultValue={set.weightKg != null ? String(set.weightKg) : ''}
                          onChangeText={(value) => {
                            const weightKg = Number.parseFloat(value.replace(',', '.'));
                            patchExercise(entry.id, (current) => ({
                              ...current,
                              sets: current.sets.map((candidate) => candidate.id === set.id
                                ? { ...candidate, weightKg: Number.isFinite(weightKg) ? weightKg : undefined }
                                : candidate),
                            }));
                          }}
                        />
                      ) : null}
                      {info.kind === 'duration' ? (
                        <TextInput
                          key={`sec-${set.id}`}
                          keyboardType="number-pad"
                          placeholder="40"
                          placeholderTextColor={palette.inkLow}
                          selectionColor={palette.lime}
                          style={styles.setInput}
                          defaultValue={set.durationSec != null ? String(set.durationSec) : ''}
                          onChangeText={(value) => {
                            const durationSec = Number.parseInt(value, 10);
                            patchExercise(entry.id, (current) => ({
                              ...current,
                              sets: current.sets.map((candidate) => candidate.id === set.id
                                ? { ...candidate, durationSec: Number.isFinite(durationSec) ? durationSec : undefined }
                                : candidate),
                            }));
                          }}
                        />
                      ) : (
                        <TextInput
                          // defaultValue lets a range like "8-12" be typed without the
                          // parser rewriting the field mid-keystroke.
                          key={`reps-${set.id}`}
                          keyboardType="numbers-and-punctuation"
                          placeholder="8-12"
                          placeholderTextColor={palette.inkLow}
                          selectionColor={palette.lime}
                          style={styles.setInput}
                          defaultValue={formatRepTarget(set.repsMin, set.repsMax, set.reps)}
                          onChangeText={(value) => {
                            const range = parseRepTarget(value);
                            patchExercise(entry.id, (current) => ({
                              ...current,
                              sets: current.sets.map((candidate) => candidate.id === set.id
                                ? { ...candidate, ...range }
                                : candidate),
                            }));
                          }}
                        />
                      )}
                      <Tap
                        accessibilityLabel={`Remove set ${setIndex + 1}`}
                        hitSlop={8}
                        onPress={() => patchExercise(entry.id, (current) => ({
                          ...current,
                          sets: current.sets.filter((candidate) => candidate.id !== set.id),
                        }))}
                        scaleTo={0.88}
                        style={styles.removeSet}>
                        <Glyph color={palette.inkLow} name="trash" size={15} />
                      </Tap>
                    </View>
                  ))}

                  <View style={styles.addSetSlot}>
                    <Tap
                      accessibilityLabel={`Add a set to ${info.name}`}
                      onPress={() => patchExercise(entry.id, (current) => ({
                        ...current,
                        sets: [...current.sets, makeRoutineSet({
                          repsMin: current.sets.at(-1)?.repsMin,
                          repsMax: current.sets.at(-1)?.repsMax,
                          reps: current.sets.at(-1)?.reps,
                          weightKg: current.sets.at(-1)?.weightKg,
                          durationSec: current.sets.at(-1)?.durationSec,
                        })],
                      }))}
                      scaleTo={0.96}
                      style={styles.addSet}>
                      <Glyph color={palette.lime} name="plus" size={14} />
                      <Text style={styles.addSetText}>Add set</Text>
                    </Tap>
                  </View>

                  <View style={styles.exerciseFooter}>
                    <Tap
                      accessibilityLabel="Change rest time"
                      onPress={() => {
                        const currentIndex = REST_CHOICES.indexOf(entry.restSec);
                        const nextRest = REST_CHOICES[(currentIndex + 1) % REST_CHOICES.length];
                        patchExercise(entry.id, (current) => ({ ...current, restSec: nextRest }));
                      }}
                      scaleTo={0.93}
                      style={styles.footChip}>
                      <Glyph color={entry.restSec ? palette.lime : palette.inkLow} name="timer" size={13} />
                      <Text style={styles.footChipText}>
                        Rest: {entry.restSec ? `${entry.restSec}s` : 'off'}
                      </Text>
                    </Tap>
                    {index > 0 ? (
                      <Tap
                        accessibilityLabel={linkedWithAbove ? 'Unlink superset' : 'Superset with the exercise above'}
                        onPress={() => toggleSuperset(entry.id)}
                        scaleTo={0.93}
                        style={[styles.footChip, linkedWithAbove && styles.footChipOn]}>
                        <Glyph color={palette.lime} name="link" size={13} />
                        <Text style={[styles.footChipText, styles.footChipTextOn]}>
                          {linkedWithAbove ? 'Unlink superset' : 'Superset with above'}
                        </Text>
                      </Tap>
                    ) : null}
                  </View>

                  <Well style={styles.noteWell}>
                    <TextInput
                      accessibilityLabel={`Note for ${info.name}`}
                      onChangeText={(value) => patchExercise(entry.id, (current) => ({ ...current, note: value || undefined }))}
                      placeholder="Routine note for this exercise (form cue, setup…)"
                      placeholderTextColor={palette.inkLow}
                      selectionColor={palette.lime}
                      style={styles.noteInput}
                      value={entry.note ?? ''}
                    />
                  </Well>
                </Card>
              </Reveal>
            );
          })}

          <Reveal index={6} style={styles.addExerciseSlot}>
            <Tap
              accessibilityLabel="Add exercises"
              haptic="medium"
              onPress={addExercises}
              scaleTo={0.98}
              style={styles.addExercise}>
              <Glyph color={palette.lime} name="plus" size={18} />
              <Text style={styles.addExerciseText}>Add exercises</Text>
            </Tap>
          </Reveal>
        </ScrollView>

        <GlassFooter>
          <View style={styles.footerRow}>
            <View style={styles.footerCancel}>
              <GhostButton icon="close" label="Cancel" onPress={() => router.back()} />
            </View>
            <View style={styles.footerSave}>
              <PrimaryButton icon="check" label="Save routine" onPress={save} />
            </View>
          </View>
        </GlassFooter>
      </KeyboardAvoidingView>
    </Screen>
  );
}

function HeadButton({
  icon,
  label,
  onPress,
  danger,
}: {
  icon: GlyphName;
  label: string;
  onPress: () => void;
  danger?: boolean;
}) {
  return (
    <Tap accessibilityLabel={label} hitSlop={4} onPress={onPress} scaleTo={0.88} style={styles.headButton}>
      <Glyph color={danger ? palette.danger : palette.inkMid} name={icon} size={15} />
    </Tap>
  );
}

function formatRepTarget(repsMin?: number, repsMax?: number, reps?: number) {
  if (repsMin && repsMax) return `${repsMin}-${repsMax}`;
  if (reps != null) return String(reps);
  return '';
}

function parseRepTarget(input: string): { reps?: number; repsMin?: number; repsMax?: number } {
  const cleaned = input.trim();
  if (!cleaned) return { reps: undefined, repsMin: undefined, repsMax: undefined };
  const range = cleaned.match(/^(\d+)\s*[-–]\s*(\d+)$/);
  if (range) {
    const repsMin = Number.parseInt(range[1], 10);
    const repsMax = Number.parseInt(range[2], 10);
    return { reps: undefined, repsMin: Math.min(repsMin, repsMax), repsMax: Math.max(repsMin, repsMax) };
  }
  const single = Number.parseInt(cleaned, 10);
  return Number.isFinite(single)
    ? { reps: single, repsMin: undefined, repsMax: undefined }
    : { reps: undefined, repsMin: undefined, repsMax: undefined };
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  content: { paddingHorizontal: space.md, paddingTop: 8, paddingBottom: space.lg },

  label: { ...text.label, color: palette.inkLow, marginBottom: 6 },
  nameInput: { ...text.title, color: palette.ink, padding: 0, paddingVertical: 2 },
  folderBlock: { marginTop: 18, borderTopWidth: 1, borderTopColor: palette.line, paddingTop: 14 },
  folderRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  folderInput: { flex: 1, ...text.row, color: palette.ink, padding: 0, paddingVertical: 2 },

  exerciseSlot: { marginTop: 10 },
  exerciseSlotLinked: { marginTop: 4 },
  exerciseCard: { padding: 13 },
  exerciseCardLinked: { borderColor: `${palette.lime}33` },
  exerciseHead: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  exerciseHeadCopy: { flex: 1 },
  exerciseName: { ...text.row, color: palette.ink },
  exerciseMeta: { ...text.caption, fontSize: 10.5, color: palette.inkLow, marginTop: 3, textTransform: 'capitalize' },
  headActions: { flexDirection: 'row', gap: 6 },
  headButton: {
    width: 32,
    height: 32,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceLo,
    alignItems: 'center',
    justifyContent: 'center',
  },

  supersetTag: { flexDirection: 'row', marginTop: 10 },

  setHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 14, marginBottom: 6 },
  setHeaderText: { ...text.label, fontSize: 8, letterSpacing: 1, color: palette.inkLow },
  colSet: { width: 36, textAlign: 'center' },
  colInput: { flex: 1 },
  colRemove: { width: 28, alignItems: 'center' },
  setRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  setBadge: {
    width: 36,
    height: 44,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceLo,
    borderWidth: 1,
    borderColor: palette.line,
    alignItems: 'center',
    justifyContent: 'center',
  },
  setBadgeSpecial: { backgroundColor: palette.limeSoft, borderColor: `${palette.lime}44` },
  setBadgeText: { ...text.value, fontSize: 13, color: palette.inkMid, ...tabular },
  setBadgeTextSpecial: { color: palette.lime },
  setInput: {
    flex: 1,
    height: 44,
    borderWidth: 1,
    borderColor: palette.line,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceLo,
    color: palette.ink,
    ...text.value,
    fontSize: 14,
    paddingHorizontal: 12,
    ...tabular,
  },
  removeSet: { width: 28, height: 44, alignItems: 'center', justifyContent: 'center' },

  addSetSlot: { alignSelf: 'flex-start' },
  addSet: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    minHeight: 40,
    paddingRight: 12,
  },
  addSetText: { ...text.value, fontSize: 12.5, color: palette.lime },

  exerciseFooter: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 6 },
  footChip: {
    minHeight: 34,
    paddingHorizontal: 11,
    borderRadius: radius.pill,
    backgroundColor: palette.surfaceLo,
    borderWidth: 1,
    borderColor: palette.lineHi,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  footChipOn: { borderColor: `${palette.lime}44`, backgroundColor: palette.limeSoft },
  footChipText: { ...text.micro, fontFamily: text.value.fontFamily, fontSize: 11, color: palette.ink, ...tabular },
  footChipTextOn: { color: palette.lime },

  noteWell: { marginTop: 10, paddingVertical: 8 },
  noteInput: { ...text.caption, minHeight: 34, color: palette.ink, padding: 0 },

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

  footerRow: { flexDirection: 'row', gap: 10 },
  footerCancel: { flex: 1 },
  footerSave: { flex: 2 },
});
