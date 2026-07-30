import { useRouter } from 'expo-router';
import { useState } from 'react';
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
import { palette, radius, space, type } from '@/src/theme';
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
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <View style={styles.nameCard}>
            <Text style={styles.label}>ROUTINE NAME</Text>
            <TextInput
              placeholder="e.g. Push Day"
              placeholderTextColor="#8A938B"
              style={styles.nameInput}
              value={draft.name}
              onChangeText={(name) => setDraft((current) => ({ ...current, name }))}
            />
            <Text style={styles.label}>FOLDER (OPTIONAL)</Text>
            <TextInput
              placeholder="e.g. Push · Pull · Legs"
              placeholderTextColor="#8A938B"
              style={styles.folderInput}
              value={draft.folder ?? ''}
              onChangeText={(folder) => setDraft((current) => ({ ...current, folder: folder || undefined }))}
            />
          </View>

          {draft.exercises.map((entry, index) => {
            const info = exerciseInfo(data.training, entry.exerciseId);
            const inSuperset = Boolean(entry.supersetId);
            const linkedWithAbove = index > 0 && inSuperset && draft.exercises[index - 1].supersetId === entry.supersetId;
            return (
              <View key={entry.id} style={[styles.exercise, linkedWithAbove && styles.exerciseLinked]}>
                <View style={styles.exerciseHead}>
                  <Pressable style={{ flex: 1 }} onPress={() => router.push({ pathname: '/exercise/[id]', params: { id: entry.exerciseId } })}>
                    <Text style={styles.exerciseName}>{info.name}</Text>
                    <Text style={styles.exerciseMeta}>{info.primaryMuscle} · {info.equipment}</Text>
                  </Pressable>
                  <View style={styles.headActions}>
                    <HeadButton label="↑" onPress={() => move(entry.id, -1)} />
                    <HeadButton label="↓" onPress={() => move(entry.id, 1)} />
                    <HeadButton label="✕" onPress={() => removeExercise(entry.id)} />
                  </View>
                </View>

                {inSuperset ? (
                  <View style={styles.supersetTag}><Text style={styles.supersetTagText}>SUPERSET</Text></View>
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
                    <Pressable
                      onPress={() => patchExercise(entry.id, (current) => ({
                        ...current,
                        sets: current.sets.map((candidate) => candidate.id === set.id
                          ? { ...candidate, type: nextSetType(candidate.type) }
                          : candidate),
                      }))}
                      style={[styles.setBadge, set.type !== 'normal' && styles.setBadgeSpecial]}>
                      <Text style={[styles.setBadgeText, set.type !== 'normal' && styles.setBadgeTextSpecial]}>
                        {SET_TYPE_LABEL[set.type] || String(setIndex + 1)}
                      </Text>
                    </Pressable>
                    {info.kind === 'weight-reps' ? (
                      <TextInput
                        // defaultValue keeps decimal typing intact ("62." stays visible).
                        key={`kg-${set.id}`}
                        keyboardType="decimal-pad"
                        placeholder="—"
                        placeholderTextColor="#A8B0A6"
                        style={styles.setInput}
                        defaultValue={set.weightKg != null ? String(set.weightKg) : ''}
                        onChangeText={(text) => {
                          const weightKg = Number.parseFloat(text.replace(',', '.'));
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
                        placeholderTextColor="#A8B0A6"
                        style={styles.setInput}
                        defaultValue={set.durationSec != null ? String(set.durationSec) : ''}
                        onChangeText={(text) => {
                          const durationSec = Number.parseInt(text, 10);
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
                        placeholderTextColor="#A8B0A6"
                        style={styles.setInput}
                        defaultValue={formatRepTarget(set.repsMin, set.repsMax, set.reps)}
                        onChangeText={(text) => {
                          const range = parseRepTarget(text);
                          patchExercise(entry.id, (current) => ({
                            ...current,
                            sets: current.sets.map((candidate) => candidate.id === set.id
                              ? { ...candidate, ...range }
                              : candidate),
                          }));
                        }}
                      />
                    )}
                    <Pressable
                      hitSlop={8}
                      onPress={() => patchExercise(entry.id, (current) => ({
                        ...current,
                        sets: current.sets.filter((candidate) => candidate.id !== set.id),
                      }))}
                      style={styles.colRemove}>
                      <Glyph name="trash" color={palette.muted} size={15} />
                    </Pressable>
                  </View>
                ))}
                <Pressable
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
                  style={styles.addSet}>
                  <Text style={styles.addSetText}>+ Add set</Text>
                </Pressable>

                <View style={styles.exerciseFooter}>
                  <Pressable
                    onPress={() => {
                      const currentIndex = REST_CHOICES.indexOf(entry.restSec);
                      const nextRest = REST_CHOICES[(currentIndex + 1) % REST_CHOICES.length];
                      patchExercise(entry.id, (current) => ({ ...current, restSec: nextRest }));
                    }}
                    style={styles.restChip}>
                    <Text style={styles.restChipText}>
                      Rest: {entry.restSec ? `${entry.restSec}s` : 'off'}
                    </Text>
                  </Pressable>
                  {index > 0 ? (
                    <Pressable onPress={() => toggleSuperset(entry.id)} style={styles.supersetChip}>
                      <Text style={styles.supersetChipText}>
                        {linkedWithAbove ? 'Unlink superset' : 'Superset with above'}
                      </Text>
                    </Pressable>
                  ) : null}
                </View>
                <TextInput
                  placeholder="Routine note for this exercise (form cue, setup…)"
                  placeholderTextColor="#A8B0A6"
                  style={styles.noteInput}
                  value={entry.note ?? ''}
                  onChangeText={(text) => patchExercise(entry.id, (current) => ({ ...current, note: text || undefined }))}
                />
              </View>
            );
          })}

          <Pressable onPress={addExercises} style={({ pressed }) => [styles.addExercise, pressed && styles.pressed]}>
            <Glyph name="plus" color={palette.forest} size={18} />
            <Text style={styles.addExerciseText}>Add exercises</Text>
          </Pressable>
        </ScrollView>

        <View style={styles.footer}>
          <Pressable onPress={() => router.back()} style={({ pressed }) => [styles.cancel, pressed && styles.pressed]}>
            <Text style={styles.cancelText}>Cancel</Text>
          </Pressable>
          <Pressable onPress={save} style={({ pressed }) => [styles.save, pressed && styles.pressed]}>
            <Text style={styles.saveText}>Save routine</Text>
          </Pressable>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function HeadButton({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable hitSlop={6} onPress={onPress} style={styles.headButton}>
      <Text style={styles.headButtonText}>{label}</Text>
    </Pressable>
  );
}

function formatRepTarget(repsMin?: number, repsMax?: number, reps?: number) {
  if (repsMin && repsMax) return `${repsMin}-${repsMax}`;
  if (reps != null) return String(reps);
  return '';
}

function parseRepTarget(text: string): { reps?: number; repsMin?: number; repsMax?: number } {
  const cleaned = text.trim();
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
  safe: { flex: 1, backgroundColor: palette.canvas },
  fill: { flex: 1 },
  content: { padding: space.md, paddingBottom: 24, gap: 10 },
  pressed: { opacity: 0.85 },
  nameCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14 },
  label: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.1, marginBottom: 4 },
  nameInput: { color: palette.ink, fontFamily: type.demi, fontSize: 17, paddingVertical: 6, marginBottom: 10 },
  folderInput: { color: palette.ink, fontFamily: type.medium, fontSize: 13, paddingVertical: 4 },
  exercise: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 13 },
  exerciseLinked: { borderColor: palette.limeDark, marginTop: -4 },
  exerciseHead: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  exerciseName: { color: palette.ink, fontFamily: type.demi, fontSize: 14 },
  exerciseMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 2, textTransform: 'capitalize' },
  headActions: { flexDirection: 'row', gap: 6 },
  headButton: { width: 30, height: 30, borderRadius: 10, borderWidth: 1, borderColor: palette.line, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.canvas },
  headButtonText: { color: palette.ink, fontFamily: type.medium, fontSize: 12 },
  supersetTag: { alignSelf: 'flex-start', backgroundColor: palette.softLime, borderRadius: radius.pill, paddingHorizontal: 8, paddingVertical: 3, marginTop: 8 },
  supersetTagText: { color: palette.limeDark, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 1 },
  setHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 12, marginBottom: 5 },
  setHeaderText: { color: palette.muted, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 0.8 },
  colSet: { width: 34 },
  colInput: { flex: 1 },
  colRemove: { width: 26, alignItems: 'center' },
  setRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 7 },
  setBadge: { width: 34, height: 34, borderRadius: 11, backgroundColor: palette.canvas, borderWidth: 1, borderColor: palette.line, alignItems: 'center', justifyContent: 'center' },
  setBadgeSpecial: { backgroundColor: palette.softLime, borderColor: palette.limeDark },
  setBadgeText: { color: palette.ink, fontFamily: type.demi, fontSize: 12 },
  setBadgeTextSpecial: { color: palette.limeDark },
  setInput: { flex: 1, height: 38, borderWidth: 1, borderColor: palette.line, borderRadius: radius.sm, backgroundColor: palette.canvas, color: palette.ink, fontFamily: type.medium, fontSize: 13, paddingHorizontal: 10 },
  addSet: { alignSelf: 'flex-start', paddingVertical: 6 },
  addSetText: { color: palette.forest, fontFamily: type.demi, fontSize: 11.5 },
  exerciseFooter: { flexDirection: 'row', gap: 8, marginTop: 6 },
  restChip: { height: 30, paddingHorizontal: 11, borderRadius: radius.pill, backgroundColor: palette.canvas, borderWidth: 1, borderColor: palette.line, alignItems: 'center', justifyContent: 'center' },
  restChipText: { color: palette.ink, fontFamily: type.medium, fontSize: 10.5 },
  supersetChip: { height: 30, paddingHorizontal: 11, borderRadius: radius.pill, backgroundColor: palette.canvas, borderWidth: 1, borderColor: palette.line, alignItems: 'center', justifyContent: 'center' },
  supersetChipText: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10.5 },
  noteInput: { marginTop: 8, minHeight: 34, borderWidth: 1, borderColor: palette.line, borderRadius: radius.sm, backgroundColor: palette.canvas, color: palette.ink, fontFamily: type.regular, fontSize: 11.5, paddingHorizontal: 10, paddingVertical: 8 },
  addExercise: { height: 52, borderRadius: radius.md, borderWidth: 1.5, borderColor: palette.forest, borderStyle: 'dashed', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  addExerciseText: { color: palette.forest, fontFamily: type.demi, fontSize: 13 },
  footer: { flexDirection: 'row', gap: 10, padding: space.md, borderTopWidth: 1, borderTopColor: palette.line, backgroundColor: palette.canvas },
  cancel: { flex: 1, height: 50, borderRadius: radius.md, borderWidth: 1, borderColor: palette.line, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.paper },
  cancelText: { color: palette.ink, fontFamily: type.medium, fontSize: 13 },
  save: { flex: 2, height: 50, borderRadius: radius.md, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  saveText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
});
