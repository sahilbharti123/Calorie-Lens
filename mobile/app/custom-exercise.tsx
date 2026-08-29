import { useLocalSearchParams, useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import {
  Alert,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { Glyph } from '@/src/components/glyph';
import { Card, Chip, GhostButton, GlassFooter, PrimaryButton, Screen, SectionTitle } from '@/src/components/ui';
import { EQUIPMENT_TYPES, MUSCLE_GROUPS, type Equipment, type ExerciseKind, type MuscleGroup } from '@/src/lib/exercises';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';
import { palette, radius, space, text } from '@/src/theme';

const KINDS: { value: ExerciseKind; label: string; detail: string }[] = [
  { value: 'weight-reps', label: 'KG × reps', detail: 'Weights, machines and loaded movements' },
  { value: 'reps-only', label: 'Reps', detail: 'Bodyweight movements counted by repetition' },
  { value: 'duration', label: 'Timed', detail: 'Holds, cardio and interval movements' },
];

function titleCase(value: string) {
  return value.replace(/(^|\s)\S/g, (character) => character.toUpperCase());
}

export default function CustomExerciseScreen() {
  const router = useRouter();
  const { id, name: suggestedName } = useLocalSearchParams<{ id?: string; name?: string }>();
  const { data } = useApp();
  const workouts = useWorkouts();
  const existing = useMemo(
    () => data.training.customExercises.find((exercise) => exercise.id === id),
    [data.training.customExercises, id],
  );
  const [name, setName] = useState(existing?.name ?? suggestedName?.trim() ?? '');
  const [kind, setKind] = useState<ExerciseKind>(existing?.kind ?? 'weight-reps');
  const [equipment, setEquipment] = useState<Equipment>((existing?.equipment as Equipment) ?? 'other');
  const [muscle, setMuscle] = useState<MuscleGroup>((existing?.primaryMuscle as MuscleGroup) ?? 'full body');
  const valid = name.trim().length >= 2;

  function save() {
    if (!valid) return;
    const value = { name: name.trim(), kind, equipment, primaryMuscle: muscle };
    if (existing) workouts.updateCustomExercise(existing.id, value);
    else workouts.createCustomExercise(value);
    Keyboard.dismiss();
    router.back();
  }

  function remove() {
    Alert.alert(
      'Delete custom exercise?',
      'Past completed workouts keep their recorded sets. Exercises used by an active workout or routine must be removed there first.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            if (!existing) return;
            if (!workouts.deleteCustomExercise(existing.id)) {
              Alert.alert('Exercise is still in use', 'Remove it from the active workout and every routine before deleting it.');
              return;
            }
            router.back();
          },
        },
      ],
    );
  }

  return (
    <Screen edges={['bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          onTouchMove={Keyboard.dismiss}
          showsVerticalScrollIndicator={false}>
          <Card>
            <View style={styles.heroIcon}>
              <Glyph color={palette.lime} name={kind === 'duration' ? 'timer' : 'dumbbell'} size={22} />
            </View>
            <Text style={styles.title}>{existing ? 'Edit custom exercise' : 'Create an exercise'}</Text>
            <Text style={styles.body}>Give unfamiliar equipment, rehabilitation movements, or intervals the exact logging controls they need.</Text>
            <Text style={styles.label}>NAME</Text>
            <TextInput
              accessibilityLabel="Exercise name"
              autoCapitalize="words"
              autoFocus={!existing}
              maxLength={60}
              onChangeText={setName}
              placeholder="e.g. Cable-supported split squat"
              placeholderTextColor={palette.inkLow}
              returnKeyType="done"
              style={styles.input}
              value={name}
            />
          </Card>

          <SectionTitle title="How it is logged" />
          <View style={styles.kindGrid}>
            {KINDS.map((option) => (
              <Chip
                active={kind === option.value}
                key={option.value}
                label={option.label}
                onPress={() => setKind(option.value)}
              />
            ))}
          </View>
          <Text style={styles.selectionDetail}>{KINDS.find((option) => option.value === kind)?.detail}</Text>

          <SectionTitle title="Equipment" />
          <View style={styles.chipGrid}>
            {EQUIPMENT_TYPES.map((option) => (
              <Chip active={equipment === option} key={option} label={titleCase(option)} onPress={() => setEquipment(option)} />
            ))}
          </View>

          <SectionTitle title="Primary area" />
          <View style={styles.chipGrid}>
            {MUSCLE_GROUPS.map((option) => (
              <Chip active={muscle === option} key={option} label={titleCase(option)} onPress={() => setMuscle(option)} />
            ))}
          </View>

          {existing ? <GhostButton icon="trash" label="Delete custom exercise" onPress={remove} tone="danger" /> : null}
        </ScrollView>

        <GlassFooter>
          <PrimaryButton disabled={!valid} icon="check" label={existing ? 'Save exercise' : 'Create exercise'} onPress={save} />
        </GlassFooter>
      </KeyboardAvoidingView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  content: { padding: space.md, paddingBottom: space.tabClearance, gap: 12 },
  heroIcon: { width: 46, height: 46, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.limeSoft },
  title: { ...text.section, color: palette.ink, marginTop: 14 },
  body: { ...text.caption, color: palette.inkMid, marginTop: 6 },
  label: { ...text.label, color: palette.inkLow, marginTop: 18, marginBottom: 7 },
  input: { ...text.row, minHeight: 54, color: palette.ink, backgroundColor: palette.surfaceLo, borderWidth: 1, borderColor: palette.lineHi, borderRadius: radius.sm, paddingHorizontal: 14 },
  kindGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chipGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  selectionDetail: { ...text.caption, color: palette.inkMid },
});
