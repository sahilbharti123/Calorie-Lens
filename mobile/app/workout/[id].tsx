import { Stack, useLocalSearchParams, useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { exerciseInfo, formatDuration, formatSet } from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';
import { palette, radius, space, type } from '@/src/theme';

export default function WorkoutDetailScreen() {
  const { id, celebrate } = useLocalSearchParams<{ id: string; celebrate?: string }>();
  const router = useRouter();
  const { data } = useApp();
  const workouts = useWorkouts();
  const training = data.training;
  const session = training.sessions.find((candidate) => candidate.id === id);
  const [savingRoutine, setSavingRoutine] = useState(false);
  const [routineName, setRoutineName] = useState('');

  if (!session) {
    return (
      <SafeAreaView style={styles.safe}>
        <Stack.Screen options={{ title: 'Workout' }} />
        <View style={styles.missing}><Text style={styles.missingText}>This workout is no longer available.</Text></View>
      </SafeAreaView>
    );
  }

  const prSets = session.exercises.flatMap((entry) => entry.sets.filter((set) => set.prFlags?.length));

  function deleteWorkout() {
    Alert.alert('Delete workout', 'Remove this session from your history? Daily calorie logs are kept.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          workouts.deleteSession(session!.id);
          router.back();
        },
      },
    ]);
  }

  function saveAsRoutine() {
    const name = routineName.trim() || session!.name;
    workouts.saveSessionAsRoutine(session!.id, name);
    setSavingRoutine(false);
    setRoutineName('');
    Alert.alert('Routine saved', `“${name}” was added to your routines.`);
  }

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <Stack.Screen options={{ title: celebrate ? 'Workout complete' : session.name }} />
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {celebrate ? (
          <View style={styles.celebrate}>
            <View style={styles.celebrateIcon}><Glyph name="spark" color={palette.forest} size={26} /></View>
            <Text style={styles.celebrateTitle}>Nice work.</Text>
            <Text style={styles.celebrateMeta}>
              Workout #{training.sessions.length} saved{prSets.length ? ` · ${prSets.length} personal record${prSets.length > 1 ? 's' : ''}` : ''}
            </Text>
          </View>
        ) : null}

        <View style={styles.statsRow}>
          <Stat label="Duration" value={formatDuration((session.durationMin ?? 0) * 60)} />
          <Stat label="Volume" value={`${(session.totalVolumeKg ?? 0).toLocaleString()} kg`} />
          <Stat label="Sets" value={String(session.totalSets ?? 0)} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardLabel}>
            {new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' })}
            {' · '}
            {new Date(session.startedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </Text>
          {session.exercises.map((entry) => {
            const info = exerciseInfo(training, entry.exerciseId);
            return (
              <View key={entry.id} style={styles.exercise}>
                <Pressable onPress={() => router.push({ pathname: '/exercise/[id]', params: { id: entry.exerciseId } })}>
                  <Text style={styles.exerciseName}>{info.name}</Text>
                </Pressable>
                {entry.sets.map((set, index) => (
                  <View key={set.id} style={styles.setLine}>
                    <Text style={styles.setIndex}>{index + 1}</Text>
                    <Text style={styles.setText}>{formatSet(set, info.kind)}</Text>
                    {set.type !== 'normal' ? <Text style={styles.setTag}>{set.type}</Text> : null}
                    {set.rpe ? <Text style={styles.setTag}>RPE {set.rpe}</Text> : null}
                    {set.prFlags?.map((flag) => (
                      <Text key={flag} style={styles.prFlag}>🏆 {flag}</Text>
                    ))}
                  </View>
                ))}
              </View>
            );
          })}
        </View>

        {session.calories ? (
          <View style={styles.energy}>
            <Text style={styles.energyTitle}>
              ~{session.calorieLow}–{session.calorieHigh} active kcal (midpoint {session.calories})
            </Text>
            <Text style={styles.energyBasis}>{session.calorieBasis}</Text>
            <Text style={styles.energyNote}>Added to today’s activity total automatically.</Text>
          </View>
        ) : null}

        {savingRoutine ? (
          <View style={styles.saveRoutineCard}>
            <Text style={styles.saveRoutineLabel}>ROUTINE NAME</Text>
            <TextInput
              autoFocus
              placeholder={session.name}
              placeholderTextColor="#8A938B"
              style={styles.saveRoutineInput}
              value={routineName}
              onChangeText={setRoutineName}
            />
            <View style={styles.saveRoutineRow}>
              <Pressable onPress={() => setSavingRoutine(false)} style={[styles.actionButton, styles.actionGhost]}>
                <Text style={styles.actionGhostText}>Cancel</Text>
              </Pressable>
              <Pressable onPress={saveAsRoutine} style={[styles.actionButton, styles.actionPrimary]}>
                <Text style={styles.actionPrimaryText}>Save routine</Text>
              </Pressable>
            </View>
          </View>
        ) : (
          <Pressable onPress={() => setSavingRoutine(true)} style={({ pressed }) => [styles.saveAs, pressed && styles.pressed]}>
            <Text style={styles.saveAsText}>Save this workout as a routine</Text>
          </Pressable>
        )}

        <Pressable onPress={deleteWorkout} style={styles.delete}>
          <Text style={styles.deleteText}>Delete workout</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label.toUpperCase()}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { padding: space.md, paddingBottom: 30 },
  pressed: { opacity: 0.85 },
  missing: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  missingText: { color: palette.muted, fontFamily: type.regular, fontSize: 13 },
  celebrate: { alignItems: 'center', backgroundColor: palette.forest, borderRadius: radius.lg, paddingVertical: 24, marginBottom: 12 },
  celebrateIcon: { width: 52, height: 52, borderRadius: 26, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center', marginBottom: 10 },
  celebrateTitle: { color: palette.white, fontFamily: type.demi, fontSize: 22, letterSpacing: -0.6 },
  celebrateMeta: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 11.5, marginTop: 5 },
  statsRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  stat: { flex: 1, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingVertical: 13, alignItems: 'center' },
  statValue: { color: palette.ink, fontFamily: type.demi, fontSize: 15 },
  statLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 1, marginTop: 3 },
  card: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 15, marginBottom: 12 },
  cardLabel: { color: palette.muted, fontFamily: type.medium, fontSize: 11, marginBottom: 4 },
  exercise: { borderTopWidth: 1, borderTopColor: palette.line, marginTop: 11, paddingTop: 11 },
  exerciseName: { color: palette.ink, fontFamily: type.demi, fontSize: 14, marginBottom: 6 },
  setLine: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 8, paddingVertical: 3.5 },
  setIndex: { width: 18, color: palette.muted, fontFamily: type.demi, fontSize: 10.5 },
  setText: { color: palette.ink, fontFamily: type.medium, fontSize: 12.5 },
  setTag: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10, textTransform: 'uppercase' },
  prFlag: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10.5 },
  energy: { backgroundColor: palette.softLime, borderRadius: radius.md, padding: 14, marginBottom: 12 },
  energyTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 13 },
  energyBasis: { color: palette.limeDark, fontFamily: type.medium, fontSize: 10, lineHeight: 15, marginTop: 5 },
  energyNote: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, marginTop: 5 },
  saveAs: { height: 50, borderRadius: radius.md, borderWidth: 1.5, borderColor: palette.forest, alignItems: 'center', justifyContent: 'center', marginBottom: 8 },
  saveAsText: { color: palette.forest, fontFamily: type.demi, fontSize: 13 },
  saveRoutineCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14, marginBottom: 8 },
  saveRoutineLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.1, marginBottom: 4 },
  saveRoutineInput: { color: palette.ink, fontFamily: type.demi, fontSize: 15, paddingVertical: 6 },
  saveRoutineRow: { flexDirection: 'row', gap: 8, marginTop: 10 },
  actionButton: { flex: 1, height: 42, borderRadius: radius.sm, alignItems: 'center', justifyContent: 'center' },
  actionGhost: { borderWidth: 1, borderColor: palette.line, backgroundColor: palette.canvas },
  actionGhostText: { color: palette.ink, fontFamily: type.medium, fontSize: 12 },
  actionPrimary: { backgroundColor: palette.forest },
  actionPrimaryText: { color: palette.lime, fontFamily: type.demi, fontSize: 12.5 },
  delete: { alignItems: 'center', paddingVertical: 10 },
  deleteText: { color: '#B64B45', fontFamily: type.demi, fontSize: 12 },
});
