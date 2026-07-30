import { Stack, useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';

import { Glyph } from '@/src/components/glyph';
import {
  Card,
  CountUp,
  EmptyState,
  GhostButton,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  SectionTitle,
  Tap,
  Well,
} from '@/src/components/ui';
import { exerciseInfo, formatDuration, formatSet } from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';
import { motion, palette, space, tabular, text } from '@/src/theme';

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
      <Screen edges={['bottom']}>
        <Stack.Screen options={{ title: 'Workout' }} />
        <View style={styles.missing}>
          <EmptyState
            action={<PrimaryButton icon="chevronLeft" label="Back to history" onPress={() => router.back()} />}
            body="This workout is no longer available."
            icon="calendar"
            title="Nothing to show here"
          />
        </View>
      </Screen>
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
    <Screen edges={['bottom']}>
      <Stack.Screen options={{ title: celebrate ? 'Workout complete' : session.name }} />
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {celebrate ? (
          <Celebration
            meta={`Workout #${training.sessions.length} saved${prSets.length ? ` · ${prSets.length} personal record${prSets.length > 1 ? 's' : ''}` : ''}`}
          />
        ) : null}

        {/* ---------- Summary ---------- */}
        <Reveal index={1} style={celebrate ? styles.summarySlot : undefined}>
          <Card glow={!celebrate} raised={!celebrate}>
            <Text style={styles.dateLine}>
              {new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' })}
              {' · '}
              {new Date(session.startedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </Text>

            <View style={styles.volumeRow}>
              <CountUp style={styles.volume} value={session.totalVolumeKg ?? 0} />
              <Text style={styles.volumeUnit}>kg</Text>
            </View>
            <Text style={styles.volumeCaption}>Total volume lifted</Text>

            <View style={styles.statRow}>
              <Well style={styles.stat}>
                <Text style={styles.statLabel}>DURATION</Text>
                <Text style={styles.statValue}>{formatDuration((session.durationMin ?? 0) * 60)}</Text>
              </Well>
              <Well style={styles.stat}>
                <Text style={styles.statLabel}>SETS</Text>
                <Text style={styles.statValue}>{String(session.totalSets ?? 0)}</Text>
              </Well>
            </View>

            {prSets.length ? (
              <View style={styles.prRow}>
                <Pill
                  icon="trophy"
                  label={`${prSets.length} personal record${prSets.length > 1 ? 's' : ''}`}
                  tone="accent"
                />
              </View>
            ) : null}
          </Card>
        </Reveal>

        {/* ---------- Breakdown ---------- */}
        <SectionTitle title="Exercises" />
        <Reveal index={2}>
          <Card padded={false} style={styles.breakdown}>
            {session.exercises.map((entry, index) => {
              const info = exerciseInfo(training, entry.exerciseId);
              return (
                <View key={entry.id} style={[styles.exercise, index > 0 && styles.exerciseDivider]}>
                  <Tap
                    accessibilityLabel={`How to do ${info.name}`}
                    onPress={() => router.push({ pathname: '/exercise/[id]', params: { id: entry.exerciseId } })}
                    scaleTo={0.99}>
                    <View style={styles.exerciseHead}>
                      <Text numberOfLines={1} style={styles.exerciseName}>{info.name}</Text>
                      <Glyph color={palette.inkLow} name="chevron" size={14} />
                    </View>
                  </Tap>
                  {entry.sets.map((set, setIndex) => (
                    <View key={set.id} style={styles.setLine}>
                      <Text style={styles.setIndex}>{setIndex + 1}</Text>
                      <Text style={styles.setText}>{formatSet(set, info.kind)}</Text>
                      {set.type !== 'normal' ? <Text style={styles.setTag}>{set.type}</Text> : null}
                      {set.rpe ? <Text style={styles.setTag}>RPE {set.rpe}</Text> : null}
                      {set.prFlags?.map((flag) => (
                        <Pill icon="trophy" key={flag} label={flag} tone="accent" />
                      ))}
                    </View>
                  ))}
                </View>
              );
            })}
          </Card>
        </Reveal>

        {/* ---------- Energy ---------- */}
        {session.calories ? (
          <Reveal index={3} style={styles.blockSlot}>
            <Card>
              <View style={styles.energyHead}>
                <View style={styles.energyIcon}>
                  <Glyph color={palette.fat} name="flame" size={16} />
                </View>
                <Text style={styles.energyTitle}>
                  ~{session.calorieLow}–{session.calorieHigh} active kcal (midpoint {session.calories})
                </Text>
              </View>
              <Text style={styles.energyBasis}>{session.calorieBasis}</Text>
              <Text style={styles.energyNote}>Added to today’s activity total automatically.</Text>
            </Card>
          </Reveal>
        ) : null}

        {/* ---------- Actions ---------- */}
        {savingRoutine ? (
          <Reveal index={4} style={styles.blockSlot}>
            <Card>
              <Text style={styles.label}>ROUTINE NAME</Text>
              <TextInput
                accessibilityLabel="Routine name"
                autoFocus
                onChangeText={setRoutineName}
                placeholder={session.name}
                placeholderTextColor={palette.inkLow}
                selectionColor={palette.lime}
                style={styles.routineInput}
                value={routineName}
              />
              <View style={styles.actionRow}>
                <View style={styles.actionSlot}>
                  <GhostButton icon="close" label="Cancel" onPress={() => setSavingRoutine(false)} />
                </View>
                <View style={styles.actionSlot}>
                  <PrimaryButton icon="check" label="Save routine" onPress={saveAsRoutine} />
                </View>
              </View>
            </Card>
          </Reveal>
        ) : (
          <Reveal index={4} style={styles.blockSlot}>
            <GhostButton
              icon="copy"
              label="Save this workout as a routine"
              onPress={() => setSavingRoutine(true)}
            />
          </Reveal>
        )}

        <Reveal index={5} style={styles.deleteSlot}>
          <GhostButton icon="trash" label="Delete workout" onPress={deleteWorkout} tone="danger" />
        </Reveal>
      </ScrollView>
    </Screen>
  );
}

function Celebration({ meta }: { meta: string }) {
  const enter = useSharedValue(0);

  useEffect(() => {
    enter.value = withSpring(1, motion.bouncy);
  }, [enter]);

  const badge = useAnimatedStyle(() => ({
    transform: [
      { scale: 0.5 + enter.value * 0.5 },
      { rotate: `${(1 - enter.value) * -25}deg` },
    ],
  }));

  const copy = useAnimatedStyle(() => ({
    opacity: Math.min(1, enter.value * 1.5),
    transform: [{ translateY: (1 - enter.value) * 10 }],
  }));

  return (
    <Card glow raised>
      <View style={styles.celebrate}>
        <Animated.View style={[styles.celebrateIcon, badge]}>
          <Glyph color={palette.onLime} name="spark" size={26} />
        </Animated.View>
        <Animated.View style={[styles.celebrateCopy, copy]}>
          <Text style={styles.celebrateTitle}>Nice work.</Text>
          <Text style={styles.celebrateMeta}>{meta}</Text>
        </Animated.View>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: space.md, paddingTop: 10, paddingBottom: space.tabClearance },
  missing: { flex: 1, justifyContent: 'center' },

  celebrate: { alignItems: 'center', paddingVertical: 8 },
  celebrateIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: palette.lime,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
  },
  celebrateCopy: { alignItems: 'center' },
  celebrateTitle: { ...text.title, color: palette.ink },
  celebrateMeta: { ...text.caption, color: palette.inkMid, marginTop: 6, textAlign: 'center' },

  summarySlot: { marginTop: 10 },
  dateLine: { ...text.label, color: palette.inkLow },
  volumeRow: { flexDirection: 'row', alignItems: 'baseline', gap: 6, marginTop: 10 },
  volume: { ...text.hero, color: palette.ink, ...tabular },
  volumeUnit: { ...text.headline, fontSize: 17, color: palette.inkMid },
  volumeCaption: { ...text.caption, color: palette.inkMid, marginTop: 2 },
  statRow: { flexDirection: 'row', gap: 8, marginTop: 18 },
  stat: { flex: 1, paddingVertical: 11 },
  statLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1, color: palette.inkLow },
  statValue: { ...text.headline, fontSize: 18, color: palette.ink, marginTop: 4, ...tabular },
  prRow: { flexDirection: 'row', marginTop: 14 },

  breakdown: { paddingHorizontal: 14, paddingVertical: 4 },
  exercise: { paddingVertical: 12 },
  exerciseDivider: { borderTopWidth: 1, borderTopColor: palette.line },
  exerciseHead: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  exerciseName: { ...text.row, flex: 1, color: palette.ink },
  setLine: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 8, minHeight: 30 },
  setIndex: { width: 18, ...text.value, fontSize: 11, color: palette.inkLow, ...tabular },
  setText: { ...text.value, color: palette.ink, ...tabular },
  setTag: { ...text.micro, fontSize: 9.5, letterSpacing: 0.6, color: palette.inkMid, textTransform: 'uppercase' },

  blockSlot: { marginTop: 12 },
  energyHead: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  energyIcon: {
    width: 32,
    height: 32,
    borderRadius: 11,
    backgroundColor: `${palette.fat}1A`,
    alignItems: 'center',
    justifyContent: 'center',
  },
  energyTitle: { ...text.value, flex: 1, fontSize: 13, color: palette.ink },
  energyBasis: { ...text.caption, fontSize: 11, color: palette.inkMid, marginTop: 10 },
  energyNote: { ...text.caption, fontSize: 10.5, color: palette.inkLow, marginTop: 6 },

  label: { ...text.label, color: palette.inkLow, marginBottom: 6 },
  routineInput: {
    ...text.headline,
    fontSize: 18,
    color: palette.ink,
    padding: 0,
    paddingVertical: 4,
    borderBottomWidth: 1,
    borderBottomColor: palette.line,
  },
  actionRow: { flexDirection: 'row', gap: 10, marginTop: 16 },
  actionSlot: { flex: 1 },

  deleteSlot: { marginTop: 12 },
});
