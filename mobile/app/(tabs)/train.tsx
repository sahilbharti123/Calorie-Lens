import { useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { EmptyState, ScreenHeader, SectionTitle } from '@/src/components/ui';
import { TEMPLATE_ROUTINE_SEEDS, findExercise } from '@/src/lib/exercises';
import {
  REST_CHOICES,
  completedSessions,
  exerciseInfo,
  formatDuration,
  weeklyMuscleSets,
  weeklyTrainingSummary,
} from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { useWorkouts } from '@/src/store/workout-store';
import { palette, radius, space, type } from '@/src/theme';
import type { Routine } from '@/src/types';

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

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader eyebrow="Strength & movement" title="Train" />

        {active ? (
          <Pressable onPress={() => router.push('/workout-session')} style={({ pressed }) => [styles.resume, pressed && styles.pressed]}>
            <View style={styles.resumePulse}><Glyph name="dumbbell" color={palette.forest} size={20} /></View>
            <View style={{ flex: 1 }}>
              <Text style={styles.resumeTitle}>Workout in progress</Text>
              <Text style={styles.resumeMeta}>{active.name} · started {new Date(active.startedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</Text>
            </View>
            <Text style={styles.resumeAction}>Resume</Text>
          </Pressable>
        ) : (
          <Pressable onPress={startEmpty} style={({ pressed }) => [styles.startEmpty, pressed && styles.pressed]}>
            <View style={styles.startIcon}><Glyph name="plus" color={palette.forest} size={22} /></View>
            <View style={{ flex: 1 }}>
              <Text style={styles.startTitle}>Start empty workout</Text>
              <Text style={styles.startMeta}>Log sets as you go · rest timer · PR detection</Text>
            </View>
            <Glyph name="chevron" color={palette.lime} size={18} />
          </Pressable>
        )}

        <View style={styles.weekRow}>
          <WeekStat label="Workouts" value={String(week.workouts)} />
          <WeekStat label="Time" value={week.minutes ? formatDuration(week.minutes * 60) : '0m'} />
          <WeekStat label="Volume" value={`${week.volumeKg.toLocaleString()} kg`} />
        </View>

        <SectionTitle title="Routines" aside={`${routines.length} saved`} />
        {routines.length ? (
          <View style={styles.routineList}>
            {routines.map((routine) => {
              const expanded = expandedRoutineId === routine.id;
              const preview = routine.exercises
                .map((entry) => exerciseInfo(training, entry.exerciseId).name)
                .slice(0, 4)
                .join(' · ');
              return (
                <View key={routine.id} style={styles.routineCard}>
                  <Pressable
                    onPress={() => setExpandedRoutineId(expanded ? null : routine.id)}
                    style={styles.routineTop}>
                    <View style={{ flex: 1 }}>
                      {routine.folder ? <Text style={styles.routineFolder}>{routine.folder.toUpperCase()}</Text> : null}
                      <Text style={styles.routineName}>{routine.name}</Text>
                      <Text numberOfLines={2} style={styles.routinePreview}>
                        {preview || 'No exercises yet'}{routine.exercises.length > 4 ? ` +${routine.exercises.length - 4}` : ''}
                      </Text>
                    </View>
                    <Pressable onPress={() => startRoutine(routine)} style={({ pressed }) => [styles.routineStart, pressed && styles.pressed]}>
                      <Text style={styles.routineStartText}>Start</Text>
                    </Pressable>
                  </Pressable>
                  {expanded ? (
                    <View style={styles.routineActions}>
                      <RoutineAction label="Edit" onPress={() => editRoutine(routine.id)} />
                      <RoutineAction label="Duplicate" onPress={() => workouts.duplicateRoutine(routine.id)} />
                      <RoutineAction label="Delete" destructive onPress={() => confirmDelete(routine)} />
                    </View>
                  ) : null}
                </View>
              );
            })}
          </View>
        ) : (
          <EmptyState
            icon="dumbbell"
            title="No routines yet"
            body="Build your own program or import a template below. Routines remember your targets and last weights."
          />
        )}

        <Pressable onPress={() => editRoutine()} style={({ pressed }) => [styles.newRoutine, pressed && styles.pressed]}>
          <Glyph name="plus" color={palette.forest} size={17} />
          <Text style={styles.newRoutineText}>New routine</Text>
        </Pressable>

        <Pressable onPress={() => setShowTemplates((value) => !value)} style={styles.templateToggle}>
          <Text style={styles.templateToggleText}>
            {showTemplates ? 'Hide template routines' : 'Explore template routines'}
          </Text>
        </Pressable>
        {showTemplates ? (
          <View style={styles.templateList}>
            {TEMPLATE_ROUTINE_SEEDS.map((seed) => (
              <View key={seed.name} style={styles.templateCard}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.templateFolder}>{seed.folder.toUpperCase()}</Text>
                  <Text style={styles.templateName}>{seed.name}</Text>
                  <Text numberOfLines={1} style={styles.templateMeta}>
                    {seed.items.map((item) => findExercise(item.exerciseId)?.name ?? item.exerciseId).slice(0, 3).join(' · ')}
                    {seed.items.length > 3 ? ` +${seed.items.length - 3}` : ''}
                  </Text>
                </View>
                <Pressable
                  disabled={importedNames.has(seed.name)}
                  onPress={() => workouts.importTemplateRoutine(seed.name)}
                  style={({ pressed }) => [styles.templateAdd, importedNames.has(seed.name) && styles.templateAdded, pressed && styles.pressed]}>
                  <Text style={styles.templateAddText}>{importedNames.has(seed.name) ? 'Added' : 'Add'}</Text>
                </Pressable>
              </View>
            ))}
          </View>
        ) : null}

        {muscles.length ? (
          <>
            <SectionTitle title="Sets this week" aside="per muscle" />
            <View style={styles.muscleCard}>
              {muscles.map((entry) => (
                <View key={entry.muscle} style={styles.muscleRow}>
                  <Text style={styles.muscleName}>{entry.muscle}</Text>
                  <View style={styles.muscleTrack}>
                    <View style={[styles.muscleFill, { width: `${Math.min(1, entry.sets / 20) * 100}%` }]} />
                  </View>
                  <Text style={styles.muscleSets}>{entry.sets}</Text>
                </View>
              ))}
              <Text style={styles.muscleHint}>10–20 hard sets per muscle per week is a common hypertrophy guideline.</Text>
            </View>
          </>
        ) : null}

        <SectionTitle title="History" aside={history.length ? `${history.length} workouts` : undefined} />
        {history.length ? (
          <View style={styles.historyList}>
            {history.slice(0, 3).map((session) => (
              <Pressable
                key={session.id}
                onPress={() => router.push({ pathname: '/workout/[id]', params: { id: session.id } })}
                style={({ pressed }) => [styles.historyCard, pressed && styles.pressed]}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.historyName}>{session.name}</Text>
                  <Text style={styles.historyMeta}>
                    {new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}
                    {' · '}{formatDuration((session.durationMin ?? 0) * 60)}
                    {' · '}{(session.totalVolumeKg ?? 0).toLocaleString()} kg
                  </Text>
                </View>
                {session.records ? (
                  <View style={styles.historyPr}><Text style={styles.historyPrText}>{session.records} PR</Text></View>
                ) : null}
                <Glyph name="chevron" color={palette.muted} size={16} />
              </Pressable>
            ))}
            {history.length > 3 ? (
              <Pressable onPress={() => router.push('/workout-history')} style={styles.seeAll}>
                <Text style={styles.seeAllText}>See all workouts</Text>
              </Pressable>
            ) : null}
          </View>
        ) : (
          <EmptyState
            icon="chart"
            title="No workouts logged"
            body="Finish your first session and your history, records and charts appear here."
          />
        )}

        <Text style={styles.settingsLabel}>WORKOUT SETTINGS</Text>
        <View style={styles.settingsRow}>
          <Pressable
            onPress={() => {
              const index = REST_CHOICES.indexOf(training.defaultRestSec);
              workouts.setDefaultRest(REST_CHOICES[(index + 1) % REST_CHOICES.length]);
            }}
            style={styles.settingChip}>
            <Text style={styles.settingChipText}>
              Default rest: {training.defaultRestSec ? `${training.defaultRestSec}s` : 'off'}
            </Text>
          </Pressable>
          <Pressable
            onPress={() => workouts.setRpeEnabled(!training.rpeEnabled)}
            style={[styles.settingChip, training.rpeEnabled && styles.settingChipOn]}>
            <Text style={[styles.settingChipText, training.rpeEnabled && styles.settingChipTextOn]}>
              RPE: {training.rpeEnabled ? 'on' : 'off'}
            </Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function WeekStat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.weekStat}>
      <Text style={styles.weekValue}>{value}</Text>
      <Text style={styles.weekLabel}>{label.toUpperCase()}</Text>
    </View>
  );
}

function RoutineAction({ label, destructive, onPress }: { label: string; destructive?: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.routineAction, pressed && styles.pressed]}>
      <Text style={[styles.routineActionText, destructive && { color: '#B64B45' }]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { paddingHorizontal: space.md, paddingBottom: 28 },
  pressed: { opacity: 0.85, transform: [{ scale: 0.995 }] },
  resume: { minHeight: 76, backgroundColor: palette.lime, borderRadius: radius.md, flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 14, marginBottom: 12 },
  resumePulse: { width: 42, height: 42, borderRadius: 21, backgroundColor: '#D3F694', alignItems: 'center', justifyContent: 'center' },
  resumeTitle: { color: palette.forest, fontFamily: type.demi, fontSize: 14 },
  resumeMeta: { color: '#3E5030', fontFamily: type.regular, fontSize: 10.5, marginTop: 2 },
  resumeAction: { color: palette.forest, fontFamily: type.demi, fontSize: 13 },
  startEmpty: { minHeight: 76, backgroundColor: palette.forest, borderRadius: radius.md, flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 14, marginBottom: 12 },
  startIcon: { width: 42, height: 42, borderRadius: 21, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center' },
  startTitle: { color: palette.white, fontFamily: type.demi, fontSize: 14.5 },
  startMeta: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 10.5, marginTop: 2 },
  weekRow: { flexDirection: 'row', gap: 8, marginBottom: 22 },
  weekStat: { flex: 1, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingVertical: 13, alignItems: 'center' },
  weekValue: { color: palette.ink, fontFamily: type.demi, fontSize: 16 },
  weekLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 1, marginTop: 3 },
  routineList: { gap: 9 },
  routineCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14 },
  routineTop: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  routineFolder: { color: palette.limeDark, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 1, marginBottom: 3 },
  routineName: { color: palette.ink, fontFamily: type.demi, fontSize: 15 },
  routinePreview: { color: palette.muted, fontFamily: type.regular, fontSize: 10.5, lineHeight: 15, marginTop: 4 },
  routineStart: { height: 38, paddingHorizontal: 16, borderRadius: radius.pill, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  routineStartText: { color: palette.lime, fontFamily: type.demi, fontSize: 12 },
  routineActions: { flexDirection: 'row', gap: 8, marginTop: 12, borderTopWidth: 1, borderTopColor: palette.line, paddingTop: 12 },
  routineAction: { flex: 1, height: 36, borderRadius: radius.sm, borderWidth: 1, borderColor: palette.line, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.canvas },
  routineActionText: { color: palette.ink, fontFamily: type.medium, fontSize: 11.5 },
  newRoutine: { height: 48, borderRadius: radius.md, borderWidth: 1.5, borderColor: palette.forest, borderStyle: 'dashed', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, marginTop: 10 },
  newRoutineText: { color: palette.forest, fontFamily: type.demi, fontSize: 13 },
  templateToggle: { alignItems: 'center', paddingVertical: 14 },
  templateToggleText: { color: palette.coral, fontFamily: type.demi, fontSize: 11.5 },
  templateList: { gap: 8, marginBottom: 6 },
  templateCard: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 13 },
  templateFolder: { color: palette.limeDark, fontFamily: type.demi, fontSize: 8.5, letterSpacing: 1 },
  templateName: { color: palette.ink, fontFamily: type.demi, fontSize: 13.5, marginTop: 2 },
  templateMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 3 },
  templateAdd: { height: 34, paddingHorizontal: 14, borderRadius: radius.pill, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  templateAdded: { opacity: 0.5 },
  templateAddText: { color: palette.limeDark, fontFamily: type.demi, fontSize: 11.5 },
  muscleCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14, marginBottom: 22 },
  muscleRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 9 },
  muscleName: { width: 86, color: palette.ink, fontFamily: type.medium, fontSize: 11, textTransform: 'capitalize' },
  muscleTrack: { flex: 1, height: 6, borderRadius: radius.pill, backgroundColor: '#E8ECE3', overflow: 'hidden' },
  muscleFill: { height: '100%', borderRadius: radius.pill, backgroundColor: palette.limeDark },
  muscleSets: { width: 24, textAlign: 'right', color: palette.ink, fontFamily: type.demi, fontSize: 11 },
  muscleHint: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 14, marginTop: 6 },
  historyList: { gap: 8, marginBottom: 8 },
  historyCard: { minHeight: 64, flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 14 },
  historyName: { color: palette.ink, fontFamily: type.demi, fontSize: 13.5 },
  historyMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 10.5, marginTop: 3 },
  historyPr: { backgroundColor: palette.softLime, borderRadius: radius.pill, paddingHorizontal: 9, paddingVertical: 4 },
  historyPrText: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10 },
  seeAll: { alignItems: 'center', paddingVertical: 10 },
  seeAllText: { color: palette.forest, fontFamily: type.demi, fontSize: 12 },
  settingsLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.3, marginTop: 22, marginBottom: 8 },
  settingsRow: { flexDirection: 'row', gap: 8 },
  settingChip: { height: 38, paddingHorizontal: 14, borderRadius: radius.pill, borderWidth: 1, borderColor: palette.line, backgroundColor: palette.paper, alignItems: 'center', justifyContent: 'center' },
  settingChipOn: { backgroundColor: palette.forest, borderColor: palette.forest },
  settingChipText: { color: palette.ink, fontFamily: type.medium, fontSize: 11.5 },
  settingChipTextOn: { color: palette.lime, fontFamily: type.demi },
});
