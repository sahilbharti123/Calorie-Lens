import { useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { EmptyState, ScreenHeader, SectionTitle, VoiceBar } from '@/src/components/ui';
import { workoutTotals } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';

export default function TrainScreen() {
  const router = useRouter();
  const { today, removeWorkout } = useApp();
  const totals = workoutTotals(today);

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader eyebrow="Movement" title="Train" />

        <View style={styles.hero}>
          <View style={styles.heroIcon}><Glyph name="dumbbell" color={palette.forest} size={26} /></View>
          <View style={{ flex: 1 }}>
            <Text style={styles.heroLabel}>TODAY’S TRAINING</Text>
            <Text style={styles.heroValue}>{totals.minutes || 0} <Text style={styles.heroUnit}>minutes</Text></Text>
            <Text style={styles.heroMeta}>~{totals.calories} active kcal midpoint</Text>
          </View>
          <Pressable
            onPress={() => router.push({ pathname: '/quick-log', params: { prefill: '30 min strength workout' } })}
            style={styles.heroAdd}>
            <Glyph name="plus" color={palette.lime} size={20} />
          </Pressable>
        </View>

        <VoiceBar label="Say “45 min hard strength workout”" />

        <View style={styles.recovery}>
          <Text style={styles.recoveryEyebrow}>RECOVERY CUE</Text>
          <Text style={styles.recoveryTitle}>
            {today.sleepHours >= 7 ? 'You’re ready to push.' : 'Keep the first set easy.'}
          </Text>
          <Text style={styles.recoveryBody}>
            {today.sleepHours
              ? `${today.sleepHours} hours of sleep logged. Use the warm-up to decide today’s intensity.`
              : 'Sync sleep from your health app or log it by voice for a more useful training cue.'}
          </Text>
        </View>

        <SectionTitle title="Workout log" aside={`${today.workouts.length} sessions`} />
        {today.workouts.length ? (
          <View style={styles.workoutList}>
            {today.workouts.map((workout, index) => (
              <View key={workout.id} style={[styles.workout, index < today.workouts.length - 1 && styles.workoutBorder]}>
                <View style={styles.workoutIndex}><Text style={styles.workoutIndexText}>{String(index + 1).padStart(2, '0')}</Text></View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.workoutName}>{workout.name}</Text>
                  <Text style={styles.workoutMeta}>
                    {workout.durationMin} min · {workout.intensity} · {workout.calorieLow ?? workout.calories}–{workout.calorieHigh ?? workout.calories} active kcal
                  </Text>
                </View>
                <Pressable hitSlop={10} onPress={() => removeWorkout(workout.id)}>
                  <Glyph name="trash" color={palette.muted} size={18} />
                </Pressable>
              </View>
            ))}
          </View>
        ) : (
          <EmptyState
            icon="dumbbell"
            title="No workout yet"
            body="Log gym sessions, runs, walks, yoga, or any movement in one short sentence."
          />
        )}

        <Text style={styles.templatesLabel}>QUICK STARTS</Text>
        <View style={styles.templates}>
          {['45 min hard strength workout', '30 min brisk walk', '20 min light yoga'].map((template) => (
            <Pressable
              key={template}
              onPress={() => router.push({ pathname: '/quick-log', params: { prefill: template } })}
              style={styles.template}>
              <Text style={styles.templateText}>{template}</Text>
              <Glyph name="chevron" color={palette.muted} size={15} />
            </Pressable>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { paddingHorizontal: space.md, paddingBottom: 28 },
  hero: { minHeight: 132, backgroundColor: palette.forest, borderRadius: radius.lg, padding: 18, flexDirection: 'row', alignItems: 'center', gap: 13, marginBottom: 12 },
  heroIcon: { width: 50, height: 50, borderRadius: 16, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center' },
  heroLabel: { color: '#AEB9B0', fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2 },
  heroValue: { color: palette.white, fontFamily: type.demi, fontSize: 29, letterSpacing: -1, marginTop: 3 },
  heroUnit: { color: '#AEB9B0', fontFamily: type.medium, fontSize: 12 },
  heroMeta: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 10, marginTop: 2 },
  heroAdd: { width: 38, height: 38, borderRadius: 19, backgroundColor: '#2B382F', alignItems: 'center', justifyContent: 'center' },
  recovery: { backgroundColor: palette.softCoral, borderRadius: radius.md, padding: 16, marginVertical: 22 },
  recoveryEyebrow: { color: palette.coral, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.2 },
  recoveryTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 17, marginTop: 4 },
  recoveryBody: { color: palette.ink, fontFamily: type.regular, fontSize: 11, lineHeight: 17, marginTop: 5 },
  workoutList: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 14 },
  workout: { minHeight: 70, flexDirection: 'row', alignItems: 'center', gap: 11 },
  workoutBorder: { borderBottomWidth: 1, borderBottomColor: palette.line },
  workoutIndex: { width: 34, height: 34, borderRadius: 11, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  workoutIndexText: { color: palette.limeDark, fontFamily: type.demi, fontSize: 11 },
  workoutName: { color: palette.ink, fontFamily: type.demi, fontSize: 14 },
  workoutMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 3, textTransform: 'capitalize' },
  templatesLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 10, letterSpacing: 1.3, marginTop: 24, marginBottom: 8 },
  templates: { gap: 8 },
  template: { minHeight: 50, paddingHorizontal: 14, borderRadius: radius.sm, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  templateText: { color: palette.ink, fontFamily: type.medium, fontSize: 12 },
});
