import { useRouter } from 'expo-router';
import { useMemo } from 'react';
import { FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { completedSessions, exerciseInfo, formatDuration } from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';

export default function WorkoutHistoryScreen() {
  const router = useRouter();
  const { data } = useApp();
  const training = data.training;
  const history = useMemo(() => completedSessions(training), [training]);

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <FlatList
        data={history}
        keyExtractor={(session) => session.id}
        contentContainerStyle={styles.list}
        renderItem={({ item: session }) => (
          <Pressable
            onPress={() => router.push({ pathname: '/workout/[id]', params: { id: session.id } })}
            style={({ pressed }) => [styles.card, pressed && styles.pressed]}>
            <View style={styles.cardTop}>
              <View style={{ flex: 1 }}>
                <Text style={styles.name}>{session.name}</Text>
                <Text style={styles.date}>
                  {new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}
                </Text>
              </View>
              {session.records ? (
                <View style={styles.pr}><Text style={styles.prText}>{session.records} PR</Text></View>
              ) : null}
              <Glyph name="chevron" color={palette.muted} size={16} />
            </View>
            <Text numberOfLines={2} style={styles.exercises}>
              {session.exercises.map((entry) => exerciseInfo(training, entry.exerciseId).name).join(' · ')}
            </Text>
            <Text style={styles.meta}>
              {formatDuration((session.durationMin ?? 0) * 60)} · {session.totalSets ?? 0} sets · {(session.totalVolumeKg ?? 0).toLocaleString()} kg
              {session.calories ? ` · ~${session.calories} kcal` : ''}
            </Text>
          </Pressable>
        )}
        ListEmptyComponent={(
          <View style={styles.empty}>
            <Text style={styles.emptyText}>No workouts yet. Your finished sessions appear here.</Text>
          </View>
        )}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  list: { padding: space.md, gap: 9, paddingBottom: 28 },
  pressed: { opacity: 0.85 },
  card: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14 },
  cardTop: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  name: { color: palette.ink, fontFamily: type.demi, fontSize: 14.5 },
  date: { color: palette.muted, fontFamily: type.regular, fontSize: 10.5, marginTop: 2 },
  pr: { backgroundColor: palette.softLime, borderRadius: radius.pill, paddingHorizontal: 9, paddingVertical: 4 },
  prText: { color: palette.limeDark, fontFamily: type.demi, fontSize: 10 },
  exercises: { color: palette.muted, fontFamily: type.regular, fontSize: 11, lineHeight: 16, marginTop: 8 },
  meta: { color: palette.limeDark, fontFamily: type.medium, fontSize: 10.5, marginTop: 7 },
  empty: { alignItems: 'center', paddingVertical: 44 },
  emptyText: { color: palette.muted, fontFamily: type.regular, fontSize: 12 },
});
