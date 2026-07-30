import { useRouter } from 'expo-router';
import { useMemo } from 'react';
import { FlatList, StyleSheet, Text, View } from 'react-native';

import { Glyph } from '@/src/components/glyph';
import {
  Card,
  EmptyState,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  Tap,
} from '@/src/components/ui';
import { completedSessions, exerciseInfo, formatDuration } from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { palette, space, text } from '@/src/theme';

export default function WorkoutHistoryScreen() {
  const router = useRouter();
  const { data } = useApp();
  const training = data.training;
  const history = useMemo(() => completedSessions(training), [training]);

  return (
    <Screen edges={['bottom']}>
      <Reveal style={styles.fill}>
        <FlatList
          contentContainerStyle={styles.list}
          data={history}
          keyExtractor={(session) => session.id}
          ListEmptyComponent={(
            <View style={styles.emptyWrap}>
              <EmptyState
                action={<PrimaryButton icon="chevronLeft" label="Back to training" onPress={() => router.back()} />}
                body="Every session you finish lands here with its volume, sets and personal records intact."
                icon="calendar"
                title="Your logbook starts today"
              />
            </View>
          )}
          renderItem={({ item: session, index }) => (
            <Tap
              accessibilityLabel={`Open ${session.name}`}
              onPress={() => router.push({ pathname: '/workout/[id]', params: { id: session.id } })}
              scaleTo={0.985}>
              <Card glow={index === 0} raised={index === 0}>
                <View style={styles.cardTop}>
                  <View style={styles.icon}>
                    <Glyph color={palette.lime} name="dumbbell" size={16} />
                  </View>
                  <View style={styles.cardCopy}>
                    <Text numberOfLines={1} style={styles.name}>{session.name}</Text>
                    <Text style={styles.date}>
                      {new Date(session.startedAt).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}
                    </Text>
                  </View>
                  {session.records ? <Pill icon="trophy" label={`${session.records} PR`} tone="accent" /> : null}
                  <Glyph color={palette.inkLow} name="chevron" size={15} />
                </View>

                <Text numberOfLines={2} style={styles.exercises}>
                  {session.exercises.map((entry) => exerciseInfo(training, entry.exerciseId).name).join(' · ')}
                </Text>

                <View style={styles.metaRow}>
                  <Pill icon="timer" label={formatDuration((session.durationMin ?? 0) * 60)} />
                  <Pill icon="dumbbell" label={`${session.totalSets ?? 0} sets`} />
                  <Pill icon="scale" label={`${(session.totalVolumeKg ?? 0).toLocaleString()} kg`} />
                  {session.calories ? <Pill icon="flame" label={`~${session.calories} kcal`} /> : null}
                </View>
              </Card>
            </Tap>
          )}
        />
      </Reveal>
    </Screen>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },

  list: { paddingHorizontal: space.md, paddingTop: 10, paddingBottom: space.tabClearance, gap: 10 },
  emptyWrap: { paddingTop: space.lg },

  cardTop: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  icon: {
    width: 34,
    height: 34,
    borderRadius: 12,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardCopy: { flex: 1 },
  name: { ...text.row, fontSize: 15, color: palette.ink },
  date: { ...text.caption, fontSize: 11, color: palette.inkLow, marginTop: 3 },

  exercises: { ...text.caption, color: palette.inkMid, marginTop: 12 },
  metaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 12,
    borderTopWidth: 1,
    borderTopColor: palette.line,
    paddingTop: 12,
  },
});
