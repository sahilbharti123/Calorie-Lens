import { type Href, useRouter } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { Glyph } from '@/src/components/glyph';
import { Card, GhostButton, Reveal, Screen, ScreenHeader, Well } from '@/src/components/ui';
import { palette, radius, space, text } from '@/src/theme';

/**
 * The coach is not built yet, and this screen says so plainly.
 *
 * It used to show arithmetic about your own logs under the name "Coach", which
 * set the wrong expectation: people read a coach tab as something that answers
 * back. That arithmetic was not thrown away — it now lives in Progress, next to
 * the charts it describes, where it is honestly labelled as a reading of your
 * own numbers rather than advice from anyone.
 */
const WHAT_IT_WILL_DO = [
  {
    icon: 'mic',
    title: 'Ask it anything about your logs',
    body: 'In your own words — “why did I stall this week”, “what should I eat tonight to hit protein” — and get an answer that cites the entries it used.',
  },
  {
    icon: 'target',
    title: 'Adjust the plan as you go',
    body: 'Targets that move with your actual adherence and training load instead of a number set once during onboarding.',
  },
  {
    icon: 'dumbbell',
    title: 'Programme your training',
    body: 'Progression suggested from the sets you logged, and a session built around the time you actually have.',
  },
] as const;

export default function CoachScreen() {
  const router = useRouter();

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader eyebrow="In development" title="Coach" />

        <Reveal index={1}>
          <Card glow raised>
            <View style={styles.badge}>
              <Glyph color={palette.lime} name="spark" size={26} />
            </View>
            <Text style={styles.title}>Coming soon</Text>
            <Text style={styles.body}>
              A real coach you can talk to is being built. It is not here yet, so rather than
              dress up a page that cannot answer you, this tab stays empty until it can.
            </Text>
          </Card>
        </Reveal>

        <Reveal index={2} style={styles.block}>
          <Text style={styles.label}>WHAT IT WILL DO</Text>
          <View style={styles.list}>
            {WHAT_IT_WILL_DO.map((item) => (
              <Well key={item.title} style={styles.row}>
                <View style={styles.rowIcon}>
                  <Glyph color={palette.lime} name={item.icon} size={15} />
                </View>
                <View style={styles.rowCopy}>
                  <Text style={styles.rowTitle}>{item.title}</Text>
                  <Text style={styles.rowBody}>{item.body}</Text>
                </View>
              </Well>
            ))}
          </View>
        </Reveal>

        <Reveal index={3} style={styles.block}>
          <Well style={styles.moved}>
            <Glyph color={palette.inkMid} name="chart" size={15} />
            <Text style={styles.movedText}>
              The read on your own numbers — protein and calorie adherence, training volume, how
              wide your estimates are running — moved to Progress.
            </Text>
          </Well>
        </Reveal>

        <Reveal index={4} style={styles.block}>
          <GhostButton
            icon="chart"
            label="Open Progress"
            onPress={() => router.push('/(tabs)/progress' as Href)}
          />
        </Reveal>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  badge: {
    width: 56,
    height: 56,
    borderRadius: radius.md,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: { ...text.headline, color: palette.ink, marginTop: space.md },
  body: { ...text.body, color: palette.inkMid, marginTop: space.xs },

  block: { marginTop: space.lg },
  label: { ...text.label, color: palette.inkLow, marginBottom: space.sm },
  list: { gap: space.sm },
  row: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm, padding: 14 },
  rowIcon: {
    width: 30,
    height: 30,
    borderRadius: radius.sm,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rowCopy: { flex: 1 },
  rowTitle: { ...text.value, color: palette.ink },
  rowBody: { ...text.caption, color: palette.inkMid, marginTop: space.xs },

  moved: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm, padding: 14 },
  movedText: { ...text.caption, color: palette.inkMid, flex: 1 },
});
