import { type Href, useRouter } from 'expo-router';
import { useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { Glyph } from '@/src/components/glyph';
import {
  Card,
  CountUp,
  EmptyState,
  ListRow,
  PrimaryButton,
  Reveal,
  Screen,
  ScreenHeader,
  SectionTitle,
  VoiceBar,
  Well,
} from '@/src/components/ui';
import { buildInsights, loggedDayCount, type Insight } from '@/src/lib/insights';
import { personalDailyNudge } from '@/src/lib/personalization';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, tabular, text } from '@/src/theme';

/**
 * Amber is the only colour that carries meaning on this screen — lime is the
 * app's accent and grey is simply quiet, so a warm chip always means "look".
 */
const TONE_COLOR: Record<Insight['tone'], string> = {
  good: palette.lime,
  watch: palette.warn,
  neutral: palette.inkMid,
};

export default function CoachScreen() {
  const router = useRouter();
  const { data, today } = useApp();
  const [openId, setOpenId] = useState<string | null>(null);

  const nudge = personalDailyNudge(data, today);
  const insights = buildInsights(data, today);
  /** `buildInsights` ranks by importance, so the first item leads the screen. */
  const hero: Insight | undefined = insights[0];
  const rest = insights.slice(1);
  const heroAction = hero?.action;
  const heroTone = hero ? TONE_COLOR[hero.tone] : palette.lime;
  const daysLogged = loggedDayCount(data, 7);

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader eyebrow={`${daysLogged} of the last 7 days logged`} title="Coach" />

        {/* ---------- Hero: the strongest thing the logs say today ---------- */}
        <Reveal index={1}>
          <Card glow raised>
            <View style={styles.heroHead}>
              <View style={[styles.heroIcon, { backgroundColor: `${heroTone}1A` }]}>
                <Glyph color={heroTone} name={hero?.icon ?? 'spark'} size={18} />
              </View>
              <Text style={[styles.heroEyebrow, { color: heroTone }]}>
                {hero ? 'WHAT YOUR LOGS SHOW' : "TODAY'S FOCUS"}
              </Text>
            </View>
            <Text style={styles.heroTitle}>{hero?.title ?? nudge.title}</Text>
            <Text style={styles.heroBody}>{hero?.body ?? nudge.body}</Text>
            {heroAction ? (
              <PrimaryButton
                compact
                icon={heroAction.route === '/quick-log' ? 'mic' : 'settings'}
                label={heroAction.label}
                onPress={() => router.push(heroAction.route as Href)}
                style={styles.heroButton}
              />
            ) : null}
          </Card>
        </Reveal>

        {/* ---------- Today's focus — quieter once an insight leads ---------- */}
        {hero ? (
          <Reveal index={2} style={styles.focusWrap}>
            <Well style={styles.focus}>
              <View style={styles.focusIcon}>
                <Glyph color={palette.lime} name="spark" size={15} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.focusLabel}>TODAY&apos;S FOCUS</Text>
                <Text style={styles.focusTitle}>{nudge.title}</Text>
                <Text style={styles.focusBody}>{nudge.body}</Text>
              </View>
            </Well>
          </Reveal>
        ) : null}

        {/* ---------- The plan ---------- */}
        <SectionTitle title="Your plan" />
        <Reveal index={3}>
          <Card>
            <View style={styles.planRow}>
              <View style={styles.planCell}>
                <Text style={styles.planLabel}>DAILY CALORIES</Text>
                <CountUp style={styles.planValue} value={data.goals.calories} />
              </View>
              <View style={styles.planDivider} />
              <View style={styles.planCell}>
                <Text style={styles.planLabel}>PROTEIN</Text>
                <CountUp style={styles.planValue} suffix=" g" value={data.goals.protein} />
              </View>
              <View style={styles.planDivider} />
              <View style={styles.planCell}>
                <Text style={styles.planLabel}>TRAINING</Text>
                <CountUp style={styles.planValue} suffix=" min" value={data.goals.weeklyWorkoutMinutes} />
              </View>
            </View>

            {data.plan.summary ? (
              <Well style={styles.planWell}>
                <Text style={styles.planSummary}>{data.plan.summary}</Text>
                {data.plan.method ? <Text style={styles.planMethod}>{data.plan.method}</Text> : null}
              </Well>
            ) : null}

            <View style={styles.planRows}>
              <ListRow
                detail="Recalculates from your profile"
                icon="target"
                onPress={() => router.push('/settings' as Href)}
                title="Adjust goals and targets"
              />
              <ListRow
                detail="Tone, injuries and food notes the coach remembers"
                icon="settings"
                last
                onPress={() => router.push('/account' as Href)}
                title="Set your coaching preferences"
              />
            </View>
          </Card>
        </Reveal>

        {/* ---------- The rest of the read: one row each, opened on tap ---------- */}
        {rest.length ? (
          <>
            <SectionTitle title="What else your logs show" />
            <Reveal index={4}>
              <Card padded={false} style={styles.listCard}>
                {rest.map((insight, index) => {
                  const open = openId === insight.id;
                  return (
                    <View
                      key={insight.id}
                      style={index < rest.length - 1 ? styles.insightRow : undefined}>
                      <ListRow
                        accent={TONE_COLOR[insight.tone]}
                        detail={insight.summary}
                        icon={insight.icon}
                        last
                        onPress={() => setOpenId(open ? null : insight.id)}
                        right={
                          <View style={open ? styles.chevronOpen : undefined}>
                            <Glyph color={palette.inkLow} name="chevronDown" size={15} />
                          </View>
                        }
                        title={insight.title}
                      />
                      {open ? (
                        <Reveal from={6}>
                          <Text style={styles.insightBody}>{insight.body}</Text>
                        </Reveal>
                      ) : null}
                    </View>
                  );
                })}
              </Card>
            </Reveal>
          </>
        ) : null}

        {hero ? null : (
          <>
            <SectionTitle title="What your logs show" />
            <Reveal index={4}>
              <EmptyState
                action={
                  <PrimaryButton
                    icon="mic"
                    label="Log your first meal"
                    onPress={() => router.push('/quick-log' as Href)}
                  />
                }
                body="Log three days of meals and this page starts reading patterns back to you — protein adherence, calorie drift, training volume, and how wide your estimates are running."
                icon="spark"
                title="Not enough logged yet"
              />
            </Reveal>
          </>
        )}

        {hero ? (
          <Reveal index={5} style={styles.voice}>
            <VoiceBar title="Log something now" />
          </Reveal>
        ) : null}

        <Reveal index={6}>
          <Text style={styles.footnote}>
            Insights are arithmetic on your own entries, not medical advice. Everything on this
            screen is computed on this device — it costs nothing to run and works offline. Review
            targets against a two-to-four week trend.
          </Text>
        </Reveal>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  /* hero */
  heroHead: { flexDirection: 'row', alignItems: 'center', gap: space.sm, marginBottom: space.sm },
  heroIcon: {
    width: 34,
    height: 34,
    borderRadius: radius.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroEyebrow: { ...text.label },
  heroTitle: { ...text.headline, color: palette.ink },
  heroBody: { ...text.body, color: palette.inkMid, marginTop: space.xs },
  heroButton: { marginTop: space.md, alignSelf: 'flex-start' },

  /* today's focus */
  focusWrap: { marginTop: space.sm },
  focus: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm },
  focusIcon: {
    width: 30,
    height: 30,
    borderRadius: radius.sm,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  focusLabel: { ...text.label, color: palette.lime, marginBottom: space.xs },
  focusTitle: { ...text.value, color: palette.ink },
  focusBody: { ...text.caption, color: palette.inkMid, marginTop: space.xs },

  /* plan */
  planRow: { flexDirection: 'row', alignItems: 'center' },
  planCell: { flex: 1, alignItems: 'center' },
  planDivider: { width: 1, alignSelf: 'stretch', backgroundColor: palette.line },
  planLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1, color: palette.inkLow, marginBottom: space.xs },
  planValue: { ...text.headline, fontSize: 17, color: palette.ink, textAlign: 'center', ...tabular },
  planWell: { marginTop: space.md },
  planSummary: { ...text.body, fontSize: 12.5, color: palette.ink },
  planMethod: { ...text.caption, fontSize: 10.5, color: palette.inkLow, marginTop: space.xs },
  planRows: { marginTop: space.sm },

  /* insight list */
  listCard: { paddingHorizontal: 14 },
  insightRow: { borderBottomWidth: 1, borderBottomColor: palette.line },
  insightBody: { ...text.body, color: palette.inkMid, paddingBottom: space.sm },
  chevronOpen: { transform: [{ rotate: '180deg' }] },

  voice: { marginTop: space.md },

  footnote: {
    ...text.caption,
    color: palette.inkMid,
    textAlign: 'center',
    marginTop: space.lg,
    paddingHorizontal: space.sm,
  },
});
