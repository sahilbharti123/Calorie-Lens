import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import { palette, radius, space, type } from '@/src/theme';

export function ScreenHeader({ eyebrow, title, action }: { eyebrow: string; title: string; action?: React.ReactNode }) {
  return (
    <View style={styles.header}>
      <View style={{ flex: 1 }}>
        <Text style={styles.eyebrow}>{eyebrow.toUpperCase()}</Text>
        <Text style={styles.title}>{title}</Text>
      </View>
      {action}
    </View>
  );
}

export function SectionTitle({ title, aside }: { title: string; aside?: string }) {
  return (
    <View style={styles.sectionTitle}>
      <Text style={styles.sectionTitleText}>{title}</Text>
      {aside ? <Text style={styles.sectionAside}>{aside}</Text> : null}
    </View>
  );
}

export function ProgressBar({ value, color = palette.lime }: { value: number; color?: string }) {
  return (
    <View style={styles.progressTrack}>
      <View style={[styles.progressFill, {
        width: `${Math.max(0, Math.min(value, 1)) * 100}%`,
        backgroundColor: color,
      }]} />
    </View>
  );
}

export function Metric({
  icon,
  label,
  value,
  detail,
}: {
  icon: GlyphName;
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <View style={styles.metric}>
      <View style={styles.metricIcon}><Glyph name={icon} color={palette.forest} size={18} /></View>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
      {detail ? <Text style={styles.metricDetail}>{detail}</Text> : null}
    </View>
  );
}

export function VoiceBar({ label = 'Tell me what you ate or did' }: { label?: string }) {
  const router = useRouter();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel="Open voice quick log"
      onPress={() => {
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        router.push('/quick-log');
      }}
      style={({ pressed }) => [styles.voiceBar, pressed && { transform: [{ scale: 0.99 }] }]}>
      <View style={styles.voiceIcon}><Glyph name="mic" color={palette.lime} size={21} /></View>
      <View style={{ flex: 1 }}>
        <Text style={styles.voiceLabel}>QUICK LOG</Text>
        <Text numberOfLines={1} style={styles.voiceText}>{label}</Text>
      </View>
      <Glyph name="chevron" color={palette.lime} size={18} />
    </Pressable>
  );
}

export function EmptyState({ icon, title, body }: { icon: GlyphName; title: string; body: string }) {
  return (
    <View style={styles.empty}>
      <Glyph name={icon} color={palette.muted} size={24} />
      <Text style={styles.emptyTitle}>{title}</Text>
      <Text style={styles.emptyBody}>{body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', paddingTop: 8, marginBottom: space.lg },
  eyebrow: { color: palette.muted, fontFamily: type.demi, fontSize: 11, letterSpacing: 1.5, marginBottom: 4 },
  title: { color: palette.ink, fontFamily: type.demi, fontSize: 30, letterSpacing: -1.1 },
  sectionTitle: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 },
  sectionTitleText: { color: palette.ink, fontFamily: type.demi, fontSize: 19, letterSpacing: -0.4 },
  sectionAside: { color: palette.muted, fontFamily: type.medium, fontSize: 12 },
  progressTrack: { height: 5, borderRadius: radius.pill, backgroundColor: '#313D34', overflow: 'hidden' },
  progressFill: { height: '100%', borderRadius: radius.pill },
  metric: { flex: 1, minHeight: 126, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 13 },
  metricIcon: { width: 32, height: 32, borderRadius: 10, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  metricLabel: { color: palette.muted, fontFamily: type.medium, fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.7 },
  metricValue: { color: palette.ink, fontFamily: type.demi, fontSize: 20, marginTop: 2 },
  metricDetail: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 1 },
  voiceBar: { backgroundColor: palette.forest, borderRadius: radius.md, minHeight: 72, paddingHorizontal: 14, alignItems: 'center', flexDirection: 'row', gap: 12 },
  voiceIcon: { width: 42, height: 42, borderRadius: 21, backgroundColor: '#26332A', alignItems: 'center', justifyContent: 'center' },
  voiceLabel: { color: palette.lime, fontFamily: type.demi, fontSize: 10, letterSpacing: 1.4, marginBottom: 2 },
  voiceText: { color: palette.white, fontFamily: type.medium, fontSize: 14 },
  empty: { alignItems: 'center', justifyContent: 'center', paddingVertical: 28, paddingHorizontal: 22, backgroundColor: palette.paper, borderRadius: radius.md, borderWidth: 1, borderColor: palette.line },
  emptyTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 15, marginTop: 10 },
  emptyBody: { color: palette.muted, fontFamily: type.regular, fontSize: 12, lineHeight: 18, textAlign: 'center', marginTop: 4 },
});
