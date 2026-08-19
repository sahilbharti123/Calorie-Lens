import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import { useRouter } from 'expo-router';
import React, { useEffect } from 'react';
import {
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
  type StyleProp,
  type TextStyle,
  type ViewStyle,
} from 'react-native';
import Animated, {
  Easing,
  interpolateColor,
  useAnimatedProps,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { SafeAreaView, type Edge } from 'react-native-safe-area-context';
import Svg, {
  Circle,
  Defs,
  LinearGradient as SvgGradient,
  RadialGradient,
  Stop,
} from 'react-native-svg';
import { Glyph, type GlyphName } from '@/src/components/glyph';
import { useReducedMotion } from '@/src/lib/accessibility';
import type { MealSlot } from '@/src/types';
import {
  alpha,
  gradient,
  motion,
  palette,
  radius,
  shadow,
  space,
  tabular,
  text,
} from '@/src/theme';
const AnimatedCircle = Animated.createAnimatedComponent(Circle);
/* ------------------------------------------------------------------ *
 * Layout
 * ------------------------------------------------------------------ */
/**
 * Every screen sits on this. Paints the canvas and the two ambient glows that
 * stop the dark background from reading as flat black.
 */
export function Screen({
  children,
  edges = ['top'],
  glow = true,
  style,
}: React.PropsWithChildren<{ edges?: Edge[]; glow?: boolean; style?: StyleProp<ViewStyle> }>) {
  return (
    <View style={styles.canvas}>
      {glow ? <AmbientGlow /> : null}
      <SafeAreaView edges={edges} style={[styles.safe, style]}>
        {children}
      </SafeAreaView>
    </View>
  );
}
/**
 * Two soft light sources behind the content. Drawn as radial gradients rather
 * than tinted circles — a flat circle on a near-black canvas reads as a shape,
 * which is the opposite of the intent.
 */
function AmbientGlow() {
  return (
    <Svg height="100%" pointerEvents="none" style={StyleSheet.absoluteFill} width="100%">
      <Defs>
        <RadialGradient cx="50%" cy="50%" id="ambient-lime" r="50%">
          <Stop offset="0%" stopColor={palette.lime} stopOpacity={0.11} />
          <Stop offset="50%" stopColor={palette.lime} stopOpacity={0.028} />
          <Stop offset="100%" stopColor={palette.lime} stopOpacity={0} />
        </RadialGradient>
        <RadialGradient cx="50%" cy="50%" id="ambient-blue" r="50%">
          <Stop offset="0%" stopColor={palette.info} stopOpacity={0.09} />
          <Stop offset="55%" stopColor={palette.info} stopOpacity={0.018} />
          <Stop offset="100%" stopColor={palette.info} stopOpacity={0} />
        </RadialGradient>
      </Defs>
      <Circle cx="92%" cy="1%" fill="url(#ambient-lime)" r="250" />
      <Circle cx="2%" cy="80%" fill="url(#ambient-blue)" r="220" />
    </Svg>
  );
}

export function ScreenHeader({
  eyebrow,
  title,
  action,
  style,
}: {
  eyebrow?: string;
  title: string;
  action?: React.ReactNode;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <View style={[styles.header, style]}>
      <View style={{ flex: 1 }}>
        {eyebrow ? <Text style={styles.eyebrow}>{eyebrow.toUpperCase()}</Text> : null}
        <Text accessibilityRole="header" style={styles.headerTitle}>{title}</Text>
      </View>
      {action}
    </View>
  );
}
export function SectionTitle({
  title,
  aside,
  onPressAside,
  style,
}: {
  title: string;
  aside?: string;
  onPressAside?: () => void;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <View style={[styles.sectionTitle, style]}>
      <Text style={styles.sectionTitleText}>{title}</Text>
      {aside ? (
        onPressAside ? (
          <Tap onPress={onPressAside} scaleTo={0.94}>
            <Text style={[styles.sectionAside, { color: palette.lime }]}>{aside}</Text>
          </Tap>
        ) : (
          <Text style={styles.sectionAside}>{aside}</Text>
        )
      ) : null}
    </View>
  );
}
/* ------------------------------------------------------------------ *
 * Motion primitives
 * ------------------------------------------------------------------ */
/** Compatibility wrapper: content appears immediately without page-load choreography. */
export function Reveal({
  children,
  index = 0,
  from = 14,
  style,
}: React.PropsWithChildren<{ index?: number; from?: number; style?: StyleProp<ViewStyle> }>) {
  void index;
  void from;
  return <View style={style}>{children}</View>;
}
/**
 * A pressable that springs under the finger and fires a haptic. Use this
 * instead of a bare `Pressable` anywhere the user is meant to tap.
 */
export function Tap({
  children,
  onPress,
  onLongPress,
  disabled,
  scaleTo = 0.97,
  haptic = 'light',
  style,
  accessibilityLabel,
  accessibilityRole = 'button',
  hitSlop,
}: React.PropsWithChildren<{
  onPress?: () => void;
  onLongPress?: () => void;
  disabled?: boolean;
  scaleTo?: number;
  haptic?: 'light' | 'medium' | 'none';
  style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string;
  accessibilityRole?: 'button' | 'link' | 'tab';
  hitSlop?: number;
}>) {
  const reducedMotion = useReducedMotion();
  const pressed = useSharedValue(0);
  const animated = useAnimatedStyle(() => ({
    transform: [{ scale: 1 - pressed.value * (1 - scaleTo) }],
    opacity: 1 - pressed.value * 0.12,
  }));
  return (
    <Pressable
      accessibilityLabel={accessibilityLabel}
      accessibilityRole={accessibilityRole}
      accessibilityState={{ disabled: Boolean(disabled) }}
      disabled={disabled}
      hitSlop={hitSlop ?? 5}
      onPress={() => {
        if (haptic !== 'none') {
          void Haptics.impactAsync(
            haptic === 'medium' ? Haptics.ImpactFeedbackStyle.Medium : Haptics.ImpactFeedbackStyle.Light,
          );
        }
        onPress?.();
      }}
      onLongPress={onLongPress}
      onPressIn={() => {
        pressed.value = reducedMotion ? 0 : withSpring(1, motion.press);
      }}
      onPressOut={() => {
        pressed.value = reducedMotion ? 0 : withSpring(0, motion.press);
      }}>
      <Animated.View style={[style, animated, disabled && styles.disabled]}>{children}</Animated.View>
    </Pressable>
  );
}
function group(value: number, decimals: number) {
  const [whole, fraction] = value.toFixed(decimals).split('.');
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return fraction ? `${grouped}.${fraction}` : grouped;
}
/**
 * A number that counts up to its value on mount and eases between values
 * afterwards.
 *
 * Deliberately a plain `<Text>` driven by `requestAnimationFrame` rather than
 * the usual animated-`TextInput` trick: a `TextInput` brings its own intrinsic
 * width and clips or shoves aside anything next to it, which is wrong for a
 * number that sits inline with a unit. Digits are tabular so the layout does
 * not jitter while the value runs.
 */
export function CountUp({
  value,
  decimals = 0,
  duration = motion.slow,
  prefix = '',
  suffix = '',
  style,
  numberOfLines,
}: {
  value: number;
  decimals?: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  style?: StyleProp<TextStyle>;
  numberOfLines?: number;
}) {
  const target = Number.isFinite(value) ? value : 0;
  const reducedMotion = useReducedMotion();
  const [shown, setShown] = React.useState(target);
  const fromRef = React.useRef(target);
  useEffect(() => {
    const from = fromRef.current;
    if (from === target || duration <= 0 || reducedMotion) {
      fromRef.current = target;
      setShown(target);
      return;
    }
    let frame = 0;
    let start = 0;
    const step = (now: number) => {
      if (!start) start = now;
      const progress = Math.min(1, (now - start) / duration);
      // Cubic ease-out, matching the rest of the system's entrance motion.
      const eased = 1 - (1 - progress) ** 3;
      setShown(from + (target - from) * eased);
      if (progress < 1) {
        frame = requestAnimationFrame(step);
      } else {
        fromRef.current = target;
      }
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, duration, reducedMotion]);
  return (
    <Text
      accessibilityLabel={`${prefix}${group(target, decimals)}${suffix}`}
      numberOfLines={numberOfLines}
      style={[styles.countUp, style]}>
      {prefix}
      {group(shown, decimals)}
      {suffix}
    </Text>
  );
}
/* ------------------------------------------------------------------ *
 * Surfaces
 * ------------------------------------------------------------------ */
export function Card({
  children,
  raised,
  glow,
  padded = true,
  style,
}: React.PropsWithChildren<{
  raised?: boolean;
  glow?: boolean;
  padded?: boolean;
  style?: StyleProp<ViewStyle>;
}>) {
  if (raised) {
    return (
      <View style={[glow ? shadow.glowSoft : shadow.card, styles.cardRaisedWrap, style]}>
        <LinearGradient colors={gradient.hero} end={{ x: 1, y: 1 }} start={{ x: 0, y: 0 }} style={styles.cardRaised}>
          {glow ? <CardGlow /> : null}
          <View style={padded ? styles.cardPad : undefined}>{children}</View>
        </LinearGradient>
      </View>
    );
  }
  return <View style={[styles.card, padded && styles.cardPad, style]}>{children}</View>;
}
/**
 * The soft light in the corner of a raised card. A radial gradient, not a
 * tinted circle — a flat disc clipped by the card edge reads as a stray shape.
 */
function CardGlow() {
  return (
    <Svg height={230} pointerEvents="none" style={styles.cardGlow} width={230}>
      <Defs>
        <RadialGradient cx="50%" cy="50%" id="card-glow" r="50%">
          <Stop offset="0%" stopColor={palette.lime} stopOpacity={0.15} />
          <Stop offset="55%" stopColor={palette.lime} stopOpacity={0.035} />
          <Stop offset="100%" stopColor={palette.lime} stopOpacity={0} />
        </RadialGradient>
      </Defs>
      <Circle cx={115} cy={115} fill="url(#card-glow)" r={115} />
    </Svg>
  );
}

/** A muted well for content nested inside a Card. */
export function Well({
  children,
  style,
  accessible,
  accessibilityLabel,
}: React.PropsWithChildren<{
  style?: StyleProp<ViewStyle>;
  accessible?: boolean;
  accessibilityLabel?: string;
}>) {
  return (
    <View
      accessibilityLabel={accessibilityLabel}
      accessible={accessible ?? Boolean(accessibilityLabel)}
      style={[styles.well, style]}>
      {children}
    </View>
  );
}
/* ------------------------------------------------------------------ *
 * Progress
 * ------------------------------------------------------------------ */
/**
 * The signature ring. Sweeps from empty to `value` on mount, with a gradient
 * stroke and an optional centre label.
 */
export function Ring({
  value,
  size = 96,
  thickness = 9,
  track = palette.line,
  colors = gradient.ring,
  children,
  delay = 120,
}: React.PropsWithChildren<{
  value: number;
  size?: number;
  thickness?: number;
  track?: string;
  colors?: readonly [string, string];
  delay?: number;
}>) {
  const clamped = Math.max(0, Math.min(Number.isFinite(value) ? value : 0, 1));
  const reducedMotion = useReducedMotion();
  const r = (size - thickness) / 2;
  const circumference = 2 * Math.PI * r;
  const progress = useSharedValue(reducedMotion ? clamped : 0);
  useEffect(() => {
    progress.value = reducedMotion
      ? clamped
      : withDelay(delay, withTiming(clamped, { duration: 900, easing: Easing.out(Easing.cubic) }));
  }, [clamped, delay, progress, reducedMotion]);
  const animatedProps = useAnimatedProps(() => ({
    strokeDashoffset: circumference * (1 - progress.value),
  }));
  const gradientId = `ring-${Math.round(size)}-${colors[0].replace('#', '')}`;
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <Svg height={size} style={StyleSheet.absoluteFill} width={size}>
        <Defs>
          <SvgGradient id={gradientId} x1="0" x2="1" y1="0" y2="1">
            <Stop offset="0%" stopColor={colors[0]} />
            <Stop offset="100%" stopColor={colors[1]} />
          </SvgGradient>
        </Defs>
        <Circle cx={size / 2} cy={size / 2} fill="none" r={r} stroke={track} strokeWidth={thickness} />
        <AnimatedCircle
          animatedProps={animatedProps}
          cx={size / 2}
          cy={size / 2}
          fill="none"
          origin={`${size / 2}, ${size / 2}`}
          r={r}
          rotation={-90}
          stroke={`url(#${gradientId})`}
          strokeDasharray={`${circumference} ${circumference}`}
          strokeLinecap="round"
          strokeWidth={thickness}
        />
      </Svg>
      {children}
    </View>
  );
}
/** A horizontal progress bar that fills on mount and turns coral past 100%. */
export function Bar({
  value,
  color = palette.lime,
  height = 5,
  track = palette.line,
  overColor = palette.danger,
  delay = 100,
  style,
}: {
  value: number;
  color?: string;
  height?: number;
  track?: string;
  overColor?: string;
  delay?: number;
  style?: StyleProp<ViewStyle>;
}) {
  const safe = Number.isFinite(value) ? value : 0;
  const over = safe > 1.001;
  const clamped = Math.max(0, Math.min(safe, 1));
  const reducedMotion = useReducedMotion();
  const progress = useSharedValue(reducedMotion ? clamped : 0);
  useEffect(() => {
    progress.value = reducedMotion
      ? clamped
      : withDelay(delay, withTiming(clamped, { duration: 700, easing: Easing.out(Easing.cubic) }));
  }, [clamped, delay, progress, reducedMotion]);
  const animated = useAnimatedStyle(() => ({ width: `${progress.value * 100}%` }));
  return (
    <View style={[{ height, borderRadius: height, backgroundColor: track, overflow: 'hidden' }, style]}>
      <Animated.View
        style={[
          { height: '100%', borderRadius: height, backgroundColor: over ? overColor : color },
          animated,
        ]}
      />
    </View>
  );
}
/** Label + value + bar, used for macros and any capped metric. */
export function MacroChip({
  label,
  value,
  goal,
  color,
  unit = 'g',
}: {
  label: string;
  value: number;
  goal: number;
  color: string;
  unit?: string;
}) {
  return (
    <View style={styles.macro}>
      <Text style={styles.macroLabel}>{label.toUpperCase()}</Text>
      <View style={styles.macroValueRow}>
        <CountUp style={styles.macroValue} value={Math.round(value)} />
        <Text style={styles.macroUnit}>{unit}</Text>
      </View>
      <Bar color={color} height={3} track={palette.line} value={goal > 0 ? value / goal : 0} />
    </View>
  );
}
/* ------------------------------------------------------------------ *
 * Actions
 * ------------------------------------------------------------------ */
export function PrimaryButton({
  label,
  onPress,
  disabled,
  loading,
  icon,
  compact,
  style,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
  icon?: GlyphName;
  compact?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <Tap
      accessibilityLabel={label}
      disabled={disabled || loading}
      haptic="medium"
      onPress={onPress}
      scaleTo={0.975}
      style={[disabled ? undefined : shadow.glow, style]}>
      <LinearGradient
        colors={disabled ? [palette.surfaceHi, palette.surface] : [...gradient.lime]}
        end={{ x: 1, y: 1 }}
        start={{ x: 0, y: 0 }}
        style={[styles.primary, compact && styles.primaryCompact]}>
        {icon ? <Glyph color={disabled ? palette.inkLow : palette.onLime} name={icon} size={17} /> : null}
        <Text maxFontSizeMultiplier={1.8} style={[styles.primaryLabel, disabled && { color: palette.inkLow }]}>
          {loading ? 'Working…' : label}
        </Text>
      </LinearGradient>
    </Tap>
  );
}
export function GhostButton({
  label,
  onPress,
  icon,
  tone = 'default',
  compact,
  disabled,
  style,
}: {
  label: string;
  onPress: () => void;
  icon?: GlyphName;
  tone?: 'default' | 'danger';
  compact?: boolean;
  disabled?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  const color = disabled ? palette.inkLow : tone === 'danger' ? palette.danger : palette.ink;
  return (
    <Tap accessibilityLabel={label} disabled={disabled} onPress={onPress} style={style}>
      <View style={[
        styles.ghost,
        compact && styles.primaryCompact,
        tone === 'danger' && styles.ghostDanger,
        disabled && styles.ghostDisabled,
      ]}>
        {icon ? <Glyph color={color} name={icon} size={16} /> : null}
        <Text maxFontSizeMultiplier={1.8} style={[styles.ghostLabel, { color }]}>{label}</Text>
      </View>
    </Tap>
  );
}
export function Chip({
  label,
  active,
  onPress,
  icon,
}: {
  label: string;
  active?: boolean;
  onPress: () => void;
  icon?: GlyphName;
}) {
  const reducedMotion = useReducedMotion();
  const on = useSharedValue(active ? 1 : 0);
  useEffect(() => {
    on.value = reducedMotion
      ? (active ? 1 : 0)
      : withTiming(active ? 1 : 0, { duration: motion.quick });
  }, [active, on, reducedMotion]);
  const animated = useAnimatedStyle(() => ({
    backgroundColor: interpolateColor(on.value, [0, 1], [palette.surface, palette.lime]),
    borderColor: interpolateColor(on.value, [0, 1], [palette.lineHi, palette.lime]),
  }));
  return (
    <Tap onPress={onPress} scaleTo={0.93}>
      <Animated.View style={[styles.chip, animated]}>
        {icon ? <Glyph color={active ? palette.onLime : palette.inkMid} name={icon} size={13} /> : null}
        <Text style={[styles.chipLabel, active && styles.chipLabelOn]}>{label}</Text>
      </Animated.View>
    </Tap>
  );
}
/** Sliding segmented control. The thumb animates between options. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  style,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (next: T) => void;
  style?: StyleProp<ViewStyle>;
}) {
  const index = Math.max(0, options.findIndex((option) => option.value === value));
  const reducedMotion = useReducedMotion();
  const position = useSharedValue(index);
  const [width, setWidth] = React.useState(0);
  useEffect(() => {
    position.value = reducedMotion ? index : withSpring(index, motion.enter);
  }, [index, position, reducedMotion]);
  const segment = width > 0 ? (width - 6) / options.length : 0;
  const thumb = useAnimatedStyle(() => ({
    transform: [{ translateX: 3 + position.value * segment }],
    width: segment,
  }));
  return (
    <View onLayout={(event) => setWidth(event.nativeEvent.layout.width)} style={[styles.segmented, style]}>
      {width > 0 ? <Animated.View style={[styles.segmentedThumb, thumb]} /> : null}
      {options.map((option) => {
        const active = option.value === value;
        return (
          <Pressable
            accessibilityRole="tab"
            accessibilityState={{ selected: active }}
            key={option.value}
            onPress={() => {
              void Haptics.selectionAsync();
              onChange(option.value);
            }}
            style={styles.segmentedItem}>
            <Text maxFontSizeMultiplier={1.4} numberOfLines={1} style={[styles.segmentedLabel, active && styles.segmentedLabelOn]}>
              {option.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}
/* ------------------------------------------------------------------ *
 * Content
 * ------------------------------------------------------------------ */
export function Metric({
  icon,
  label,
  value,
  detail,
  accent = palette.lime,
  onPress,
  progress,
}: {
  icon: GlyphName;
  label: string;
  value: string;
  detail?: string;
  accent?: string;
  onPress?: () => void;
  /** Optional 0–1 progress turns the icon into a compact visual gauge. */
  progress?: number;
}) {
  const body = (
    <View style={styles.metric}>
      <Ring
        colors={[accent, accent]}
        delay={160}
        size={44}
        thickness={4}
        track={`${accent}20`}
        value={progress ?? 0}>
        <View style={[styles.metricIcon, { backgroundColor: `${accent}14` }]}>
          <Glyph color={accent} name={icon} size={15} />
        </View>
      </Ring>
      <Text style={styles.metricLabel}>{label.toUpperCase()}</Text>
      <Text numberOfLines={1} style={styles.metricValue}>{value}</Text>
      {detail ? <Text numberOfLines={1} style={styles.metricDetail}>{detail}</Text> : null}
    </View>
  );
  return onPress ? <Tap onPress={onPress} style={{ flex: 1 }}>{body}</Tap> : <View style={{ flex: 1 }}>{body}</View>;
}
export function ListRow({
  title,
  detail,
  value,
  unit,
  icon,
  accent = palette.lime,
  onPress,
  right,
  last,
}: {
  title: string;
  detail?: string;
  value?: string;
  /** Rendered quietly beside `value`, e.g. "kcal". */
  unit?: string;
  icon?: GlyphName;
  accent?: string;
  onPress?: () => void;
  right?: React.ReactNode;
  last?: boolean;
}) {
  const body = (
    <View style={[styles.row, !last && styles.rowBorder]}>
      {icon ? (
        <View style={[styles.rowIcon, { backgroundColor: `${accent}14` }]}>
          <Glyph color={accent} name={icon} size={16} />
        </View>
      ) : null}
      <View style={{ flex: 1 }}>
        <Text numberOfLines={1} style={styles.rowTitle}>{title}</Text>
        {detail ? <Text numberOfLines={1} style={styles.rowDetail}>{detail}</Text> : null}
      </View>
      {value ? (
        <View style={styles.rowValueWrap}>
          <Text style={styles.rowValue}>{value}</Text>
          {unit ? <Text style={styles.rowUnit}>{unit}</Text> : null}
        </View>
      ) : null}
      {right ?? (onPress ? <Glyph color={palette.inkLow} name="chevron" size={15} /> : null)}
    </View>
  );
  return onPress ? <Tap onPress={onPress} scaleTo={0.985}>{body}</Tap> : body;
}
export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon: GlyphName;
  title: string;
  body: string;
  action?: React.ReactNode;
}) {
  return (
    <View style={styles.empty}>
      <View style={styles.emptyIcon}>
        <Glyph color={palette.lime} name={icon} size={22} />
      </View>
      <Text style={styles.emptyTitle}>{title}</Text>
      <Text style={styles.emptyBody}>{body}</Text>
      {action ? <View style={{ marginTop: 16, alignSelf: 'stretch' }}>{action}</View> : null}
    </View>
  );
}
/** The always-visible entry point to logging. */
export function VoiceBar({
  label = '“Two rotis and a bowl of rajma”',
  slot,
  title = 'Log with your voice',
}: {
  label?: string;
  slot?: MealSlot;
  title?: string;
}) {
  const router = useRouter();
  return (
    <Tap
      accessibilityLabel="Open quick log"
      haptic="medium"
      onPress={() => router.push(slot ? { pathname: '/quick-log', params: { slot } } : '/quick-log')}
      scaleTo={0.975}
      style={shadow.glow}>
      <LinearGradient colors={[...gradient.lime]} end={{ x: 1, y: 0.6 }} start={{ x: 0, y: 0 }} style={styles.voiceBar}>
        <View style={styles.voiceIcon}>
          <Glyph color={palette.onLime} name="mic" size={19} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.voiceTitle}>{title}</Text>
          <Text numberOfLines={1} style={styles.voiceHint}>{label}</Text>
        </View>
        <Glyph color={palette.onLime} name="chevron" size={17} />
      </LinearGradient>
    </Tap>
  );
}
/** Small status pill — "3 day streak", "Synced", "Offline". */
export function Pill({
  label,
  tone = 'default',
  icon,
}: {
  label: string;
  tone?: 'default' | 'accent' | 'danger';
  icon?: GlyphName;
}) {
  const color =
    tone === 'accent' ? palette.lime : tone === 'danger' ? palette.danger : palette.inkMid;
  return (
    <View style={[styles.pill, { borderColor: `${color}33`, backgroundColor: `${color}12` }]}>
      {icon ? <Glyph color={color} name={icon} size={11} /> : null}
      <Text style={[styles.pillLabel, { color }]}>{label}</Text>
    </View>
  );
}
/** Frosted bar for sticky footers over scrolling content. */
export function GlassFooter({ children }: React.PropsWithChildren) {
  return (
    <BlurView intensity={Platform.OS === 'ios' ? 40 : 0} style={styles.glassFooter} tint="dark">
      <View style={styles.glassFooterInner}>{children}</View>
    </BlurView>
  );
}
const styles = StyleSheet.create({
  canvas: { flex: 1, backgroundColor: palette.bg },
  // zIndex is explicit so the ambient glow always stays behind content, even
  // where a platform orders absolutely-positioned siblings differently.
  safe: { flex: 1, zIndex: 1 },
  header: { flexDirection: 'row', alignItems: 'center', paddingTop: 6, marginBottom: space.md },
  eyebrow: { ...text.label, color: palette.inkLow, marginBottom: 5 },
  headerTitle: { ...text.title, color: palette.ink },
  sectionTitle: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginTop: space.lg,
    marginBottom: space.sm,
  },
  sectionTitleText: { ...text.section, color: palette.ink },
  sectionAside: { ...text.caption, fontFamily: text.value.fontFamily, color: palette.inkMid },
  card: {
    backgroundColor: palette.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: palette.line,
  },
  cardRaisedWrap: { borderRadius: radius.lg },
  cardRaised: {
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: palette.lineHi,
    overflow: 'hidden',
  },
  cardPad: { padding: 18 },
  cardGlow: { position: 'absolute', top: -96, right: -74 },
  well: {
    backgroundColor: palette.surfaceLo,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: palette.line,
    padding: 12,
  },
  countUp: { padding: 0, margin: 0, ...tabular },
  macro: { flex: 1, backgroundColor: palette.surfaceLo, borderRadius: radius.sm, borderWidth: 1, borderColor: palette.line, padding: 10 },
  macroLabel: { ...text.label, fontSize: 9, letterSpacing: 0.9, color: palette.inkLow },
  macroValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 2, marginTop: 3, marginBottom: 6 },
  macroValue: { ...text.value, fontSize: 15, color: palette.ink, ...tabular },
  macroUnit: { ...text.caption, fontSize: 10, color: palette.inkLow },
  primary: {
    minHeight: 54,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 20,
  },
  primaryCompact: { minHeight: 44, borderRadius: radius.sm },
  primaryLabel: { ...text.row, fontSize: 15, color: palette.onLime },
  ghost: {
    minHeight: 52,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surface,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 18,
  },
  ghostDanger: { borderColor: `${palette.danger}44`, backgroundColor: `${palette.danger}10` },
  ghostDisabled: { opacity: 0.5 },
  ghostLabel: { ...text.row, fontSize: 14 },
  chip: {
    minHeight: 44,
    paddingHorizontal: 14,
    borderRadius: radius.pill,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 6,
  },
  chipLabel: { ...text.micro, fontFamily: text.value.fontFamily, fontSize: 11.5, color: palette.inkMid },
  chipLabelOn: { color: palette.onLime },
  segmented: {
    flexDirection: 'row',
    backgroundColor: palette.surfaceLo,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: palette.line,
    padding: 3,
    height: 44,
  },
  segmentedThumb: {
    position: 'absolute',
    top: 3,
    bottom: 3,
    borderRadius: radius.pill,
    backgroundColor: alpha.limeFaint,
    borderWidth: 1,
    borderColor: alpha.limeGlowSoft,
  },
  segmentedItem: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  segmentedLabel: { ...text.micro, fontFamily: text.value.fontFamily, fontSize: 11.5, color: palette.inkMid },
  segmentedLabelOn: { color: palette.lime },
  metric: {
    flex: 1,
    minHeight: 126,
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.line,
    borderRadius: radius.md,
    padding: 12,
  },
  metricIcon: {
    width: 32,
    height: 32,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  metricLabel: { ...text.label, fontSize: 8.5, letterSpacing: 0.9, color: palette.inkLow, marginTop: 9 },
  metricValue: { ...text.headline, fontSize: 18, color: palette.ink, marginTop: 3, ...tabular },
  metricDetail: { ...text.caption, fontSize: 10, color: palette.inkLow, marginTop: 1 },
  row: { minHeight: 62, flexDirection: 'row', alignItems: 'center', gap: 11, paddingVertical: 10 },
  rowBorder: { borderBottomWidth: 1, borderBottomColor: palette.line },
  rowIcon: { width: 34, height: 34, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  rowTitle: { ...text.row, color: palette.ink },
  rowDetail: { ...text.caption, color: palette.inkLow, marginTop: 2 },
  rowValueWrap: { flexDirection: 'row', alignItems: 'baseline', gap: 3 },
  rowValue: { ...text.value, color: palette.lime, ...tabular },
  rowUnit: { ...text.caption, fontSize: 10, color: palette.inkLow },
  empty: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 30,
    paddingHorizontal: 24,
    backgroundColor: palette.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: palette.line,
  },
  emptyIcon: {
    width: 46,
    height: 46,
    borderRadius: 16,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  emptyTitle: { ...text.section, color: palette.ink, textAlign: 'center' },
  emptyBody: { ...text.body, color: palette.inkMid, textAlign: 'center', marginTop: 5 },
  voiceBar: {
    minHeight: 68,
    borderRadius: radius.md,
    paddingHorizontal: 14,
    alignItems: 'center',
    flexDirection: 'row',
    gap: 12,
  },
  voiceIcon: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: alpha.onLimeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceTitle: { ...text.row, fontSize: 14.5, color: palette.onLime },
  voiceHint: { ...text.caption, fontSize: 11, color: 'rgba(10, 36, 5, 0.62)', marginTop: 1 },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    height: 24,
    paddingHorizontal: 9,
    borderRadius: radius.pill,
    borderWidth: 1,
  },
  pillLabel: { ...text.micro, fontFamily: text.value.fontFamily, fontSize: 10.5 },
  glassFooter: { borderTopWidth: 1, borderTopColor: palette.line, backgroundColor: 'rgba(9,12,13,0.9)' },
  glassFooterInner: { padding: space.md, paddingBottom: space.md + 6, gap: 10 },
  disabled: { opacity: 0.5 },
});
