import { useEffect, useId, useMemo, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedProps,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  Line,
  LinearGradient as SvgGradient,
  Path,
  Stop,
} from 'react-native-svg';

import { Glyph } from '@/src/components/glyph';
import { CountUp } from '@/src/components/ui';
import { useReducedMotion } from '@/src/lib/accessibility';
import type { TrendPoint } from '@/src/lib/training';
import { motion, palette, radius, shadow, tabular, text } from '@/src/theme';

const AnimatedPath = Animated.createAnimatedComponent(Path);

const PAD_LEFT = 14;
/**
 * Deep enough that the reference caption, which hangs above its rule, still has
 * air when the goal is the highest value on the chart and its rule sits on the
 * very top of the plot.
 */
const PAD_TOP = 26;
const HALO = 30;
const DOT = 9;
/** Past this many readings the individual markers turn into noise. */
const DOT_LIMIT = 16;
/** Height of a value label in the right gutter; it is centred on its value. */
const AXIS_H = 18;
/** The reference caption box, and the air between it and its dashed rule. */
const CAPTION_H = 12;
const CAPTION_GAP = 3;

/** `2026-07-03` → `3 Jul`, in the reader's locale. */
const DAY_MONTH = new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short' });

type Coordinate = { x: number; y: number };

export type TrendChartProps = {
  points: TrendPoint[];
  height?: number;
  unit?: string;
  /** Line, fill and marker colour. Defaults to the accent. */
  color?: string;
  /** Draws a dashed rule at this value — a goal or target line. */
  reference?: number;
  /** Tiny caption printed beside the reference rule. */
  referenceLabel?: string;
  /** Copy shown when there is not enough data to draw a line. */
  emptyBody?: string;
  /** First / last date captions under the plot. */
  showDates?: boolean;
  /** Set false to render the finished chart without the draw-on animation. */
  animate?: boolean;
};

/**
 * Line chart for any dated series — exercise performance, weight, intake.
 *
 * Draws a monotone (never overshooting) curve over a lime gradient wash, with
 * the latest reading marked by a glowing dot. The line traces itself in on
 * mount; the wash blooms behind it and the marker springs in last.
 */
export function TrendChart({
  points,
  height = 160,
  unit = '',
  color = palette.lime,
  reference,
  referenceLabel,
  emptyBody = 'Log this exercise in at least two workouts to see the trend.',
  showDates = true,
  animate = true,
}: TrendChartProps) {
  const [width, setWidth] = useState(0);
  const gradientId = `trend${useId().replace(/[^a-zA-Z0-9]/g, '')}`;
  const enough = points.length >= 2;

  const plot = useMemo(() => {
    if (!enough || width <= 0) return null;

    const padBottom = showDates ? 26 : 12;

    const values = points.map((point) => point.value);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const floor = reference === undefined ? low : Math.min(low, reference);
    const ceiling = reference === undefined ? high : Math.max(high, reference);
    const span = ceiling - floor;

    /** Gutter sized to the widest value label so nothing ever clips. */
    const labelChars = Math.max(
      format(low, decimalsOf(low)).length,
      format(high, decimalsOf(high)).length,
    ) + unit.length;
    const padRight = Math.min(Math.round(width * 0.32), Math.max(44, labelChars * 6 + 14));
    const plotW = Math.max(1, width - PAD_LEFT - padRight);
    const plotH = Math.max(1, height - PAD_TOP - padBottom);
    const baseline = PAD_TOP + plotH;

    const at = (index: number) =>
      PAD_LEFT + (points.length === 1 ? plotW / 2 : (index / (points.length - 1)) * plotW);
    const height01 = (value: number) =>
      span === 0 ? PAD_TOP + plotH / 2 : PAD_TOP + (1 - (value - floor) / span) * plotH;

    const coordinates: Coordinate[] = points.map((point, index) => ({
      x: at(index),
      y: height01(point.value),
    }));
    const last = coordinates[coordinates.length - 1];
    const first = coordinates[0];
    const line = curve(coordinates);

    const lowY = height01(low);
    const highY = height01(high);
    const referenceY = reference === undefined ? null : height01(reference);
    /** The caption sits above the rule, so the pair occupy this band. */
    const captionTop = referenceY === null ? null : referenceY - CAPTION_H - CAPTION_GAP;

    /**
     * When a value label and the goal caption want the same strip of gutter the
     * *value* gives way. A dashed rule with no caption is unreadable, and the
     * number stranded beside it gets misread as the goal it is sitting on.
     */
    const buried = (y: number) => referenceY !== null && captionTop !== null
      && y + AXIS_H / 2 > captionTop - 2
      && y - AXIS_H / 2 < referenceY + 2;

    return {
      plotW,
      padRight,
      coordinates,
      line,
      area: `${line} L ${round(last.x)} ${round(baseline)} L ${round(first.x)} ${round(baseline)} Z`,
      /** Generous over-estimate so the dash reveal always completes. */
      length: polylineLength(coordinates) * 1.3 + 48,
      /**
       * Structural rules. Any that the goal rule would land on is dropped —
       * a semantic line is never drawn underneath a decorative one.
       */
      grid: [PAD_TOP, PAD_TOP + plotH / 2, baseline]
        .filter((y) => referenceY === null || Math.abs(y - referenceY) > 1),
      marker: last,
      low,
      high,
      lowY,
      highY,
      referenceY,
      captionTop,
      hideHigh: buried(highY),
      hideLow: buried(lowY),
    };
  }, [enough, width, height, points, reference, showDates, unit]);

  const reducedMotion = useReducedMotion();
  const ready = plot !== null;
  const pathLength = plot?.length ?? 1;

  const draw = useSharedValue(animate ? 0 : 1);
  const bloom = useSharedValue(animate ? 0 : 1);
  const marker = useSharedValue(animate ? 0 : 1);

  useEffect(() => {
    if (!ready) return;
    if (!animate || reducedMotion) {
      draw.value = 1;
      bloom.value = 1;
      marker.value = 1;
      return;
    }
    draw.value = 0;
    bloom.value = 0;
    marker.value = 0;
    draw.value = withTiming(1, { duration: 1000, easing: Easing.out(Easing.cubic) });
    bloom.value = withDelay(220, withTiming(1, { duration: motion.slow, easing: Easing.out(Easing.quad) }));
    marker.value = withDelay(780, withSpring(1, motion.bouncy));
  }, [ready, animate, points, draw, bloom, marker, reducedMotion]);

  const lineProps = useAnimatedProps(() => ({
    strokeDashoffset: pathLength * (1 - draw.value),
  }));
  const bloomStyle = useAnimatedStyle(() => ({ opacity: bloom.value }));
  const markerStyle = useAnimatedStyle(() => ({
    opacity: Math.min(1, marker.value),
    transform: [{ scale: 0.3 + marker.value * 0.7 }],
  }));

  if (!enough) {
    return (
      <View style={[styles.empty, { height }]}>
        <View style={styles.emptyIcon}>
          <Glyph color={palette.lime} name="trend" size={17} />
        </View>
        <Text style={styles.emptyText}>{emptyBody}</Text>
      </View>
    );
  }

  const firstPoint = points[0];
  const lastPoint = points[points.length - 1];

  return (
    <View onLayout={(event) => setWidth(event.nativeEvent.layout.width)} style={{ height }}>
      {plot ? (
        <>
          {/* Gradient wash, blooming in behind the line. */}
          <Animated.View pointerEvents="none" style={[StyleSheet.absoluteFill, bloomStyle]}>
            <Svg height={height} width={width}>
              <Defs>
                <SvgGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
                  <Stop offset="0" stopColor={color} stopOpacity={0.34} />
                  <Stop offset="0.55" stopColor={color} stopOpacity={0.1} />
                  <Stop offset="1" stopColor={color} stopOpacity={0} />
                </SvgGradient>
              </Defs>
              <Path d={plot.area} fill={`url(#${gradientId})`} />
            </Svg>
          </Animated.View>

          <Svg height={height} width={width}>
            {plot.grid.map((y) => (
              <Line
                key={y}
                stroke={palette.line}
                strokeWidth={1}
                x1={PAD_LEFT}
                x2={PAD_LEFT + plot.plotW}
                y1={y}
                y2={y}
              />
            ))}
            {plot.referenceY === null ? null : (
              <Line
                stroke={palette.inkFaint}
                strokeDasharray={[3, 6]}
                strokeWidth={1}
                x1={PAD_LEFT}
                x2={PAD_LEFT + plot.plotW}
                y1={plot.referenceY}
                y2={plot.referenceY}
              />
            )}
            <AnimatedPath
              animatedProps={lineProps}
              d={plot.line}
              fill="none"
              stroke={color}
              strokeDasharray={[plot.length, plot.length]}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2.6}
            />
            {points.length > DOT_LIMIT
              ? null
              : plot.coordinates.slice(0, -1).map((coordinate, index) => (
                <Circle
                  cx={coordinate.x}
                  cy={coordinate.y}
                  fill={palette.bg}
                  key={`${points[index].date}-${index}`}
                  r={2.4}
                  stroke={`${color}66`}
                  strokeWidth={1.4}
                />
              ))}
          </Svg>

          {/* The latest reading, glowing. */}
          <Animated.View
            pointerEvents="none"
            style={[
              styles.marker,
              { left: plot.marker.x - HALO / 2, top: plot.marker.y - HALO / 2 },
              markerStyle,
            ]}>
            <View style={[styles.halo, { backgroundColor: `${color}1F`, borderColor: `${color}3D` }]} />
            <View style={[styles.dot, shadow.glow, { backgroundColor: color, shadowColor: color }]} />
          </Animated.View>

          {/* Min / max, parked in the right gutter — one voice for one axis. */}
          {plot.hideHigh ? null : (
            <View
              pointerEvents="none"
              style={[styles.axis, { top: plot.highY - AXIS_H / 2, width: plot.padRight - 8 }]}>
              <CountUp
                decimals={decimalsOf(plot.high)}
                style={styles.axisValue}
                suffix={unit}
                value={plot.high}
              />
            </View>
          )}
          {plot.low === plot.high || plot.hideLow ? null : (
            <View
              pointerEvents="none"
              style={[styles.axis, { top: plot.lowY - AXIS_H / 2, width: plot.padRight - 8 }]}>
              <CountUp
                decimals={decimalsOf(plot.low)}
                style={styles.axisValue}
                suffix={unit}
                value={plot.low}
              />
            </View>
          )}
          {/* The goal rule always says what it is, sitting just above itself. */}
          {plot.captionTop === null || !referenceLabel ? null : (
            <View
              pointerEvents="none"
              style={[styles.caption, { top: plot.captionTop, width: plot.padRight - 8 }]}>
              <Text numberOfLines={1} style={styles.captionText}>{referenceLabel.toUpperCase()}</Text>
            </View>
          )}

          {showDates ? (
            <View pointerEvents="none" style={[styles.dates, { left: PAD_LEFT, width: plot.plotW }]}>
              <Text style={styles.dateText}>{dayMonth(firstPoint.date)}</Text>
              <Text style={styles.dateText}>{dayMonth(lastPoint.date)}</Text>
            </View>
          ) : null}
        </>
      ) : null}
    </View>
  );
}

/* ------------------------------------------------------------------ *
 * Geometry
 * ------------------------------------------------------------------ */

function round(value: number) {
  return Math.round(value * 10) / 10;
}

/**
 * Monotone cubic interpolation (Fritsch–Carlson). Smooths the line without
 * inventing peaks or troughs the data never had.
 */
function curve(points: Coordinate[]) {
  const count = points.length;
  if (count === 0) return '';
  if (count === 1) return `M ${round(points[0].x)} ${round(points[0].y)}`;

  const widths: number[] = [];
  const slopes: number[] = [];
  for (let index = 0; index < count - 1; index += 1) {
    const dx = points[index + 1].x - points[index].x || 1;
    widths.push(dx);
    slopes.push((points[index + 1].y - points[index].y) / dx);
  }

  const tangents: number[] = new Array<number>(count);
  tangents[0] = slopes[0];
  tangents[count - 1] = slopes[count - 2];
  for (let index = 1; index < count - 1; index += 1) {
    const before = slopes[index - 1];
    const after = slopes[index];
    if (before * after <= 0) {
      tangents[index] = 0;
      continue;
    }
    const w1 = 2 * widths[index] + widths[index - 1];
    const w2 = widths[index] + 2 * widths[index - 1];
    tangents[index] = (w1 + w2) / (w1 / before + w2 / after);
  }

  let d = `M ${round(points[0].x)} ${round(points[0].y)}`;
  for (let index = 0; index < count - 1; index += 1) {
    const step = widths[index] / 3;
    const from = points[index];
    const to = points[index + 1];
    d += ` C ${round(from.x + step)} ${round(from.y + tangents[index] * step)}`
      + ` ${round(to.x - step)} ${round(to.y - tangents[index + 1] * step)}`
      + ` ${round(to.x)} ${round(to.y)}`;
  }
  return d;
}

function polylineLength(points: Coordinate[]) {
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    total += Math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y);
  }
  return total;
}

/** Keeps a label showing exactly the digits the value carries, up to two. */
function decimalsOf(value: number) {
  const fraction = String(value).split('.')[1];
  return fraction ? Math.min(fraction.length, 2) : 0;
}

/**
 * `2026-07-03` → `3 Jul`. The parts are read out by hand and rebuilt as a local
 * date: `new Date('2026-07-03')` is parsed as UTC midnight, which slips a day
 * backwards for every reader west of Greenwich.
 */
function dayMonth(date: string) {
  const parts = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);
  if (!parts) return date;
  const parsed = new Date(Number(parts[1]), Number(parts[2]) - 1, Number(parts[3]));
  return Number.isNaN(parsed.getTime()) ? date : DAY_MONTH.format(parsed);
}

/** Mirrors the grouping `CountUp` applies, so the gutter is sized correctly. */
function format(value: number, decimals: number) {
  const [whole, fraction] = value.toFixed(decimals).split('.');
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return fraction ? `${grouped}.${fraction}` : grouped;
}

const styles = StyleSheet.create({
  empty: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 26,
    gap: 10,
  },
  emptyIcon: {
    width: 38,
    height: 38,
    borderRadius: radius.sm,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyText: { ...text.caption, color: palette.inkMid, textAlign: 'center' },

  marker: {
    position: 'absolute',
    width: HALO,
    height: HALO,
    alignItems: 'center',
    justifyContent: 'center',
  },
  halo: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: HALO / 2,
    borderWidth: 1,
  },
  dot: { width: DOT, height: DOT, borderRadius: DOT / 2 },

  axis: {
    position: 'absolute',
    right: 0,
    height: AXIS_H,
    justifyContent: 'center',
  },
  axisValue: {
    ...text.micro,
    ...tabular,
    fontSize: 10.5,
    color: palette.inkMid,
    textAlign: 'right',
    textAlignVertical: 'center',
    includeFontPadding: false,
    height: 16,
  },

  caption: {
    position: 'absolute',
    right: 0,
    height: CAPTION_H,
    justifyContent: 'center',
  },
  captionText: {
    ...text.label,
    fontSize: 8,
    color: palette.inkLow,
    textAlign: 'right',
    includeFontPadding: false,
  },

  dates: {
    position: 'absolute',
    bottom: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  dateText: { ...text.micro, ...tabular, fontSize: 9.5, color: palette.inkLow },
});
