import { Text, View } from 'react-native';
import Svg, { Circle, Line, Path, Text as SvgText } from 'react-native-svg';

import type { TrendPoint } from '@/src/lib/training';
import { palette, type } from '@/src/theme';

/** Minimal line chart for exercise performance trends (react-native-svg). */
export function TrendChart({
  points,
  height = 160,
  unit = '',
}: {
  points: TrendPoint[];
  height?: number;
  unit?: string;
}) {
  if (points.length < 2) {
    return (
      <View style={{ height, alignItems: 'center', justifyContent: 'center' }}>
        <Text style={{ color: palette.muted, fontFamily: type.regular, fontSize: 11, textAlign: 'center' }}>
          Log this exercise in at least two workouts to see the trend.
        </Text>
      </View>
    );
  }
  const width = 320;
  const padX = 14;
  const padTop = 16;
  const padBottom = 26;
  const values = points.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const plotW = width - padX * 2;
  const plotH = height - padTop - padBottom;
  const x = (index: number) => padX + (points.length === 1 ? plotW / 2 : (index / (points.length - 1)) * plotW);
  const y = (value: number) => padTop + (1 - (value - min) / span) * plotH;
  const path = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(index).toFixed(1)} ${y(point.value).toFixed(1)}`)
    .join(' ');
  const first = points[0];
  const last = points[points.length - 1];
  const gridYs = [padTop, padTop + plotH / 2, padTop + plotH];

  return (
    <View>
      <Svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
        {gridYs.map((gy) => (
          <Line key={gy} x1={padX} y1={gy} x2={width - padX} y2={gy} stroke={palette.line} strokeWidth={1} />
        ))}
        <Path d={path} fill="none" stroke={palette.limeDark} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />
        {points.map((point, index) => (
          <Circle key={`${point.date}-${index}`} cx={x(index)} cy={y(point.value)} r={index === points.length - 1 ? 4 : 2.6} fill={index === points.length - 1 ? palette.forest : palette.limeDark} />
        ))}
        <SvgText x={padX} y={padTop - 5} fontSize={10} fill={palette.muted}>{`${max}${unit}`}</SvgText>
        <SvgText x={padX} y={padTop + plotH + 14} fontSize={10} fill={palette.muted}>{`${min}${unit}`}</SvgText>
        <SvgText x={padX} y={height - 2} fontSize={9} fill={palette.muted}>{first.date.slice(5)}</SvgText>
        <SvgText x={width - padX} y={height - 2} fontSize={9} fill={palette.muted} textAnchor="end">{last.date.slice(5)}</SvgText>
      </Svg>
    </View>
  );
}
