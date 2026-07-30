import Svg, { Circle, Defs, LinearGradient, Path, Stop } from 'react-native-svg';

import { palette } from '@/src/theme';

/**
 * The Calorie Lens mark: an open aperture ring — the "lens" — closing around a
 * flame. It doubles as the progress ring used throughout the app, so the logo
 * and the product's core visual are the same idea.
 */
export function BrandMark({ size = 64, mono }: { size?: number; mono?: string }) {
  const ring = mono ?? 'url(#brandRing)';
  const flame = mono ?? 'url(#brandFlame)';
  return (
    <Svg fill="none" height={size} viewBox="0 0 64 64" width={size}>
      <Defs>
        <LinearGradient id="brandRing" x1="0" x2="1" y1="0" y2="1">
          <Stop offset="0%" stopColor={palette.lime} />
          <Stop offset="100%" stopColor="#5FD97A" />
        </LinearGradient>
        <LinearGradient id="brandFlame" x1="0.5" x2="0.5" y1="0" y2="1">
          <Stop offset="0%" stopColor={palette.lime} />
          <Stop offset="100%" stopColor={palette.limeDeep} />
        </LinearGradient>
      </Defs>

      {/* Aperture ring, open at the top-right like a progress arc at ~78%. */}
      <Circle
        cx="32"
        cy="32"
        fill="none"
        r="25"
        stroke={ring}
        strokeDasharray="122 157"
        strokeLinecap="round"
        strokeWidth="7"
        transform="rotate(-72 32 32)"
      />

      {/* Flame. */}
      <Path
        d="M32 47.5c5.6 0 9.4-3.4 9.4-8.4 0-6-5.1-9.1-6.4-15.9-3.1 2.4-4.8 5.6-4.8 8.8 0 1.9-1 2.8-2.1 2.8-1.3 0-2.2-1-2.3-2.7-2 2.3-3.2 4.9-3.2 7.4 0 5 3.8 8 9.4 8Z"
        fill={flame}
      />
    </Svg>
  );
}
