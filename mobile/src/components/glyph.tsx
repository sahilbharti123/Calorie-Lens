import Svg, { Circle, Line, Path, Rect } from 'react-native-svg';

export type GlyphName =
  | 'home' | 'bowl' | 'dumbbell' | 'chart' | 'mic' | 'water' | 'steps'
  | 'sleep' | 'plus' | 'chevron' | 'heart' | 'watch' | 'trash' | 'spark';

export function Glyph({ name, color, size = 24 }: { name: GlyphName; color: string; size?: number }) {
  const common = {
    stroke: color,
    strokeWidth: 1.9,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  };
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      {name === 'home' && <><Path d="M3.5 10.5 12 3.8l8.5 6.7" {...common} /><Path d="M5.8 9.2v10.5h12.4V9.2M9.5 19.7v-6.1h5v6.1" {...common} /></>}
      {name === 'bowl' && <><Path d="M3.4 10.2h17.2c-.5 5-3.5 8.1-8.6 8.1s-8.1-3.1-8.6-8.1Z" {...common} /><Path d="M7.8 6.8c1.1-1.2 2.3-1.2 3.4 0s2.3 1.2 3.4 0 2.3-1.2 3.4 0" {...common} /></>}
      {name === 'dumbbell' && <><Line x1="7" y1="12" x2="17" y2="12" {...common} /><Rect x="3" y="8.2" width="3.4" height="7.6" rx="1.2" {...common} /><Rect x="17.6" y="8.2" width="3.4" height="7.6" rx="1.2" {...common} /></>}
      {name === 'chart' && <><Line x1="4" y1="20" x2="20" y2="20" {...common} /><Rect x="5" y="12" width="3" height="6" rx=".8" {...common} /><Rect x="10.5" y="8" width="3" height="10" rx=".8" {...common} /><Rect x="16" y="4" width="3" height="14" rx=".8" {...common} /></>}
      {name === 'mic' && <><Rect x="8.3" y="3" width="7.4" height="11.7" rx="3.7" {...common} /><Path d="M5.4 11.3a6.6 6.6 0 0 0 13.2 0M12 17.9V21M8.8 21h6.4" {...common} /></>}
      {name === 'water' && <Path d="M12 2.9S6.3 9.4 6.3 14a5.7 5.7 0 1 0 11.4 0C17.7 9.4 12 2.9 12 2.9Z" {...common} />}
      {name === 'steps' && <><Path d="M9.4 4.2c1.7.9 1.9 4.2.8 6.5S6.7 14 5.1 13.2c-1.6-.8-1.3-3.1-.1-5.4s2.7-4.5 4.4-3.6Z" {...common} /><Path d="M15.8 12.1c1.6-.4 3 2.2 3.4 4.3.4 2.2-.6 3.5-2.4 3.9-1.8.3-2.9-1.4-3.3-3.5-.4-2.2.7-4.4 2.3-4.7Z" {...common} /></>}
      {name === 'sleep' && <Path d="M18.7 16.7A8.2 8.2 0 0 1 8.1 5.4a8.2 8.2 0 1 0 10.6 11.3Z" {...common} />}
      {name === 'plus' && <><Line x1="12" y1="5" x2="12" y2="19" {...common} /><Line x1="5" y1="12" x2="19" y2="12" {...common} /></>}
      {name === 'chevron' && <Path d="m9 5 7 7-7 7" {...common} />}
      {name === 'heart' && <Path d="M20.5 8.7c0 5.1-8.5 10-8.5 10s-8.5-4.9-8.5-10A4.5 4.5 0 0 1 12 6.6a4.5 4.5 0 0 1 8.5 2.1Z" {...common} />}
      {name === 'watch' && <><Rect x="7" y="6" width="10" height="12" rx="3" {...common} /><Path d="M9 6 10 2h4l1 4M9 18l1 4h4l1-4" {...common} /><Circle cx="12" cy="12" r="2.4" {...common} /></>}
      {name === 'trash' && <><Path d="M5.5 7.5h13M9 4.5h6M7.2 7.5l.7 12h8.2l.7-12M10 11v5M14 11v5" {...common} /></>}
      {name === 'spark' && <><Path d="M12 3.3c.6 4.6 2.1 6.1 6.7 6.7-4.6.6-6.1 2.1-6.7 6.7-.6-4.6-2.1-6.1-6.7-6.7 4.6-.6 6.1-2.1 6.7-6.7Z" {...common} /><Path d="M18.2 15.8c.2 1.8.8 2.4 2.5 2.6-1.7.2-2.3.8-2.5 2.5-.2-1.7-.8-2.3-2.5-2.5 1.7-.2 2.3-.8 2.5-2.6Z" {...common} /></>}
    </Svg>
  );
}
