import Svg, { Circle, Line, Path, Polyline, Rect } from 'react-native-svg';

export type GlyphName =
  // navigation + chrome
  | 'home' | 'bowl' | 'dumbbell' | 'chart' | 'spark' | 'chevron' | 'chevronDown'
  | 'chevronLeft' | 'close' | 'plus' | 'minus' | 'more' | 'search' | 'filter'
  // logging
  | 'mic' | 'keyboard' | 'camera' | 'edit' | 'check' | 'trash' | 'copy' | 'folder'
  // metrics
  | 'water' | 'steps' | 'sleep' | 'flame' | 'scale' | 'heart' | 'target' | 'timer'
  | 'calendar' | 'trophy' | 'arrowUp' | 'arrowDown' | 'trend'
  // training
  | 'play' | 'pause' | 'restart' | 'muscle' | 'book' | 'video'
  // system
  | 'watch' | 'settings' | 'user' | 'lock' | 'shield' | 'cloud' | 'download'
  | 'share' | 'bell' | 'info' | 'alert' | 'star' | 'link' | 'logout' | 'sun';

export function Glyph({
  name,
  color,
  size = 24,
  strokeWidth = 1.9,
}: {
  name: GlyphName;
  color: string;
  size?: number;
  strokeWidth?: number;
}) {
  const s = {
    stroke: color,
    strokeWidth,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  };
  return (
    <Svg fill="none" height={size} viewBox="0 0 24 24" width={size}>
      {/* ---------- navigation + chrome ---------- */}
      {name === 'home' && <><Path d="M3.5 10.5 12 3.8l8.5 6.7" {...s} /><Path d="M5.8 9.2v10.5h12.4V9.2M9.5 19.7v-6.1h5v6.1" {...s} /></>}
      {name === 'bowl' && <><Path d="M3.4 10.2h17.2c-.5 5-3.5 8.1-8.6 8.1s-8.1-3.1-8.6-8.1Z" {...s} /><Path d="M7.8 6.8c1.1-1.2 2.3-1.2 3.4 0s2.3 1.2 3.4 0 2.3-1.2 3.4 0" {...s} /></>}
      {/*
        Plates are 5.6 units wide on a slightly thinner stroke, so at tab-bar
        size (21px → 4.9px wide, 1.4px stroke) they still read as open plates
        rather than the filled dots the old 3.4-unit plates collapsed into.
        The bar is shortened to suit, keeping the icon's overall width.
      */}
      {name === 'dumbbell' && <><Line x1="8.6" x2="15.4" y1="12" y2="12" {...s} strokeWidth={strokeWidth - 0.3} /><Rect height="10.4" rx="1.8" width="5.6" x="3" y="6.8" {...s} strokeWidth={strokeWidth - 0.3} /><Rect height="10.4" rx="1.8" width="5.6" x="15.4" y="6.8" {...s} strokeWidth={strokeWidth - 0.3} /></>}
      {name === 'chart' && <><Line x1="4" x2="20" y1="20" y2="20" {...s} /><Rect height="6" rx=".8" width="3" x="5" y="12" {...s} /><Rect height="10" rx=".8" width="3" x="10.5" y="8" {...s} /><Rect height="14" rx=".8" width="3" x="16" y="4" {...s} /></>}
      {name === 'spark' && <><Path d="M12 3.3c.6 4.6 2.1 6.1 6.7 6.7-4.6.6-6.1 2.1-6.7 6.7-.6-4.6-2.1-6.1-6.7-6.7 4.6-.6 6.1-2.1 6.7-6.7Z" {...s} /><Path d="M18.2 15.8c.2 1.8.8 2.4 2.5 2.6-1.7.2-2.3.8-2.5 2.5-.2-1.7-.8-2.3-2.5-2.5 1.7-.2 2.3-.8 2.5-2.6Z" {...s} /></>}
      {name === 'chevron' && <Path d="m9 5 7 7-7 7" {...s} />}
      {name === 'chevronDown' && <Path d="m5 9 7 7 7-7" {...s} />}
      {name === 'chevronLeft' && <Path d="m15 5-7 7 7 7" {...s} />}
      {name === 'close' && <><Line x1="6" x2="18" y1="6" y2="18" {...s} /><Line x1="18" x2="6" y1="6" y2="18" {...s} /></>}
      {name === 'plus' && <><Line x1="12" x2="12" y1="5" y2="19" {...s} /><Line x1="5" x2="19" y1="12" y2="12" {...s} /></>}
      {name === 'minus' && <Line x1="5" x2="19" y1="12" y2="12" {...s} />}
      {name === 'more' && <><Circle cx="5.5" cy="12" r="1.4" fill={color} stroke="none" /><Circle cx="12" cy="12" r="1.4" fill={color} stroke="none" /><Circle cx="18.5" cy="12" r="1.4" fill={color} stroke="none" /></>}
      {name === 'search' && <><Circle cx="10.8" cy="10.8" r="6.3" {...s} /><Line x1="15.4" x2="20" y1="15.4" y2="20" {...s} /></>}
      {name === 'filter' && <Path d="M3.8 5.5h16.4l-6.4 7.6v5.9l-3.6 1.8v-7.7L3.8 5.5Z" {...s} />}

      {/* ---------- logging ---------- */}
      {name === 'mic' && <><Rect height="11.7" rx="3.7" width="7.4" x="8.3" y="3" {...s} /><Path d="M5.4 11.3a6.6 6.6 0 0 0 13.2 0M12 17.9V21M8.8 21h6.4" {...s} /></>}
      {name === 'keyboard' && <><Rect height="12" rx="2.6" width="19" x="2.5" y="6" {...s} /><Path d="M6.4 9.8h.01M9.8 9.8h.01M13.2 9.8h.01M16.6 9.8h.01M6.4 13.2h.01M17.6 13.2h.01M9.2 15.9h5.6" {...s} /></>}
      {name === 'camera' && <><Path d="M3.5 8.4h3.3l1.5-2.4h7.4l1.5 2.4h3.3v10.1H3.5V8.4Z" {...s} /><Circle cx="12" cy="13" r="3.5" {...s} /></>}
      {name === 'edit' && <><Path d="M15.6 4.6a2.1 2.1 0 0 1 3 3L9 17.2l-4 1 1-4 9.6-9.6Z" {...s} /><Line x1="14.2" x2="17.2" y1="6" y2="9" {...s} /></>}
      {name === 'check' && <Polyline points="4.5,12.6 9.6,17.5 19.5,6.8" {...s} strokeWidth={strokeWidth + 0.3} />}
      {name === 'trash' && <Path d="M5.5 7.5h13M9 4.5h6M7.2 7.5l.7 12h8.2l.7-12M10 11v5M14 11v5" {...s} />}
      {name === 'copy' && <><Rect height="11.5" rx="2.2" width="10.5" x="8.5" y="8.5" {...s} /><Path d="M5.6 15.2A2.2 2.2 0 0 1 3.5 13V5.7a2.2 2.2 0 0 1 2.1-2.2h7.3a2.2 2.2 0 0 1 2.2 2.2" {...s} /></>}
      {name === 'folder' && <Path d="M3.5 6.6a1.8 1.8 0 0 1 1.8-1.8h3.6l2 2.6h7.8a1.8 1.8 0 0 1 1.8 1.8v8.4a1.8 1.8 0 0 1-1.8 1.8H5.3a1.8 1.8 0 0 1-1.8-1.8V6.6Z" {...s} />}

      {/* ---------- metrics ---------- */}
      {name === 'water' && <Path d="M12 2.9S6.3 9.4 6.3 14a5.7 5.7 0 1 0 11.4 0C17.7 9.4 12 2.9 12 2.9Z" {...s} />}
      {name === 'steps' && <><Path d="M9.4 4.2c1.7.9 1.9 4.2.8 6.5S6.7 14 5.1 13.2c-1.6-.8-1.3-3.1-.1-5.4s2.7-4.5 4.4-3.6Z" {...s} /><Path d="M15.8 12.1c1.6-.4 3 2.2 3.4 4.3.4 2.2-.6 3.5-2.4 3.9-1.8.3-2.9-1.4-3.3-3.5-.4-2.2.7-4.4 2.3-4.7Z" {...s} /></>}
      {name === 'sleep' && <Path d="M18.7 16.7A8.2 8.2 0 0 1 8.1 5.4a8.2 8.2 0 1 0 10.6 11.3Z" {...s} />}
      {name === 'flame' && <><Path d="M12 21c3.9 0 6.6-2.5 6.6-6.1 0-4.3-3.6-6.5-4.5-11.4-2.2 1.7-3.4 4-3.4 6.3 0 1.4-.7 2-1.5 2-.9 0-1.5-.7-1.6-1.9-1.4 1.6-2.2 3.5-2.2 5.3C5.4 18.5 8.1 21 12 21Z" {...s} /></>}
      {name === 'scale' && <><Rect height="12.5" rx="3" width="17" x="3.5" y="6" {...s} /><Path d="M12 9.4v2.6M8.6 10.6l1.4 1.8M15.4 10.6 14 12.4" {...s} /></>}
      {name === 'heart' && <Path d="M20.5 8.7c0 5.1-8.5 10-8.5 10s-8.5-4.9-8.5-10A4.5 4.5 0 0 1 12 6.6a4.5 4.5 0 0 1 8.5 2.1Z" {...s} />}
      {name === 'target' && <><Circle cx="12" cy="12" r="8.2" {...s} /><Circle cx="12" cy="12" r="4.4" {...s} /><Circle cx="12" cy="12" r="1.2" fill={color} stroke="none" /></>}
      {name === 'timer' && <><Circle cx="12" cy="13.4" r="7.6" {...s} /><Path d="M12 9.4v4h2.8M9.4 2.8h5.2" {...s} /></>}
      {name === 'calendar' && <><Rect height="15" rx="2.4" width="16.6" x="3.7" y="5" {...s} /><Path d="M3.7 9.6h16.6M8.4 3v3.6M15.6 3v3.6" {...s} /></>}
      {name === 'trophy' && <><Path d="M7.5 3.8h9v5.1a4.5 4.5 0 0 1-9 0V3.8Z" {...s} /><Path d="M7.5 5.4H5a2.3 2.3 0 0 0 2.5 3.4M16.5 5.4H19a2.3 2.3 0 0 1-2.5 3.4M12 13.4v3.4M8.8 20.2h6.4l-.7-3.4H9.5l-.7 3.4Z" {...s} /></>}
      {name === 'arrowUp' && <><Line x1="12" x2="12" y1="19.5" y2="5" {...s} /><Polyline points="6.4,10.6 12,5 17.6,10.6" {...s} /></>}
      {name === 'arrowDown' && <><Line x1="12" x2="12" y1="4.5" y2="19" {...s} /><Polyline points="6.4,13.4 12,19 17.6,13.4" {...s} /></>}
      {name === 'trend' && <><Polyline points="3.5,16.4 9,10.9 12.8,14.7 20.5,7" {...s} /><Polyline points="15.4,7 20.5,7 20.5,12.1" {...s} /></>}

      {/* ---------- training ---------- */}
      {name === 'play' && <Path d="M8 5.4 18.5 12 8 18.6V5.4Z" {...s} />}
      {name === 'pause' && <><Rect height="14" rx="1.5" width="3.6" x="7" y="5" {...s} /><Rect height="14" rx="1.5" width="3.6" x="13.4" y="5" {...s} /></>}
      {name === 'restart' && <><Path d="M20 12a8 8 0 1 1-2.6-5.9" {...s} /><Polyline points="20.3,3.6 20.3,8.2 15.7,8.2" {...s} /></>}
      {name === 'muscle' && <Path d="M4 15.2c0-3.3 1.7-4.9 4.2-4.9 1.6 0 2.4.7 3.6.7 1 0 1.4-.6 1.4-1.6 0-1.3-.8-2-.8-3.3 0-1.4 1-2.3 2.4-2.3 2.6 0 5.2 3.1 5.2 7.4 0 4.6-3.2 8.2-7.9 8.2C7.4 19.4 4 17.8 4 15.2Z" {...s} />}
      {name === 'book' && <><Path d="M4 5.2a1.7 1.7 0 0 1 1.7-1.7H18a1.7 1.7 0 0 1 1.7 1.7v13.6a1.7 1.7 0 0 0-1.7-1.7H5.7A1.7 1.7 0 0 1 4 15.4V5.2Z" {...s} /><Path d="M4 18.8a1.7 1.7 0 0 1 1.7-1.7H18M8.4 7.6h6.4M8.4 11h4.2" {...s} /></>}
      {name === 'video' && <><Rect height="12.4" rx="2.6" width="13.2" x="2.8" y="5.8" {...s} /><Path d="m16 12 5.2-3.1v6.2L16 12Z" {...s} /></>}

      {/* ---------- system ---------- */}
      {name === 'watch' && <><Rect height="12" rx="3" width="10" x="7" y="6" {...s} /><Path d="M9 6 10 2h4l1 4M9 18l1 4h4l1-4" {...s} /><Circle cx="12" cy="12" r="2.4" {...s} /></>}
      {name === 'settings' && <><Circle cx="12" cy="12" r="3.1" {...s} /><Path d="M18.9 14.6a1.5 1.5 0 0 0 .3 1.7l.1.1a1.9 1.9 0 1 1-2.7 2.7l-.1-.1a1.5 1.5 0 0 0-2.5 1v.2a1.9 1.9 0 1 1-3.8 0v-.1a1.5 1.5 0 0 0-2.6-1l-.1.1a1.9 1.9 0 1 1-2.7-2.7l.1-.1a1.5 1.5 0 0 0-1-2.5h-.2a1.9 1.9 0 1 1 0-3.8h.1a1.5 1.5 0 0 0 1-2.6l-.1-.1a1.9 1.9 0 1 1 2.7-2.7l.1.1a1.5 1.5 0 0 0 1.7.3h.1a1.5 1.5 0 0 0 .9-1.4v-.2a1.9 1.9 0 0 1 3.8 0v.1a1.5 1.5 0 0 0 2.5 1l.1-.1a1.9 1.9 0 1 1 2.7 2.7l-.1.1a1.5 1.5 0 0 0-.3 1.7v.1a1.5 1.5 0 0 0 1.4.9h.2a1.9 1.9 0 0 1 0 3.8h-.1a1.5 1.5 0 0 0-1.4.9Z" {...s} strokeWidth={1.5} /></>}
      {name === 'user' && <><Circle cx="12" cy="8.2" r="4" {...s} /><Path d="M4.8 20.2a7.4 7.4 0 0 1 14.4 0" {...s} /></>}
      {name === 'lock' && <><Rect height="10.4" rx="2.4" width="14" x="5" y="10" {...s} /><Path d="M8.2 10V7.6a3.8 3.8 0 0 1 7.6 0V10" {...s} /></>}
      {name === 'shield' && <><Path d="M12 3.2 4.8 6v6c0 4.3 3 7.4 7.2 8.8 4.2-1.4 7.2-4.5 7.2-8.8V6L12 3.2Z" {...s} /><Polyline points="9.2,11.9 11.4,14.1 15,10.5" {...s} /></>}
      {name === 'cloud' && <Path d="M7.4 18.4a4.4 4.4 0 0 1-.5-8.8 5.6 5.6 0 0 1 10.8-1 3.9 3.9 0 0 1-.9 9.8H7.4Z" {...s} />}
      {name === 'download' && <><Path d="M12 3.6v10.8" {...s} /><Polyline points="7.6,10.4 12,14.6 16.4,10.4" {...s} /><Path d="M4.4 17.4v1.6a1.6 1.6 0 0 0 1.6 1.6h12a1.6 1.6 0 0 0 1.6-1.6v-1.6" {...s} /></>}
      {name === 'share' && <><Path d="M12 15.4V3.8" {...s} /><Polyline points="7.8,7.8 12,3.6 16.2,7.8" {...s} /><Path d="M4.6 13.6v5.2a1.6 1.6 0 0 0 1.6 1.6h11.6a1.6 1.6 0 0 0 1.6-1.6v-5.2" {...s} /></>}
      {name === 'bell' && <><Path d="M18 9.4a6 6 0 1 0-12 0c0 5.2-2 6.6-2 6.6h16s-2-1.4-2-6.6Z" {...s} /><Path d="M13.7 19.4a2 2 0 0 1-3.4 0" {...s} /></>}
      {name === 'info' && <><Circle cx="12" cy="12" r="8.6" {...s} /><Path d="M12 11.2v5M12 7.9h.01" {...s} /></>}
      {name === 'alert' && <><Path d="M10.6 3.9 2.5 18.2a1.6 1.6 0 0 0 1.4 2.4h16.2a1.6 1.6 0 0 0 1.4-2.4L13.4 3.9a1.6 1.6 0 0 0-2.8 0Z" {...s} /><Path d="M12 9.4v4M12 17.2h.01" {...s} /></>}
      {name === 'star' && <Path d="m12 3.4 2.7 5.6 6.1.8-4.5 4.3 1.1 6.1-5.4-3-5.4 3 1.1-6.1L3.2 9.8l6.1-.8L12 3.4Z" {...s} />}
      {name === 'link' && <><Path d="M10.2 13.8a3.8 3.8 0 0 0 5.7.4l2.3-2.3a3.8 3.8 0 0 0-5.4-5.4l-1.3 1.3" {...s} /><Path d="M13.8 10.2a3.8 3.8 0 0 0-5.7-.4l-2.3 2.3a3.8 3.8 0 0 0 5.4 5.4l1.3-1.3" {...s} /></>}
      {name === 'logout' && <><Path d="M9.4 20.4H5.8a1.8 1.8 0 0 1-1.8-1.8V5.4a1.8 1.8 0 0 1 1.8-1.8h3.6" {...s} /><Polyline points="15.4,16.4 19.8,12 15.4,7.6" {...s} /><Line x1="19.8" x2="9.4" y1="12" y2="12" {...s} /></>}
      {name === 'sun' && <><Circle cx="12" cy="12" r="4.2" {...s} /><Path d="M12 2.4v2.2M12 19.4v2.2M4.6 12H2.4M21.6 12h-2.2M6.4 6.4 4.8 4.8M19.2 19.2l-1.6-1.6M17.6 6.4l1.6-1.6M4.8 19.2l1.6-1.6" {...s} /></>}
    </Svg>
  );
}
