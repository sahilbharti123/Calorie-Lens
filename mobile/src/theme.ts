import type { TextStyle } from 'react-native';

/**
 * Calorie Lens — "Midnight Athlete" design system.
 *
 * One dark, high-contrast surface stack with a single electric accent.
 * Everything in the app reads from these tokens; no screen defines its own
 * colour, radius, or type size.
 */

export const palette = {
  /** App canvas. Near-black with a faint green undertone so lime sits warm on it. */
  bg: '#07090A',
  /** Standard card. */
  surface: '#0F1315',
  /** Raised card (hero, sheets, active rows). */
  surfaceHi: '#14181A',
  /** Inset well — chips and fields *inside* a card. */
  surfaceLo: '#0A0E0F',
  /** Pressed / hovered surface. */
  surfacePress: '#191E21',

  /** Hairline divider. */
  line: '#1A2023',
  /** Visible border on raised surfaces. */
  lineHi: '#232B2E',

  /** Primary text. */
  ink: '#F2F6F0',
  /** Secondary text, values, captions. 6.1:1 on `surface`. */
  inkMid: '#8B9789',
  /**
   * Labels and tertiary text. Kept at ≥4.6:1 against every surface in the
   * stack so small uppercase labels still pass WCAG AA — the older, dimmer
   * value looked better in a mockup and was unreadable on a phone outdoors.
   */
  inkLow: '#78876F',
  /** Non-text only: dividers, inactive glyph strokes, chart gridlines. */
  inkFaint: '#4E5A4C',

  /** The accent. Used for progress, primary actions, and live state. */
  lime: '#C6FF3C',
  limeDeep: '#93E22B',
  limeSoft: '#1C2A0F',
  /** Text/icons drawn on top of a lime fill. */
  onLime: '#08240A',

  /** Macro + chart hues. Distinct at a glance, all AA on the dark canvas. */
  protein: '#C6FF3C',
  carbs: '#5AB4FF',
  fat: '#FF9F5A',

  /** Semantic. */
  danger: '#FF6B5A',
  warn: '#FFC65A',
  info: '#5AB4FF',
  white: '#FFFFFF',
} as const;

/** Translucent layers — for glows, scrims, and glass. */
export const alpha = {
  limeGlow: 'rgba(198, 255, 60, 0.26)',
  limeGlowSoft: 'rgba(198, 255, 60, 0.12)',
  limeFaint: 'rgba(198, 255, 60, 0.07)',
  blueGlow: 'rgba(90, 180, 255, 0.16)',
  scrim: 'rgba(4, 6, 6, 0.72)',
  onLimeSoft: 'rgba(10, 36, 5, 0.16)',
  white08: 'rgba(255, 255, 255, 0.08)',
  white12: 'rgba(255, 255, 255, 0.12)',
} as const;

/**
 * Type families. Space Grotesk carries the display numerals and screen titles
 * (it has real character — the app should not look like a default template);
 * Inter carries everything you actually read.
 */
export const font = {
  display: 'SpaceGrotesk_700Bold',
  displayMedium: 'SpaceGrotesk_500Medium',
  regular: 'Inter_400Regular',
  medium: 'Inter_500Medium',
  semi: 'Inter_600SemiBold',
  bold: 'Inter_700Bold',
  black: 'Inter_800ExtraBold',
} as const;

/** Back-compat alias for the pre-redesign token name. */
export const type = {
  regular: font.regular,
  medium: font.medium,
  demi: font.semi,
  bold: font.bold,
} as const;

/**
 * Type scale. `n` marks styles that should render tabular figures so numbers
 * do not jitter while they count up.
 */
export const text: Record<
  | 'hero' | 'title' | 'headline' | 'section' | 'row'
  | 'body' | 'value' | 'caption' | 'label' | 'micro',
  TextStyle
> = {
  /** 44 — the one big number on a screen. */
  hero: { fontFamily: font.display, fontSize: 44, letterSpacing: -2, lineHeight: 46 },
  /** 30 — screen titles. */
  title: { fontFamily: font.display, fontSize: 28, letterSpacing: -1.1, lineHeight: 33 },
  /** 22 — card headline / large value. */
  headline: { fontFamily: font.display, fontSize: 21, letterSpacing: -0.6, lineHeight: 26 },
  /** 17 — section heading. */
  section: { fontFamily: font.bold, fontSize: 16, letterSpacing: -0.3, lineHeight: 21 },
  /** 15 — row title. */
  row: { fontFamily: font.semi, fontSize: 14.5, letterSpacing: -0.1, lineHeight: 19 },
  /** 14 — body copy. */
  body: { fontFamily: font.regular, fontSize: 13.5, lineHeight: 19.5 },
  /** 13 — value in a row. */
  value: { fontFamily: font.semi, fontSize: 13, letterSpacing: -0.1, lineHeight: 17 },
  /** 12 — caption / supporting. */
  caption: { fontFamily: font.regular, fontSize: 11.5, lineHeight: 16 },
  /** 10 — ALL-CAPS eyebrow label. Always pair with `letterSpacing`. */
  label: { fontFamily: font.bold, fontSize: 9.5, letterSpacing: 1.5, lineHeight: 12 },
  /** 11 — tab bar + tiny meta. */
  micro: { fontFamily: font.medium, fontSize: 10.5, lineHeight: 13 },
};

/** Renders digits at a fixed width so counting numbers do not shift layout. */
export const tabular: Pick<TextStyle, 'fontVariant'> = { fontVariant: ['tabular-nums'] };

export const space = {
  xs: 6,
  sm: 10,
  md: 16,
  lg: 22,
  xl: 30,
  xxl: 42,
  /** Bottom padding on every scroll view so content clears the tab bar. */
  tabClearance: 104,
} as const;

export const radius = { xs: 8, sm: 12, md: 18, lg: 24, xl: 30, pill: 999 } as const;

/** Shadows. On dark surfaces a glow reads better than a drop shadow. */
export const shadow = {
  card: {
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 8 },
    elevation: 6,
  },
  glow: {
    shadowColor: palette.lime,
    shadowOpacity: 0.3,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: 6 },
    elevation: 10,
  },
  glowSoft: {
    shadowColor: palette.lime,
    shadowOpacity: 0.16,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 4 },
    elevation: 5,
  },
} as const;

/** Motion. Springs for anything a finger touched; timing for anything else. */
export const motion = {
  press: { damping: 15, stiffness: 420, mass: 0.55 },
  enter: { damping: 18, stiffness: 180, mass: 0.9 },
  bouncy: { damping: 11, stiffness: 220, mass: 0.8 },
  quick: 180,
  base: 280,
  slow: 620,
  /** Delay between items in a staggered list reveal. */
  stagger: 55,
} as const;

/** Gradient stop pairs, so gradients stay consistent between screens. */
export const gradient = {
  lime: [palette.lime, palette.limeDeep] as const,
  hero: ['#151A1C', '#0B0F11'] as const,
  card: ['#12171A', '#0D1113'] as const,
  fade: ['rgba(7,9,10,0)', palette.bg] as const,
  ring: [palette.lime, '#5FD97A'] as const,
} as const;

/** Per-macro colour lookup used by bars, rings, and legends. */
export const macroColor = {
  protein: palette.protein,
  carbs: palette.carbs,
  fat: palette.fat,
} as const;
