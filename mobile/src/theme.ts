import { Platform } from 'react-native';

export const palette = {
  canvas: '#F4F6EF',
  paper: '#FBFCF7',
  forest: '#152019',
  ink: '#18211B',
  muted: '#687169',
  line: '#DDE2D8',
  lime: '#B8EF5A',
  limeDark: '#7DAF2E',
  coral: '#D97A63',
  softLime: '#E8F7CD',
  softCoral: '#F3DDD6',
  white: '#FFFFFF',
};

export const type = {
  regular: Platform.select({ ios: 'AvenirNext-Regular', android: 'sans-serif', default: 'system-ui' }),
  medium: Platform.select({ ios: 'AvenirNext-Medium', android: 'sans-serif-medium', default: 'system-ui' }),
  demi: Platform.select({ ios: 'AvenirNext-DemiBold', android: 'sans-serif-medium', default: 'system-ui' }),
  bold: Platform.select({ ios: 'AvenirNext-Bold', android: 'sans-serif', default: 'system-ui' }),
};

export const space = { xs: 6, sm: 10, md: 16, lg: 22, xl: 30, xxl: 42 };
export const radius = { sm: 12, md: 18, lg: 26, pill: 999 };
