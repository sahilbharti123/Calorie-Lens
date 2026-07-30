import { useEffect, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import { photosFor } from '@/src/lib/exercise-photos';
import { alpha, palette, radius, space, tabular, text } from '@/src/theme';

/**
 * Real photo demonstration: crossfades between the start and end position of
 * the movement (public-domain photos from free-exercise-db). Tap to pause on
 * either position.
 */
export function PhotoDemo({
  exerciseId,
  height = 230,
  style,
}: {
  exerciseId: string;
  height?: number;
  /** Lets a screen drop the frame's radius/border for a full-bleed hero. */
  style?: StyleProp<ViewStyle>;
}) {
  const pair = photosFor(exerciseId);
  const fade = useRef(new Animated.Value(0)).current;
  const [paused, setPaused] = useState(false);
  const [position, setPosition] = useState<1 | 2>(1);

  useEffect(() => {
    if (!pair || paused) return;
    let mounted = true;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.delay(650),
        Animated.timing(fade, { toValue: 1, duration: 700, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
        Animated.delay(650),
        Animated.timing(fade, { toValue: 0, duration: 700, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
      ]),
    );
    loop.start();
    const listener = fade.addListener(({ value }) => {
      if (mounted) setPosition(value > 0.5 ? 2 : 1);
    });
    return () => {
      mounted = false;
      fade.removeListener(listener);
      loop.stop();
    };
  }, [fade, pair, paused]);

  if (!pair) return null;

  return (
    <Pressable
      accessibilityLabel="Exercise demonstration photos. Tap to pause."
      onPress={() => {
        setPaused((current) => {
          if (!current) return true;
          return false;
        });
        if (paused) {
          // resuming — nothing else needed; effect restarts the loop
        } else {
          fade.stopAnimation((value) => fade.setValue(value > 0.5 ? 1 : 0));
        }
      }}
      style={[styles.frame, { height }, style]}>
      <Image resizeMode="contain" source={pair[0]} style={styles.photo} />
      <Animated.Image resizeMode="contain" source={pair[1]} style={[styles.photo, { opacity: fade }]} />
      <View style={styles.badgeRow}>
        <View style={[styles.dot, position === 1 && styles.dotActive]} />
        <View style={[styles.dot, position === 2 && styles.dotActive]} />
        <Text style={styles.badgeText}>{position === 1 ? 'START' : 'FINISH'}{paused ? ' · PAUSED' : ''}</Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  frame: {
    width: '100%',
    borderRadius: radius.md,
    backgroundColor: palette.surface,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: palette.line,
  },
  photo: { ...StyleSheet.absoluteFillObject, width: '100%', height: '100%' },
  badgeRow: {
    position: 'absolute',
    left: space.md,
    bottom: space.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: alpha.scrim,
    borderWidth: 1,
    borderColor: alpha.white12,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: palette.inkLow },
  dotActive: { backgroundColor: palette.lime },
  badgeText: { ...text.label, fontSize: 9, letterSpacing: 1, color: palette.lime, marginLeft: 3, ...tabular },
});
