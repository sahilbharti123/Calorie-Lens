import type { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { Tabs } from 'expo-router';
import React, { useEffect } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import { useReducedMotion } from '@/src/lib/accessibility';
import { motion, palette, radius, text } from '@/src/theme';

const ICONS: Record<string, GlyphName> = {
  index: 'home',
  food: 'bowl',
  train: 'dumbbell',
  progress: 'trend',
  coach: 'spark',
};

const LABELS: Record<string, string> = {
  index: 'Today',
  food: 'Food',
  train: 'Train',
  progress: 'Progress',
  coach: 'Coach',
};

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{ headerShown: false, sceneStyle: { backgroundColor: palette.bg } }}
      tabBar={(props) => <GlassTabBar {...props} />}>
      <Tabs.Screen name="index" options={{ title: 'Today' }} />
      <Tabs.Screen name="food" options={{ title: 'Food' }} />
      <Tabs.Screen name="train" options={{ title: 'Train' }} />
      <Tabs.Screen name="progress" options={{ title: 'Progress' }} />
      <Tabs.Screen name="coach" options={{ title: 'Coach' }} />
    </Tabs>
  );
}

function GlassTabBar({ state, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  return (
    <View style={[styles.wrap, { bottom: Math.max(insets.bottom, 8) }]}>
      <BlurView intensity={Platform.OS === 'ios' ? 60 : 0} style={StyleSheet.absoluteFill} tint="dark" />
      <View style={[StyleSheet.absoluteFill, styles.tint]} />
      <View style={styles.row}>
        {state.routes.map((route, index) => {
          const focused = state.index === index;
          return (
            <TabButton
              focused={focused}
              icon={ICONS[route.name] ?? 'home'}
              key={route.key}
              label={LABELS[route.name] ?? route.name}
              onPress={() => {
                const event = navigation.emit({ type: 'tabPress', target: route.key, canPreventDefault: true });
                if (!focused && !event.defaultPrevented) {
                  void Haptics.selectionAsync();
                  navigation.navigate(route.name);
                }
              }}
            />
          );
        })}
      </View>
    </View>
  );
}

function TabButton({
  focused,
  icon,
  label,
  onPress,
}: {
  focused: boolean;
  icon: GlyphName;
  label: string;
  onPress: () => void;
}) {
  const reducedMotion = useReducedMotion();
  const on = useSharedValue(focused ? 1 : 0);

  useEffect(() => {
    on.value = reducedMotion
      ? (focused ? 1 : 0)
      : withSpring(focused ? 1 : 0, motion.enter);
  }, [focused, on, reducedMotion]);

  const halo = useAnimatedStyle(() => ({
    opacity: on.value,
    transform: [{ scale: 0.6 + on.value * 0.4 }],
  }));

  const lift = useAnimatedStyle(() => ({
    transform: [{ translateY: -on.value * 2 }],
  }));

  return (
    <Pressable
      accessibilityRole="tab"
      accessibilityState={{ selected: focused }}
      onPress={onPress}
      style={styles.tab}>
      <Animated.View style={[styles.iconWrap, lift]}>
        <Animated.View style={[styles.halo, halo]} />
        <Glyph color={focused ? palette.lime : palette.inkLow} name={icon} size={21} />
      </Animated.View>
      <Text style={[styles.label, focused && styles.labelOn]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: 12,
    right: 12,
    borderWidth: 1,
    borderColor: palette.lineHi,
    borderRadius: 28,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.38,
    shadowRadius: 20,
    elevation: 16,
  },
  tint: { backgroundColor: Platform.OS === 'ios' ? 'rgba(7,9,10,0.62)' : 'rgba(7,9,10,0.97)' },
  row: { flexDirection: 'row', paddingVertical: 9 },
  tab: { flex: 1, alignItems: 'center', gap: 2 },
  iconWrap: { width: 42, height: 30, alignItems: 'center', justifyContent: 'center' },
  halo: {
    position: 'absolute',
    width: 42,
    height: 30,
    borderRadius: radius.pill,
    backgroundColor: 'rgba(198, 255, 60, 0.14)',
  },
  label: { ...text.micro, fontSize: 9.5, color: palette.inkLow },
  labelOn: { color: palette.lime, fontFamily: text.value.fontFamily },
});
