import type { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { Tabs } from 'expo-router';
import React, { useEffect } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Glyph, type GlyphName } from '@/src/components/glyph';
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
    <View style={styles.wrap}>
      <BlurView intensity={Platform.OS === 'ios' ? 60 : 0} style={StyleSheet.absoluteFill} tint="dark" />
      <View style={[StyleSheet.absoluteFill, styles.tint]} />
      <View style={[styles.row, { paddingBottom: Math.max(insets.bottom, 10) }]}>
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
  const on = useSharedValue(focused ? 1 : 0);

  useEffect(() => {
    on.value = withSpring(focused ? 1 : 0, motion.enter);
  }, [focused, on]);

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
    left: 0,
    right: 0,
    bottom: 0,
    borderTopWidth: 1,
    borderTopColor: palette.line,
    overflow: 'hidden',
  },
  tint: { backgroundColor: Platform.OS === 'ios' ? 'rgba(7,9,10,0.62)' : 'rgba(7,9,10,0.97)' },
  row: { flexDirection: 'row', paddingTop: 10 },
  tab: { flex: 1, alignItems: 'center', gap: 3 },
  iconWrap: { width: 40, height: 28, alignItems: 'center', justifyContent: 'center' },
  halo: {
    position: 'absolute',
    width: 40,
    height: 28,
    borderRadius: radius.pill,
    backgroundColor: 'rgba(198, 255, 60, 0.14)',
  },
  label: { ...text.micro, fontSize: 9.5, color: palette.inkLow },
  labelOn: { color: palette.lime, fontFamily: text.value.fontFamily },
});
