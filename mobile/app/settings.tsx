import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, type } from '@/src/theme';
import type { Goals } from '@/src/types';

export default function SettingsScreen() {
  const router = useRouter();
  const { data, updateGoals } = useApp();
  const [values, setValues] = useState<Record<keyof Goals, string>>({
    calories: String(data.goals.calories),
    protein: String(data.goals.protein),
    waterMl: String(data.goals.waterMl),
    steps: String(data.goals.steps),
  });

  function save() {
    updateGoals({
      calories: Math.max(500, Number(values.calories) || data.goals.calories),
      protein: Math.max(10, Number(values.protein) || data.goals.protein),
      waterMl: Math.max(250, Number(values.waterMl) || data.goals.waterMl),
      steps: Math.max(500, Number(values.steps) || data.goals.steps),
    });
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    router.dismiss();
  }

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <View style={styles.intro}>
            <View style={styles.icon}><Glyph name="chart" color={palette.forest} size={24} /></View>
            <Text style={styles.title}>Set a pace you can keep.</Text>
            <Text style={styles.body}>These goals shape your daily dashboard. You can change them any time.</Text>
          </View>

          <View style={styles.form}>
            <GoalField label="Daily calories" unit="kcal" value={values.calories} onChange={(value) => setValues((current) => ({ ...current, calories: value }))} />
            <GoalField label="Daily protein" unit="g" value={values.protein} onChange={(value) => setValues((current) => ({ ...current, protein: value }))} />
            <GoalField label="Daily water" unit="ml" value={values.waterMl} onChange={(value) => setValues((current) => ({ ...current, waterMl: value }))} />
            <GoalField label="Daily steps" unit="steps" value={values.steps} onChange={(value) => setValues((current) => ({ ...current, steps: value }))} />
          </View>

          <Pressable onPress={save} style={styles.saveButton}><Text style={styles.saveText}>Save goals</Text></Pressable>
          <Text style={styles.note}>Calorie Lens is a personal tracking tool, not a medical service.</Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function GoalField({ label, unit, value, onChange }: { label: string; unit: string; value: string; onChange: (value: string) => void }) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <View style={styles.inputRow}>
        <TextInput
          keyboardType="number-pad"
          selectTextOnFocus
          value={value}
          onChangeText={(next) => onChange(next.replace(/[^\d.]/g, ''))}
          style={styles.input}
        />
        <Text style={styles.unit}>{unit}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  content: { padding: space.md, paddingBottom: 34 },
  intro: { alignItems: 'center', paddingHorizontal: 22, paddingTop: 10, marginBottom: 24 },
  icon: { width: 48, height: 48, borderRadius: 16, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  title: { color: palette.ink, fontFamily: type.demi, fontSize: 23, letterSpacing: -0.6, textAlign: 'center' },
  body: { color: palette.muted, fontFamily: type.regular, fontSize: 12, lineHeight: 18, textAlign: 'center', marginTop: 6 },
  form: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 15 },
  field: { minHeight: 72, borderBottomWidth: 1, borderBottomColor: palette.line, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  label: { color: palette.ink, fontFamily: type.medium, fontSize: 13 },
  inputRow: { flexDirection: 'row', alignItems: 'baseline', gap: 5 },
  input: { minWidth: 72, color: palette.ink, fontFamily: type.demi, fontSize: 17, textAlign: 'right', paddingVertical: 10 },
  unit: { width: 38, color: palette.muted, fontFamily: type.regular, fontSize: 10 },
  saveButton: { height: 54, backgroundColor: palette.forest, borderRadius: radius.md, alignItems: 'center', justifyContent: 'center', marginTop: 16 },
  saveText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
  note: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, textAlign: 'center', marginTop: 12 },
});
