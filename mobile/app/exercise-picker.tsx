import { useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import {
  FlatList,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ExerciseFigure } from '@/src/components/exercise-figure';
import { emitExercisePick } from '@/src/lib/exercise-pick-bus';
import {
  EQUIPMENT_TYPES,
  MUSCLE_GROUPS,
  searchExercises,
  type Equipment,
  type Exercise,
  type MuscleGroup,
} from '@/src/lib/exercises';
import { palette, radius, space, type } from '@/src/theme';

export default function ExercisePickerScreen() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [muscle, setMuscle] = useState<MuscleGroup | null>(null);
  const [equipment, setEquipment] = useState<Equipment | null>(null);
  const [selected, setSelected] = useState<string[]>([]);

  const results = useMemo(
    () => searchExercises({ query, muscle, equipment }),
    [query, muscle, equipment],
  );

  function toggle(exercise: Exercise) {
    setSelected((current) => current.includes(exercise.id)
      ? current.filter((id) => id !== exercise.id)
      : [...current, exercise.id]);
  }

  function confirm() {
    if (!selected.length) return;
    emitExercisePick(selected);
    router.back();
  }

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <View style={styles.searchRow}>
        <View style={styles.searchBox}>
          <TextInput
            autoCorrect={false}
            placeholder="Search exercises"
            placeholderTextColor="#8A938B"
            style={styles.searchInput}
            value={query}
            onChangeText={setQuery}
          />
        </View>
      </View>

      <View style={styles.filters}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chipRow}>
          <FilterChip label="All muscles" active={muscle === null} onPress={() => setMuscle(null)} />
          {MUSCLE_GROUPS.map((group) => (
            <FilterChip
              key={group}
              label={group}
              active={muscle === group}
              onPress={() => setMuscle(muscle === group ? null : group)}
            />
          ))}
        </ScrollView>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chipRow}>
          <FilterChip label="All equipment" active={equipment === null} onPress={() => setEquipment(null)} />
          {EQUIPMENT_TYPES.map((kind) => (
            <FilterChip
              key={kind}
              label={kind}
              active={equipment === kind}
              onPress={() => setEquipment(equipment === kind ? null : kind)}
            />
          ))}
        </ScrollView>
      </View>

      <FlatList
        data={results}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.list}
        keyboardShouldPersistTaps="handled"
        renderItem={({ item }) => {
          const isSelected = selected.includes(item.id);
          return (
            <Pressable onPress={() => toggle(item)} style={[styles.row, isSelected && styles.rowSelected]}>
              <View style={styles.thumb}>
                <ExerciseFigure template={item.template} gear={item.gear} size={52} paused />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.rowName}>{item.name}</Text>
                <Text style={styles.rowMeta}>
                  {item.primaryMuscle}{item.secondaryMuscles.length ? ` · +${item.secondaryMuscles.join(', ')}` : ''}
                </Text>
                <Text style={styles.rowEquip}>{item.equipment} · {item.kind === 'duration' ? 'timed' : item.kind === 'reps-only' ? 'bodyweight reps' : 'weight × reps'}</Text>
              </View>
              <Pressable
                hitSlop={8}
                onPress={() => router.push({ pathname: '/exercise/[id]', params: { id: item.id } })}>
                <Text style={styles.info}>How to</Text>
              </Pressable>
              <View style={[styles.check, isSelected && styles.checkOn]}>
                {isSelected ? <Text style={styles.checkMark}>✓</Text> : null}
              </View>
            </Pressable>
          );
        }}
        ListEmptyComponent={(
          <Text style={styles.empty}>No exercises match. Try another muscle or equipment filter.</Text>
        )}
      />

      <View style={styles.footer}>
        <Pressable
          disabled={!selected.length}
          onPress={confirm}
          style={({ pressed }) => [styles.addButton, !selected.length && styles.disabled, pressed && styles.pressed]}>
          <Text style={styles.addText}>
            {selected.length ? `Add ${selected.length} exercise${selected.length > 1 ? 's' : ''}` : 'Select exercises'}
          </Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

function FilterChip({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={[styles.chip, active && styles.chipActive]}>
      <Text style={[styles.chipText, active && styles.chipTextActive]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  searchRow: { paddingHorizontal: space.md, paddingTop: 10 },
  searchBox: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 14 },
  searchInput: { height: 46, color: palette.ink, fontFamily: type.medium, fontSize: 14 },
  filters: { gap: 6, paddingVertical: 10 },
  chipRow: { gap: 6, paddingHorizontal: space.md },
  chip: { height: 32, paddingHorizontal: 13, borderRadius: radius.pill, borderWidth: 1, borderColor: palette.line, backgroundColor: palette.paper, alignItems: 'center', justifyContent: 'center' },
  chipActive: { backgroundColor: palette.forest, borderColor: palette.forest },
  chipText: { color: palette.ink, fontFamily: type.medium, fontSize: 11, textTransform: 'capitalize' },
  chipTextActive: { color: palette.lime, fontFamily: type.demi },
  list: { paddingHorizontal: space.md, paddingBottom: 12, gap: 7 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 11, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 10 },
  rowSelected: { borderColor: palette.limeDark, backgroundColor: '#F4FBE6' },
  thumb: { width: 56, height: 56, borderRadius: radius.sm, backgroundColor: palette.canvas, alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
  rowName: { color: palette.ink, fontFamily: type.demi, fontSize: 13 },
  rowMeta: { color: palette.muted, fontFamily: type.regular, fontSize: 10, marginTop: 2, textTransform: 'capitalize' },
  rowEquip: { color: palette.limeDark, fontFamily: type.medium, fontSize: 9.5, marginTop: 2, textTransform: 'capitalize' },
  info: { color: palette.coral, fontFamily: type.demi, fontSize: 10.5, paddingHorizontal: 4 },
  check: { width: 26, height: 26, borderRadius: 13, borderWidth: 1.5, borderColor: palette.line, alignItems: 'center', justifyContent: 'center' },
  checkOn: { backgroundColor: palette.forest, borderColor: palette.forest },
  checkMark: { color: palette.lime, fontSize: 13, fontFamily: type.demi },
  empty: { color: palette.muted, fontFamily: type.regular, fontSize: 12, textAlign: 'center', paddingVertical: 30 },
  footer: { padding: space.md, borderTopWidth: 1, borderTopColor: palette.line, backgroundColor: palette.canvas },
  addButton: { height: 52, borderRadius: radius.md, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  addText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
  disabled: { opacity: 0.45 },
  pressed: { opacity: 0.85 },
});
