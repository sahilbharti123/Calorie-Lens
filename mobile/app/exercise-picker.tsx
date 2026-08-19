import { type Href, useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import {
  FlatList,
  Image,
  Keyboard,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { ExerciseFigure } from '@/src/components/exercise-figure';
import { Glyph } from '@/src/components/glyph';
import {
  Card,
  Chip,
  EmptyState,
  GlassFooter,
  PrimaryButton,
  Reveal,
  Screen,
  Tap,
} from '@/src/components/ui';
import { emitExercisePick } from '@/src/lib/exercise-pick-bus';
import { photosFor } from '@/src/lib/exercise-photos';
import {
  EQUIPMENT_TYPES,
  MUSCLE_GROUPS,
  searchExercises,
  type Equipment,
  type Exercise,
  type MuscleGroup,
} from '@/src/lib/exercises';
import { exerciseInfo } from '@/src/lib/training';
import { useApp } from '@/src/store/app-store';
import { palette, radius, space, tabular, text } from '@/src/theme';

/** 'full body' → 'Full Body'. The data is lower-case; the chips are not. */
function titleCase(value: string) {
  return value.replace(/(^|\s)\S/g, (character) => character.toUpperCase());
}

type PickerExercise = Pick<Exercise, 'id' | 'name' | 'primaryMuscle' | 'secondaryMuscles' | 'equipment' | 'kind' | 'template' | 'gear'> & { custom?: boolean };

export default function ExercisePickerScreen() {
  const router = useRouter();
  const { data } = useApp();
  const [query, setQuery] = useState('');
  const [muscle, setMuscle] = useState<MuscleGroup | null>(null);
  const [equipment, setEquipment] = useState<Equipment | null>(null);
  const [selected, setSelected] = useState<string[]>([]);

  const results = useMemo<PickerExercise[]>(() => {
    const needle = query.trim().toLowerCase();
    const custom = data.training.customExercises
      .map((exercise) => exerciseInfo(data.training, exercise.id))
      .filter((exercise) => !needle || exercise.name.toLowerCase().includes(needle))
      .filter((exercise) => !muscle || exercise.primaryMuscle === muscle)
      .filter((exercise) => !equipment || exercise.equipment === equipment);
    return [...custom, ...searchExercises({ query, muscle, equipment })];
  }, [data.training, query, muscle, equipment]);

  function toggle(exercise: PickerExercise) {
    setSelected((current) => current.includes(exercise.id)
      ? current.filter((id) => id !== exercise.id)
      : [...current, exercise.id]);
  }

  function confirm() {
    if (!selected.length) return;
    emitExercisePick(selected);
    router.back();
  }

  function clearFilters() {
    setQuery('');
    setMuscle(null);
    setEquipment(null);
  }

  return (
    <Screen edges={['bottom']}>
      <Reveal>
        <View style={styles.searchRow}>
          <View style={styles.searchBox}>
            <Glyph color={palette.inkLow} name="search" size={17} />
            <TextInput
              accessibilityLabel="Search exercises"
              autoCorrect={false}
              onChangeText={setQuery}
              onSubmitEditing={Keyboard.dismiss}
              placeholder="Search exercises"
              placeholderTextColor={palette.inkLow}
              selectionColor={palette.lime}
              returnKeyType="done"
              style={styles.searchInput}
              value={query}
            />
            {query ? (
              <Tap accessibilityLabel="Clear search" hitSlop={8} onPress={() => setQuery('')} scaleTo={0.88}>
                <View style={styles.searchClear}>
                  <Glyph color={palette.inkMid} name="close" size={13} />
                </View>
              </Tap>
            ) : null}
          </View>
        </View>
      </Reveal>

      <Reveal index={1}>
        <View style={styles.filters}>
          <ScrollView contentContainerStyle={styles.chipRow} horizontal showsHorizontalScrollIndicator={false}>
            <Chip active={muscle === null} icon="muscle" label="All muscles" onPress={() => setMuscle(null)} />
            {MUSCLE_GROUPS.map((group) => (
              <Chip
                active={muscle === group}
                key={group}
                label={titleCase(group)}
                onPress={() => setMuscle(muscle === group ? null : group)}
              />
            ))}
          </ScrollView>
          <ScrollView contentContainerStyle={styles.chipRow} horizontal showsHorizontalScrollIndicator={false}>
            <Chip active={equipment === null} icon="dumbbell" label="All equipment" onPress={() => setEquipment(null)} />
            {EQUIPMENT_TYPES.map((kind) => (
              <Chip
                active={equipment === kind}
                key={kind}
                label={titleCase(kind)}
                onPress={() => setEquipment(equipment === kind ? null : kind)}
              />
            ))}
          </ScrollView>
        </View>
      </Reveal>

      <Reveal index={2}>
        <Tap
          accessibilityLabel="Create a custom exercise"
          onPress={() => router.push('/custom-exercise' as Href)}
          scaleTo={0.97}
          style={styles.createCustom}>
          <View style={styles.createIcon}><Glyph color={palette.lime} name="plus" size={16} /></View>
          <View style={styles.rowCopy}>
            <Text style={styles.createTitle}>Create custom exercise</Text>
            <Text style={styles.createDetail}>Choose KG/reps, reps-only, or a timed movement</Text>
          </View>
          <Glyph color={palette.inkLow} name="chevron" size={15} />
        </Tap>
      </Reveal>

      <Reveal index={3} style={styles.fill}>
        <FlatList
          contentContainerStyle={styles.list}
          data={results}
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          onTouchMove={Keyboard.dismiss}
          keyExtractor={(item) => item.id}
          ListEmptyComponent={(
            <EmptyState
              action={<PrimaryButton icon="restart" label="Clear filters" onPress={clearFilters} />}
              body="Nothing matches that combination yet. Widen the muscle or equipment filter and try again."
              icon="search"
              title="No exercises match"
            />
          )}
          renderItem={({ item }) => {
            const isSelected = selected.includes(item.id);
            const photos = photosFor(item.id);
            return (
              <Tap
                accessibilityLabel={`${isSelected ? 'Deselect' : 'Select'} ${item.name}`}
                onPress={() => toggle(item)}
                scaleTo={0.985}>
                <Card padded={false} style={[styles.row, isSelected && styles.rowSelected]}>
                  <View style={[styles.thumb, photos ? styles.thumbPhotoBg : null]}>
                    {photos
                      ? <Image resizeMode="cover" source={photos[0]} style={styles.thumbPhoto} />
                      : <ExerciseFigure accent={palette.lime} gear={item.gear} paused size={52} template={item.template} tint={palette.inkMid} />}
                  </View>
                  <View style={styles.rowCopy}>
                    <Text numberOfLines={1} style={styles.rowName}>{item.name}</Text>
                    <Text numberOfLines={1} style={styles.rowMeta}>
                      {item.primaryMuscle}{item.secondaryMuscles.length ? ` · +${item.secondaryMuscles.join(', ')}` : ''}
                    </Text>
                    <Text numberOfLines={1} style={styles.rowEquip}>
                      {item.equipment} · {item.kind === 'duration' ? 'timed' : item.kind === 'reps-only' ? 'bodyweight reps' : 'weight × reps'}
                    </Text>
                  </View>
                  <Tap
                    accessibilityLabel={item.custom ? `Edit ${item.name}` : `How to do ${item.name}`}
                    hitSlop={8}
                    onPress={() => router.push((item.custom
                      ? { pathname: '/custom-exercise', params: { id: item.id } }
                      : { pathname: '/exercise/[id]', params: { id: item.id } }) as Href)}
                    scaleTo={0.9}
                    style={styles.info}>
                    <Text style={styles.infoText}>{item.custom ? 'Edit' : 'How to'}</Text>
                  </Tap>
                  <View style={[styles.check, isSelected && styles.checkOn]}>
                    {isSelected ? <Glyph color={palette.onLime} name="check" size={14} strokeWidth={2.6} /> : null}
                  </View>
                </Card>
              </Tap>
            );
          }}
        />
      </Reveal>

      <GlassFooter>
        <PrimaryButton
          disabled={!selected.length}
          icon="plus"
          label={selected.length
            ? `Add ${selected.length} exercise${selected.length > 1 ? 's' : ''}`
            : 'Select exercises'}
          onPress={confirm}
        />
      </GlassFooter>
    </Screen>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },

  searchRow: { paddingHorizontal: space.md, paddingTop: 10 },
  searchBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.lineHi,
    borderRadius: radius.md,
    paddingHorizontal: 14,
  },
  searchInput: { flex: 1, height: 50, ...text.row, fontSize: 14.5, color: palette.ink },
  searchClear: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: palette.surfaceLo,
    borderWidth: 1,
    borderColor: palette.line,
    alignItems: 'center',
    justifyContent: 'center',
  },

  filters: { gap: 7, paddingVertical: 11 },
  chipRow: { gap: 7, paddingHorizontal: space.md },
  createCustom: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginHorizontal: space.md,
    marginBottom: 10,
    padding: 12,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: `${palette.lime}55`,
    backgroundColor: palette.limeSoft,
  },
  createIcon: { width: 34, height: 34, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.surface },
  createTitle: { ...text.row, color: palette.ink },
  createDetail: { ...text.micro, color: palette.inkMid, marginTop: 2 },

  list: { paddingHorizontal: space.md, paddingBottom: space.md, gap: 8 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 11, padding: 10 },
  rowSelected: { borderColor: `${palette.lime}59`, backgroundColor: palette.limeSoft },
  rowCopy: { flex: 1 },
  thumb: {
    width: 58,
    height: 58,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceLo,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: palette.line,
  },
  thumbPhotoBg: { backgroundColor: palette.white },
  thumbPhoto: { width: '100%', height: '100%' },
  rowName: { ...text.row, color: palette.ink },
  rowMeta: { ...text.caption, fontSize: 10.5, color: palette.inkMid, marginTop: 3, textTransform: 'capitalize' },
  rowEquip: { ...text.micro, fontSize: 10, color: palette.lime, marginTop: 3, textTransform: 'capitalize' },
  info: {
    height: 30,
    paddingHorizontal: 10,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceLo,
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoText: { ...text.micro, fontFamily: text.value.fontFamily, fontSize: 10.5, color: palette.inkMid, ...tabular },
  check: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: palette.lineHi,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkOn: { backgroundColor: palette.lime, borderColor: palette.lime },
});
