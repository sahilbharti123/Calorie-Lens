import { useRouter } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import {
  Bar,
  Card,
  CountUp,
  EmptyState,
  ListRow,
  Pill,
  PrimaryButton,
  Reveal,
  Screen,
  ScreenHeader,
  SectionTitle,
  Tap,
  VoiceBar,
  Well,
} from '@/src/components/ui';
import { dayTotals, slotLabels } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { macroColor, palette, radius, space, tabular, text } from '@/src/theme';
import type { MealItem, MealSlot } from '@/src/types';

const SLOTS: { slot: MealSlot; icon: GlyphName }[] = [
  { slot: 'breakfast', icon: 'sun' },
  { slot: 'lunch', icon: 'bowl' },
  { slot: 'snack', icon: 'spark' },
  { slot: 'dinner', icon: 'sleep' },
];

/** Height of the uncertainty band. Tall enough to read as a range, not a rule. */
const BAND_HEIGHT = 12;

export default function FoodScreen() {
  const router = useRouter();
  const { data, today, removeMeal } = useApp();

  const totals = dayTotals(today);
  const logged = today.meals.length;

  // Food's question is not "how much is left" — Today answers that. It is "how
  // much do we actually know". Everything below is the day's accumulated
  // low–high band and the entry doing the most damage to it.
  const band = mealBand(today.meals);
  const spread = band.high - band.low;
  const scale = Math.max(1, data.goals.calories);
  const midRatio = totals.calories / scale;
  const widest = widestGuess(today.meals);

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          action={
            <Tap
              accessibilityLabel="Log food"
              onPress={() => router.push('/quick-log')}
              scaleTo={0.92}>
              <View style={styles.headerAction}>
                <Glyph color={palette.lime} name="plus" size={19} />
              </View>
            </Tap>
          }
          eyebrow="Nutrition"
          title="Food"
        />

        {/* ---------- Hero: how trustworthy is today's number ---------- */}
        <Reveal index={1}>
          <Card glow raised>
            <View style={styles.heroHead}>
              <Text style={styles.heroLabel}>EATEN TODAY</Text>
              {logged ? (
                <Pill
                  icon={spread > 0 ? 'target' : 'check'}
                  label={spread > 0 ? `${spread.toLocaleString()} kcal spread` : 'No spread'}
                  tone={spread > 0 ? 'default' : 'accent'}
                />
              ) : null}
            </View>

            <View style={styles.heroNumberRow}>
              <CountUp
                prefix={spread > 0 ? '~' : ''}
                style={styles.heroNumber}
                value={Math.round(totals.calories)}
              />
              <Text style={styles.heroUnit}>{spread > 0 ? 'kcal midpoint' : 'kcal'}</Text>
            </View>

            {/* The band, drawn against the day's target so the track means
                something: solid lime is what the day is at minimum, the ghost
                fill is how far it could reach, the tick is the midpoint above. */}
            <View
              accessible
              accessibilityLabel={
                spread > 0
                  ? `Between ${band.low.toLocaleString()} and ${band.high.toLocaleString()} kcal, midpoint ${Math.round(totals.calories).toLocaleString()}`
                  : `${band.low.toLocaleString()} kcal logged, no range`
              }
              style={styles.band}>
              <Bar
                color={`${palette.lime}2E`}
                delay={160}
                height={BAND_HEIGHT}
                overColor={`${palette.danger}3D`}
                style={styles.bandLayer}
                track={palette.surfaceLo}
                value={band.high / scale}
              />
              <Bar
                color={palette.lime}
                delay={220}
                height={BAND_HEIGHT}
                style={styles.bandLayer}
                // A fully transparent stop derived from the canvas token, so the
                // solid fill layers over the ghost band instead of masking it.
                track={`${palette.bg}00`}
                value={band.low / scale}
              />
              {spread > 0 ? (
                <View
                  style={[styles.bandMid, { left: `${Math.min(100, Math.max(0, midRatio * 100))}%` }]}
                />
              ) : null}
            </View>

            <View style={styles.bandFoot}>
              <Text style={styles.bandRange}>
                {spread > 0
                  ? `${band.low.toLocaleString()}–${band.high.toLocaleString()} possible`
                  : `${band.low.toLocaleString()} logged`}
              </Text>
              <Text style={styles.bandTarget}>
                of {data.goals.calories.toLocaleString()} kcal target
              </Text>
            </View>

            {/* The one entry costing the most certainty, one tap from a fix. */}
            {widest ? (
              <Well style={styles.tighten}>
                <ListRow
                  accent={palette.warn}
                  detail={`Widest guess · ${widest.meal.quantity}`}
                  icon="alert"
                  last
                  onPress={() =>
                    router.push({ pathname: '/quick-log', params: { slot: widest.meal.slot } })
                  }
                  title={widest.meal.name}
                  unit="kcal"
                  value={`${widest.low.toLocaleString()}–${widest.high.toLocaleString()}`}
                />
              </Well>
            ) : null}
          </Card>
        </Reveal>

        {/* ---------- Log ---------- */}
        <Reveal index={2} style={styles.voice}>
          <VoiceBar label="Say “2 rotis and one bowl dal”" />
        </Reveal>

        {/* ---------- The day, meal by meal ---------- */}
        <SectionTitle
          aside={logged ? `${logged} item${logged === 1 ? '' : 's'}` : undefined}
          title="Meals"
        />

        {!logged ? (
          <Reveal index={3} style={styles.emptyWrap}>
            <EmptyState
              action={
                <PrimaryButton
                  icon="mic"
                  label="Log your first meal"
                  onPress={() => router.push('/quick-log')}
                />
              }
              body="Type or say what you ate. Calorie Lens estimates calories and macros, then lets you review before saving."
              icon="bowl"
              title="Your day starts with one sentence"
            />
          </Reveal>
        ) : null}

        {SLOTS.map(({ slot, icon }, index) => {
          const meals = today.meals.filter((meal) => meal.slot === slot);
          const filled = meals.length > 0;
          return (
            <Reveal index={4 + index} key={slot}>
              <Card padded={false} style={styles.slotCard}>
                <Tap
                  accessibilityLabel={`Add to ${slotLabels[slot]}`}
                  onPress={() => router.push({ pathname: '/quick-log', params: { slot } })}
                  scaleTo={0.985}>
                  <View style={styles.slotHeader}>
                    <View style={[styles.slotIcon, filled && styles.slotIconOn]}>
                      <Glyph color={filled ? palette.lime : palette.inkLow} name={icon} size={16} />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.slotTitle}>{slotLabels[slot]}</Text>
                      <Text style={[styles.slotMeta, filled && styles.slotMetaOn]}>
                        {filled ? mealRange(meals) : 'Nothing logged'}
                      </Text>
                    </View>
                    <View style={styles.addButton}>
                      <Glyph color={palette.lime} name="plus" size={17} />
                    </View>
                  </View>
                </Tap>

                {meals.map((meal) => (
                  <MealRow key={meal.id} meal={meal} onRemove={() => removeMeal(meal.id)} />
                ))}
              </Card>
            </Reveal>
          );
        })}
      </ScrollView>
    </Screen>
  );
}

/** One logged item: calories loud, macros quiet, delete always reachable. */
function MealRow({ meal, onRemove }: { meal: MealItem; onRemove: () => void }) {
  const energy =
    meal.calorieLow != null && meal.calorieHigh != null
      ? `${meal.calorieLow}–${meal.calorieHigh}`
      : `${meal.calories}`;

  return (
    <View style={styles.mealRow}>
      <View style={styles.mealBody}>
        <Text numberOfLines={1} style={styles.mealName}>
          {meal.name}
        </Text>
        <Text numberOfLines={1} style={styles.mealQuantity}>
          {meal.quantity}
        </Text>
        <View
          accessible
          accessibilityLabel={`${meal.protein} g protein, ${meal.carbs} g carbs, ${meal.fat} g fat`}
          style={styles.macroStrip}>
          <MacroTag color={macroColor.protein} symbol="P" value={meal.protein} />
          <MacroTag color={macroColor.carbs} symbol="C" value={meal.carbs} />
          <MacroTag color={macroColor.fat} symbol="F" value={meal.fat} />
        </View>
      </View>

      <View style={styles.mealEnergy}>
        <Text style={styles.mealKcal}>{energy}</Text>
        <Text style={styles.mealKcalUnit}>KCAL</Text>
      </View>

      <Tap
        accessibilityLabel={`Delete ${meal.name}`}
        haptic="medium"
        hitSlop={10}
        onPress={onRemove}
        scaleTo={0.88}
        style={styles.trash}>
        <Glyph color={palette.inkLow} name="trash" size={16} />
      </Tap>
    </View>
  );
}

function MacroTag({ symbol, value, color }: { symbol: string; value: number; color: string }) {
  return (
    <View style={styles.macroTag}>
      <Text style={[styles.macroTagKey, { color }]}>{symbol}</Text>
      <Text style={styles.macroTagValue}>{value} g</Text>
    </View>
  );
}

/** The accumulated low–high band across a set of entries. */
function mealBand(meals: { calories: number; calorieLow?: number; calorieHigh?: number }[]) {
  const low = Math.round(meals.reduce((sum, meal) => sum + (meal.calorieLow ?? meal.calories), 0));
  const high = Math.round(meals.reduce((sum, meal) => sum + (meal.calorieHigh ?? meal.calories), 0));
  return { low, high };
}

function mealRange(meals: { calories: number; calorieLow?: number; calorieHigh?: number }[]) {
  const { low, high } = mealBand(meals);
  // Every entry was a gram weight or a package serving, so there is no spread to
  // show — printing "452–452" would imply an uncertainty the data does not have.
  if (low === high) return `${low} kcal`;
  return `${low}–${high} kcal`;
}

/** The single entry whose low–high spread costs the day the most certainty. */
function widestGuess(meals: MealItem[]) {
  return meals.reduce<{ meal: MealItem; low: number; high: number } | null>((worst, meal) => {
    if (meal.calorieLow == null || meal.calorieHigh == null) return worst;
    const low = Math.round(meal.calorieLow);
    const high = Math.round(meal.calorieHigh);
    if (high <= low) return worst;
    if (!worst || high - low > worst.high - worst.low) return { meal, low, high };
    return worst;
  }, null);
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  headerAction: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: palette.surfaceHi,
    borderWidth: 1,
    borderColor: palette.lineHi,
    alignItems: 'center',
    justifyContent: 'center',
  },

  heroHead: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: space.sm,
  },
  heroLabel: { ...text.label, color: palette.lime },
  heroNumberRow: { flexDirection: 'row', alignItems: 'flex-end', marginTop: space.sm },
  heroNumber: { ...text.hero, color: palette.ink, ...tabular },
  heroUnit: { ...text.caption, color: palette.inkMid, marginLeft: space.xs, marginBottom: space.xs },

  band: { height: BAND_HEIGHT, marginTop: space.md },
  bandLayer: { position: 'absolute', left: 0, right: 0, top: 0 },
  bandMid: {
    position: 'absolute',
    top: -3,
    // Half the tick's width, so the mark sits *on* the midpoint rather than
    // beside it. An optical correction, not spacing — no `space` token applies.
    marginLeft: -1,
    width: 2,
    height: BAND_HEIGHT + 6,
    borderRadius: 1,
    backgroundColor: palette.ink,
  },
  bandFoot: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: space.sm,
    marginTop: space.xs,
  },
  bandRange: { ...text.value, color: palette.ink, ...tabular },
  bandTarget: { ...text.caption, color: palette.inkLow, ...tabular },

  tighten: { marginTop: space.md, paddingVertical: 0 },

  voice: { marginTop: space.md },
  emptyWrap: { marginBottom: space.sm },

  slotCard: { paddingHorizontal: 14, marginBottom: space.sm },
  slotHeader: { minHeight: 66, flexDirection: 'row', alignItems: 'center', gap: space.sm },
  slotIcon: {
    width: 34,
    height: 34,
    borderRadius: 12,
    backgroundColor: `${palette.inkLow}14`,
    alignItems: 'center',
    justifyContent: 'center',
  },
  slotIconOn: { backgroundColor: palette.limeSoft },
  slotTitle: { ...text.row, color: palette.ink },
  slotMeta: { ...text.caption, color: palette.inkLow, marginTop: space.xs },
  slotMetaOn: { ...text.value, fontSize: 12, color: palette.lime, ...tabular },
  addButton: {
    width: 34,
    height: 34,
    borderRadius: radius.pill,
    backgroundColor: palette.limeSoft,
    borderWidth: 1,
    borderColor: `${palette.lime}2E`,
    alignItems: 'center',
    justifyContent: 'center',
  },

  mealRow: {
    minHeight: 66,
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    paddingVertical: 11,
    borderTopWidth: 1,
    borderTopColor: palette.line,
  },
  mealBody: { flex: 1 },
  mealName: { ...text.row, color: palette.ink },
  mealQuantity: { ...text.caption, fontSize: 11, color: palette.inkMid, marginTop: space.xs },
  macroStrip: { flexDirection: 'row', alignItems: 'center', gap: space.sm, marginTop: space.xs },
  macroTag: { flexDirection: 'row', alignItems: 'center', gap: space.xs },
  macroTagKey: { ...text.label, fontSize: 8.5, letterSpacing: 0.8 },
  macroTagValue: { ...text.micro, fontSize: 10, color: palette.inkLow, ...tabular },
  mealEnergy: { alignItems: 'flex-end' },
  mealKcal: { ...text.headline, fontSize: 17, color: palette.ink, ...tabular },
  mealKcalUnit: { ...text.label, fontSize: 7.5, letterSpacing: 1.1, color: palette.inkLow, marginTop: space.xs },
  trash: {
    width: 34,
    height: 34,
    borderRadius: 12,
    backgroundColor: palette.surfaceLo,
    borderWidth: 1,
    borderColor: palette.line,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
