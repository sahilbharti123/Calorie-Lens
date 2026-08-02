import { apiRequest, apiUrl } from '@/src/lib/api-client';
import { FOODS, type FoodReference } from '@/src/lib/food-catalog';
import { readSession } from '@/src/lib/session';
import { parseWeightInput } from '@/src/lib/weight';
import type {
  ClarificationAnswer,
  LearnedFood,
  ClarificationTarget,
  EstimationContext,
  LogOperation,
  MealItem,
  MealSlot,
  ParsedCommand,
  PendingClarification,
  Workout,
} from '@/src/types';

/** Remote language parsing is opt-in and off by default — the app ships with zero running cost. */
const AI_PARSING_ENABLED = process.env.EXPO_PUBLIC_ENABLE_AI_PARSING === '1';

const foods = FOODS;

const numberWords: Record<string, number> = {
  a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8,
  nine: 9, ten: 10, half: 0.5, quarter: 0.25,
  // Vague counts people genuinely use. A couple is two; a few is three-ish.
  couple: 2, few: 3, several: 3, dozen: 12,
  // Hindi numbers, since the app is used in Hinglish.
  ek: 1, do: 2, teen: 3, char: 4, paanch: 5, aadha: 0.5, adha: 0.5,
};

const GLASS_ML = 250;

type Quantity = { amount: number; unit: string };

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function numberValue(value?: string) {
  if (!value) return 0;
  // "a couple of", "half a" — strip the filler and read the count word.
  const word = value.replace(/^an?\s+/, '').replace(/\s+(?:of|a)$/, '').trim();
  return numberWords[word] ?? numberWords[value] ?? Number.parseFloat(value);
}

/**
 * Every spelling of a unit, mapped explicitly.
 *
 * This used to strip a trailing "s" and patch up the damage afterwards, which
 * quietly turned "glass" into "glas" — a unit nothing matched, so "a glass of
 * beer" asked how much beer you had. A table cannot go wrong that way.
 */
const UNIT_ALIASES: Record<string, string> = {
  kg: 'kg', kgs: 'kg', kilo: 'kg', kilos: 'kg', kilogram: 'kg', kilograms: 'kg',
  g: 'g', gm: 'g', gms: 'g', gram: 'g', grams: 'g',
  ml: 'ml', millilitre: 'ml', millilitres: 'ml', milliliter: 'ml', milliliters: 'ml',
  l: 'l', litre: 'l', litres: 'l', liter: 'l', liters: 'l',
  bowl: 'bowl', bowls: 'bowl', katori: 'bowl', katoris: 'bowl',
  cup: 'cup', cups: 'cup', mug: 'cup', mugs: 'cup',
  glass: 'glass', glasses: 'glass',
  piece: 'piece', pieces: 'piece', slice: 'piece', slices: 'piece',
  roti: 'piece', rotis: 'piece',
  bottle: 'serving', bottles: 'serving', can: 'serving', cans: 'serving',
  pint: 'pint', pints: 'pint', peg: 'peg', pegs: 'peg', shot: 'peg', shots: 'peg',
  plate: 'serving', plates: 'serving', serving: 'serving', servings: 'serving',
  handful: 'handful', handfuls: 'handful', packet: 'serving', packets: 'serving',
  tbsp: 'tbsp', tsp: 'tsp',
};

/** A handful is about 30 g of whatever it is — nuts, crisps, popcorn. */
const HANDFUL_G = 30;

/** A pint and a peg are fixed measures; everything else is per-food. */
const FIXED_ML: Record<string, number> = { pint: 568, peg: 30 };

function normalizeUnit(raw: string | undefined, reference: FoodReference) {
  if (raw) return UNIT_ALIASES[raw] ?? raw;
  // No unit said at all: "two beers", "a banana". Count it as one of whatever
  // one of this food is.
  if (reference.servingMl) return 'serving';
  if (reference.pieceG) return 'piece';
  return 'unknown';
}

const UNITS = 'kgs?|kilos?|kilograms?|g|gms?|grams?|ml|millilit(?:re|er)s?|l|lit(?:re|er)s?|bowls?|katoris?|cups?|mugs?|glass(?:es)?|pieces?|slices?|rotis?|bottles?|cans?|pints?|pegs?|shots?|plates?|servings?|handfuls?|packets?|tbsp|tsp';
const NUMBER = '\\d+(?:\\.\\d+)?|a couple of|a couple|a few|half a|an|a|one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|couple|few|several|dozen|ek|do|teen|char|paanch|aadha|adha';

/**
 * Finds the amount that belongs to one food.
 *
 * `loose` widens the search to allow a few words between the amount and the
 * food it describes, which is how people actually speak — "500 ml hoegaarden
 * flavoured beer" puts two words in the way. It is only safe when the sentence
 * mentions a single food, and only when a unit was stated: with two foods "2
 * rotis and rajma" would otherwise give the rajma a quantity of 2, and without
 * a unit "a plate of momos" would read as one momo.
 */
function quantityNear(
  text: string,
  alias: string,
  reference: FoodReference,
  loose = false,
): Quantity {
  const food = escapeRegex(alias);
  const explicitVolume = text.match(
    new RegExp(`(\\d+(?:\\.\\d+)?)\\s*ml\\s*(?:bowl|katori|glass)?(?:\\s+of)?\\s*${food}(?:es|s)?\\b`),
  );
  if (explicitVolume) return { amount: numberValue(explicitVolume[1]), unit: 'ml' };
  // The number is word-anchored on both sides. Without a leading \b, the bare
  // "a" alternative matched the final letter of "thod-a", so "thoda paneer"
  // read as one unnameable unit of paneer and the parser asked how much.
  const before = text.match(new RegExp(`\\b(${NUMBER})\\b\\s*(${UNITS})?\\s*(?:of\\s+)?${food}(?:es|s)?\\b`));
  const after = text.match(new RegExp(`${food}s?\\s*[:,-]?\\s*\\b(${NUMBER})\\b\\s*(${UNITS})\\b`));
  const match = before ?? after;
  if (match) return { amount: numberValue(match[1]), unit: normalizeUnit(match[2], reference) };

  if (loose) {
    const spaced = text.match(
      new RegExp(`\\b(${NUMBER})\\b\\s*(${UNITS})\\b(?:\\s+[\\w'-]+){0,4}?\\s+${food}(?:es|s)?\\b`),
    );
    if (spaced) return { amount: numberValue(spaced[1]), unit: normalizeUnit(spaced[2], reference) };
  }
  return { amount: 0, unit: 'unknown' };
}

/**
 * How much water was drunk.
 *
 * Written against the shared NUMBER pattern rather than a bare `\\d+`, because
 * people say "a litre" and "half a litre" as often as "500 ml" — and because
 * the old litre pattern required the word to end there, so "2 litres" matched
 * nothing and silently fell through to a single 250 ml glass.
 */
function waterAmount(text: string): number {
  const read = (units: string) => {
    const match = text.match(new RegExp(`\\b(${NUMBER})\\b\\s*(?:${units})\\b`));
    return match ? numberValue(match[1]) : 0;
  };
  const millilitres = read('ml|millilit(?:re|er)s?');
  if (millilitres) return millilitres;
  const litres = read('l|lit(?:re|er)s?');
  if (litres) return litres * 1000;
  const glasses = read('glass(?:es)?');
  if (glasses) return glasses * GLASS_ML;
  const bottles = read('bottles?');
  if (bottles) return bottles * 500;
  return GLASS_ML;
}

function firstNumber(text: string, pattern: RegExp) {
  const match = text.match(pattern);
  if (!match) return 0;
  const raw = match.slice(1).find(Boolean)?.replaceAll(',', '');
  return raw ? Number.parseFloat(raw) : 0;
}

function clarification(
  transcript: string,
  question: string,
  suggestions: string[],
  target: ClarificationTarget,
): ParsedCommand {
  return {
    transcript,
    confirmation: 'One detail will improve this estimate',
    operations: [],
    source: 'local',
    clarification: { question, suggestions, target },
  };
}

/* ------------------------------------------------------------------ *
 * Clarification answers
 *
 * A question is asked about one specific missing detail, and the answer is
 * applied to that detail. It is never appended to the transcript: an answer
 * like "1 bowl" says nothing about *which* food it belongs to, and quantities
 * are only ever read next to the food they describe, so re-parsing a grown
 * transcript re-asks the same question forever.
 * ------------------------------------------------------------------ */

/** Everything the user has already told us, keyed by what it answers. */
type Resolved = {
  amounts: Map<string, Quantity>;
  pieceGrams: Map<string, number>;
  minutes?: number;
  intensity?: Workout['intensity'];
  bowlMl?: number;
  weightKg?: number;
};

function resolveAnswers(answers: ClarificationAnswer[] = []): Resolved {
  const resolved: Resolved = { amounts: new Map(), pieceGrams: new Map() };
  for (const answer of answers) {
    if (answer.kind === 'foodAmount') {
      resolved.amounts.set(answer.alias, { amount: answer.amount, unit: answer.unit });
    } else if (answer.kind === 'pieceGrams') {
      resolved.pieceGrams.set(answer.alias, answer.grams);
    } else if (answer.kind === 'bowlMl') {
      resolved.bowlMl = answer.ml;
    } else if (answer.kind === 'bodyWeight') {
      resolved.weightKg = answer.kg;
    } else if (answer.kind === 'workoutMinutes') {
      resolved.minutes = answer.minutes;
    } else {
      resolved.intensity = answer.intensity;
    }
  }
  return resolved;
}

/** True when an answer already given covers the detail being asked about. */
function answers(answer: ClarificationAnswer, target: ClarificationTarget) {
  if (answer.kind !== target.kind) return false;
  if ('alias' in answer && 'alias' in target) return answer.alias === target.alias;
  return true;
}

// "Brisk" is the standard word for a moderate-paced walk, not a hard one, and
// reading it as hard overstated a half-hour walk by a third.
const INTENSITY_WORDS: [RegExp, Workout['intensity']][] = [
  [/hard|intense|vigorous|fast|heavy/, 'hard'],
  [/light|easy|gentle|slow|casual/, 'light'],
  [/moderate|medium|normal|steady|brisk/, 'moderate'],
];

/**
 * Turns a spoken or typed reply into a value for the detail that was asked
 * about, or null when the reply does not answer that question at all.
 *
 * Returning null matters as much as returning a value: it lets the screen say
 * "that isn't a number" once, instead of silently re-asking and looking stuck.
 */
export function interpretClarificationAnswer(
  target: ClarificationTarget,
  reply: string,
): ClarificationAnswer | null {
  const lowered = reply.toLowerCase().trim();
  if (!lowered) return null;

  if (target.kind === 'workoutIntensity') {
    for (const [pattern, intensity] of INTENSITY_WORDS) {
      if (pattern.test(lowered)) return { kind: 'workoutIntensity', intensity };
    }
    return null;
  }

  if (target.kind === 'bodyWeight') {
    const parsed = parseWeightInput(lowered);
    return parsed ? { kind: 'bodyWeight', kg: parsed.kg } : null;
  }

  // Everything below needs a number.
  const numberMatch = lowered.match(/(\d+(?:\.\d+)?)|\b(a|an|one|two|three|four|five|six|half|quarter)\b/);
  if (!numberMatch) return null;
  const amount = numberValue(numberMatch[1] ?? numberMatch[2]);
  if (!Number.isFinite(amount) || amount <= 0) return null;

  if (target.kind === 'workoutMinutes') {
    // An hour is a normal way to answer "how many minutes".
    const hours = /\bhours?\b|\bhrs?\b/.test(lowered);
    const minutes = hours ? amount * 60 : amount;
    if (minutes < 1 || minutes > 600) return null;
    return { kind: 'workoutMinutes', minutes };
  }

  if (target.kind === 'bowlMl') {
    const ml = /\bl\b|\blitres?\b|\bliters?\b/.test(lowered) ? amount * 1000 : amount;
    if (ml < 50 || ml > 2000) return null;
    return { kind: 'bowlMl', ml };
  }

  if (target.kind === 'pieceGrams') {
    if (amount < 1 || amount > 2000) return null;
    return { kind: 'pieceGrams', alias: target.alias, grams: amount };
  }

  if (target.kind === 'foodAmount') {
    const food = foodFor(target.alias);
    const unitMatch = lowered.match(
      /\b(kgs?|kilos?|kilograms?|grams?|gms?|g|ml|millilitres?|litres?|liters?|l|bowls?|katoris?|cups?|glass(?:es)?|pieces?|slices?|rotis?|tbsp|tsp)\b/,
    );
    if (unitMatch) {
      const unit = normalizeAnswerUnit(unitMatch[1]);
      // Reject a unit this food cannot be converted from, rather than accept it
      // and ask the same question again. Idli has a piece weight and no
      // density, so "1 bowl" is unanswerable however many times we ask.
      if (VOLUME_UNITS.has(unit) && food && !food.density) return null;
      return { kind: 'foodAmount', alias: target.alias, amount, unit };
    }
    // A bare number: "2" almost always means two of the thing, "150" means
    // grams. Nobody eats 150 rotis, and nobody weighs out 2 grams of rice.
    const bare = amount >= 20 ? 'g' : food?.pieceG ? 'piece' : food?.density ? 'bowl' : 'g';
    return { kind: 'foodAmount', alias: target.alias, amount, unit: bare };
  }

  // 'unknownFood' and 'intent' are not single values — the caller re-parses
  // the reply as ordinary text instead.
  return null;
}

function normalizeAnswerUnit(raw: string) {
  const unit = raw.replace(/s$/, '');
  if (/^(kilo|kilogram|kg)$/.test(unit)) return 'kg';
  if (/^(gram|gm|g)$/.test(unit)) return 'g';
  if (/^(millilitre|ml)$/.test(unit)) return 'ml';
  if (/^(litre|liter|l)$/.test(unit)) return 'l';
  if (/^(katori|bowl)$/.test(unit)) return 'bowl';
  if (/^glasse?$/.test(unit)) return 'glass';
  if (/^(slice|roti|piece)$/.test(unit)) return 'piece';
  return unit;
}

const UNKNOWN_FOOD_MARK = 'verified reference';

/** Marks an entry whose number the user typed, and which can be remembered. */
const LABEL_SOURCE = 'Food label supplied by user';

/**
 * Words that mean "this is food". Word-anchored on purpose: an unanchored
 * `ate` matches "w-ate-r" and "moder-ate", which sent every glass of water and
 * every moderate workout down the unknown-food path and lost the entry.
 */
const MEAL_WORDS = /\b(?:ate|eat|eating|had|having|breakfast|lunch|dinner|snack|khaya|khayi|khana)\b/;

const WATER_WORDS = /\b(?:water|paani|pani)\b/;

/* ------------------------------------------------------------------ *
 * Foods the user taught us
 * ------------------------------------------------------------------ */

/** Words too generic to identify a food on their own. */
const NAME_STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'of', 'with', 'some', 'my', 'had', 'ate', 'drank',
  'was', 'is', 'for', 'from', 'label', 'kcal', 'calories', 'calorie', 'today',
  'one', 'two', 'three', 'flavoured', 'flavored', 'plate', 'bowl', 'glass',
]);

/**
 * Turns what the user said into a name and the phrases that should match it
 * next time. Amounts, calorie figures and filler are stripped: the amount is
 * this serving, not part of the food's identity.
 */
/**
 * The phrases that should match a food with this name. Kept separate from
 * `learnableName` so a rename regenerates matching by the same rule that
 * created it — otherwise a corrected name would still be found only under the
 * garbled one.
 */
export function aliasesForName(name: string) {
  const words = name.toLowerCase().split(/\s+/).filter(Boolean);
  const cleaned = words.join(' ').trim();
  if (!cleaned) return [];
  return [...new Set([cleaned, ...words.filter((word) => word.length >= 4)])];
}

export function learnableName(transcript: string) {
  const cleaned = transcript
    .toLowerCase()
    .replace(/\d+(?:\.\d+)?\s*(?:kcal|calories?)/g, ' ')
    .replace(/\bfrom (?:the )?label\b/g, ' ')
    .replace(/\d+(?:\.\d+)?\s*(?:kg|g|grams?|ml|l|litres?|liters?|bowls?|cups?|glass(?:es)?|pieces?|slices?)\b/g, ' ')
    .replace(/\d+(?:\.\d+)?/g, ' ')
    .replace(/[^a-z\s'-]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
  const words = cleaned.filter((word) => !NAME_STOPWORDS.has(word));
  // Repeated dictation ("beer ... beer") should not become a repeated name.
  const unique = [...new Set(words)];
  const name = unique.join(' ').trim();
  if (!name) return null;
  return { name, aliases: aliasesForName(name) };
}

/** The label-calories entry that was just logged, ready to be remembered. */
export function learnableFrom(
  transcript: string,
  item: Pick<MealItem, 'calories' | 'protein' | 'carbs' | 'fat' | 'sourceLabel'>,
): Omit<LearnedFood, 'id' | 'createdAt' | 'updatedAt'> | null {
  // Only a figure the user supplied. An entry that already came from a saved
  // food must not be re-learned, or the name drifts a little every time.
  if (item.sourceLabel !== LABEL_SOURCE) return null;
  const named = learnableName(transcript);
  if (!named) return null;
  const lowered = transcript.toLowerCase();
  const serving = lowered.match(new RegExp(`(${NUMBER})\\s*(${UNITS})\\b`));
  return {
    name: named.name,
    aliases: named.aliases,
    calories: item.calories,
    protein: item.protein,
    carbs: item.carbs,
    fat: item.fat,
    servingAmount: serving ? numberValue(serving[1]) : undefined,
    servingUnit: serving ? normalizeUnit(serving[2], {} as FoodReference) : undefined,
  };
}

/** The learned food this text refers to, longest match first. */
function learnedMatch(text: string, learned: LearnedFood[] = []) {
  let best: { food: LearnedFood; alias: string } | undefined;
  for (const food of learned) {
    for (const alias of food.aliases) {
      if (!new RegExp(`\\b${escapeRegex(alias)}(?:es|s)?\\b`).test(text)) continue;
      if (!best || alias.length > best.alias.length) best = { food, alias };
    }
  }
  return best;
}

/**
 * Scales a taught figure to the amount just said. With one verified data point
 * linear scaling is the only defensible move, and the range widens because the
 * further you get from the amount that was actually checked, the less the
 * single figure tells you.
 */
function estimateLearned(
  food: LearnedFood,
  text: string,
  alias: string,
): Omit<MealItem, 'id' | 'slot' | 'loggedAt'> {
  const said = quantityNear(text, alias, { pieceG: 1 } as FoodReference, true);
  const sameUnit = Boolean(
    said.amount && food.servingAmount && food.servingUnit && said.unit === food.servingUnit,
  );
  const factor = sameUnit ? said.amount / food.servingAmount! : 1;
  const scaled = (value: number) => Math.round(value * factor * 10) / 10;
  const serving = food.servingAmount && food.servingUnit
    ? `${food.servingAmount} ${food.servingUnit}`
    : 'the amount you saved';
  const label = sameUnit && factor !== 1
    ? `${said.amount} ${said.unit} · scaled from ${serving}`
    : serving;
  return {
    name: food.name,
    quantity: label,
    calories: Math.round(food.calories * factor),
    calorieLow: Math.round(food.calories * factor * 0.95),
    calorieHigh: Math.round(food.calories * factor * 1.05),
    protein: scaled(food.protein),
    carbs: scaled(food.carbs),
    fat: scaled(food.fat),
    source: 'label',
    sourceLabel: 'Your own figure',
    sourceId: `you saved ${food.calories} kcal for ${serving}`,
    confidence: 'medium',
    basis: `${food.calories} kcal for ${serving}, as you recorded it`,
    assumptions: sameUnit && factor !== 1
      ? ['scaled in proportion from the amount you checked']
      : [],
  };
}

export function inferMealSlot(text = ''): MealSlot {
  const lowered = text.toLowerCase();
  if (/breakfast|morning|nashta/.test(lowered)) return 'breakfast';
  if (/lunch|afternoon/.test(lowered)) return 'lunch';
  if (/snack|evening|chai/.test(lowered)) return 'snack';
  if (/dinner|night/.test(lowered)) return 'dinner';
  const hour = new Date().getHours();
  if (hour < 11) return 'breakfast';
  if (hour < 16) return 'lunch';
  if (hour < 19) return 'snack';
  return 'dinner';
}

type Measured = { grams: number; low: number; high: number; label: string; assumed?: boolean };

/**
 * One usual helping of a food, in whatever unit that food is measured in.
 *
 * Used when the sentence names a food but no amount. Everything in the catalog
 * can answer this, so the parser never has to stop and ask for a number it can
 * reasonably assume — the assumption is stated on the entry instead.
 */
function defaultPortion(food: FoodReference, context: EstimationContext): Measured | null {
  const spread = 0.35;
  const measure = (grams: number, label: string): Measured => ({
    grams,
    low: grams * (1 - spread),
    high: grams * (1 + spread),
    label,
  });
  // An explicit helping weight outranks the piece weight, because some foods
  // are counted but never eaten one at a time: nobody has a single chicken
  // nugget, and reading "nuggets and fries" as one 17 g piece logged 49 kcal.
  if (food.servingG) return measure(food.servingG, `${food.servingG} g helping (assumed)`);
  // A piece weight only makes a sensible default if a piece is a portion. One
  // almond weighs 1.2 g, so "a handful of almonds" logged 7 kcal.
  if (food.pieceG && food.pieceG >= 10) {
    return measure(food.pieceG, `1 × ${food.pieceG} g standard piece (assumed)`);
  }
  if (food.servingMl && food.density) {
    return measure(food.servingMl * food.density, `1 serving, ${food.servingMl} ml (assumed)`);
  }
  if (food.density) {
    const bowlMl = context.bowlMl ?? 200;
    return measure(bowlMl * food.density, `1 × ${bowlMl} ml bowl (assumed)`);
  }
  // Nothing to scale by — a flat helping, deliberately wide.
  return measure(100, '100 g helping (assumed)');
}
type Missing = { question: string; suggestions: string[]; target: ClarificationTarget };

const VOLUME_UNITS = new Set(['bowl', 'cup', 'glass', 'ml', 'l', 'tbsp', 'tsp']);
/**
 * Units that name a container rather than a measurement. A bowl, a plate and a
 * packet are all "one of these", so for a food with no density they can be read
 * as one helping. "250 ml" cannot.
 */
const VESSEL_UNITS = new Set(['bowl', 'cup', 'glass', 'serving', 'unknown']);

function foodFor(alias: string) {
  return foods.find((food) => food.aliases.includes(alias));
}

/**
 * Words that turn the thing before them into a different food.
 *
 * Tomato sauce is not a tomato, coconut oil is not a coconut, and a mango shake
 * is not a mango — but the alias for each is sitting right there in the phrase.
 * "Pasta with tomato sauce" logged a whole extra tomato this way, and the same
 * shape of mistake put a glass of water inside "coconut water".
 */
const COMPOUND_SUFFIX = /^(?:sauce|ketchup|paste|puree|powder|juice|shake|smoothie|soup|oil|extract|essence|flavou?red?|water|milk|syrup)\b/;

/**
 * Whether the sentence really names this food, rather than merely containing
 * its letters as part of a compound.
 */
function aliasSaid(lowered: string, alias: string) {
  const pattern = new RegExp(`\\b${escapeRegex(alias)}(?:es|s)?\\b`, 'g');
  for (let hit = pattern.exec(lowered); hit; hit = pattern.exec(lowered)) {
    const after = lowered.slice(hit.index + hit[0].length).replace(/^\s+/, '');
    // A suffix already inside the alias is part of the food's own name —
    // "tomato ketchup" is allowed to be followed by nothing in particular, and
    // "coconut water" is only blocked when the alias stops at "coconut".
    if (!COMPOUND_SUFFIX.test(after)) return true;
  }
  return false;
}

/**
 * Asks for an amount in the units this particular food can actually be
 * measured in. Idli has a piece weight and no density, so offering "1 bowl"
 * invites an answer the catalog cannot convert — and a question the user
 * cannot get past.
 */
/** "Dark chocolate (70%)" is a catalog name; "dark chocolate" is a sentence. */
function spoken(name: string) {
  return name.replace(/\s*\([^)]*\)/g, '').trim().toLowerCase();
}

function amountQuestion(food: FoodReference, alias: string): Missing {
  const countable = Boolean(food.pieceG);
  const pourable = Boolean(food.density);
  const suggestions = countable && pourable
    ? ['1 piece', '1 bowl', '100 g', '150 g']
    : countable
      ? ['1 piece', '2 pieces', '3 pieces', '100 g']
      : pourable
        ? ['1 bowl', '1 cup', '100 g', '150 g']
        : ['50 g', '100 g', '150 g', '200 g'];
  const question = countable && !pourable
    ? `How many ${spoken(food.name)} did you have?`
    : `How much ${spoken(food.name)} did you have?`;
  return { question, suggestions, target: { kind: 'foodAmount', alias, foodName: food.name } };
}

function gramsFor(
  quantity: Quantity,
  food: FoodReference,
  context: EstimationContext,
  alias: string,
  resolved?: Resolved,
): Measured | Missing {
  const amount = quantity.amount;
  const unit = quantity.unit;
  if (!amount) {
    // No amount said at all. "2 rotis and dal" means a normal helping of dal —
    // and refusing to log anything because one item lacks a number threw away
    // the whole sentence. Assume one usual serving, say so on the entry, and
    // let the user change it on the review screen before it is saved.
    const assumed = defaultPortion(food, context);
    if (!assumed) return amountQuestion(food, alias);
    return { ...assumed, assumed: true };
  }
  if (unit === 'handful') {
    const grams = amount * HANDFUL_G;
    return { grams, low: grams * 0.6, high: grams * 1.4, label: `${amount} handful, about ${HANDFUL_G} g` };
  }
  if (unit === 'kg') return { grams: amount * 1000, low: amount * 980, high: amount * 1020, label: `${amount} kg` };
  if (unit === 'g') return { grams: amount, low: amount * 0.98, high: amount * 1.02, label: `${amount} g` };
  // A packet or plate of something that states its own helping weight — "a
  // packet of biscuits" is a packet, not one biscuit. Checked before the piece
  // branch below so the helping wins over the piece for these foods.
  if (unit === 'serving' && !food.servingMl && food.servingG) {
    const grams = amount * food.servingG;
    return { grams, low: grams * 0.65, high: grams * 1.35, label: `${amount} × ${food.servingG} g helping`, assumed: true };
  }
  if (unit === 'piece' || (unit === 'serving' && !food.servingMl && food.pieceG)) {
    const pieceG = food.pieceG ?? resolved?.pieceGrams.get(alias);
    if (!pieceG) {
      return {
        question: `About how many grams was each ${food.name.toLowerCase()}?`,
        suggestions: ['30 g', '50 g', '75 g', '100 g'],
        target: { kind: 'pieceGrams', alias, foodName: food.name },
      };
    }
    const grams = amount * pieceG;
    const variance = food.pieceVariance ?? 0.12;
    return { grams, low: grams * (1 - variance), high: grams * (1 + variance), label: `${amount} × ${pieceG} g standard piece` };
  }
  let volumeMl = 0;
  let label = '';
  if (unit === 'bowl') {
    const bowlMl = context.bowlMl ?? resolved?.bowlMl;
    // Only worth asking when there is a density to multiply the answer by.
    // For a food measured any other way the bowl size changes nothing, so the
    // question is one the user cannot get anything out of answering.
    if (!bowlMl && food.density) {
      return {
        question: 'About how large is your usual bowl?',
        suggestions: ['150 ml', '200 ml', '250 ml', '300 ml'],
        target: { kind: 'bowlMl' },
      };
    }
    volumeMl = amount * (bowlMl ?? 0);
    label = `${amount} × ${bowlMl} ml bowl`;
  } else if (unit === 'cup') {
    volumeMl = amount * context.cupMl;
    label = `${amount} × ${context.cupMl} ml cup`;
  } else if (unit === 'glass') {
    // A glass of wine is 150 ml and a glass of whisky is a peg. Where a drink
    // states its own serving and that serving is smaller than a tumbler, the
    // glass it is actually poured into is the smaller one — "a glass of red
    // wine" was logging a 250 ml pour, half a bottle over two.
    const glassMl = Math.min(food.servingMl ?? GLASS_ML, GLASS_ML);
    volumeMl = amount * glassMl;
    label = `${amount} × ${glassMl} ml glass`;
  } else if (unit === 'ml') {
    volumeMl = amount;
    label = `${amount} ml`;
  } else if (unit === 'l') {
    volumeMl = amount * 1000;
    label = `${amount} l`;
  } else if (unit === 'serving' && food.servingMl) {
    volumeMl = amount * food.servingMl;
    label = amount === 1 ? `1 serving (${food.servingMl} ml)` : `${amount} × ${food.servingMl} ml`;
  } else if (FIXED_ML[unit]) {
    volumeMl = amount * FIXED_ML[unit];
    label = `${amount} × ${FIXED_ML[unit]} ml ${unit}`;
  } else if (unit === 'tbsp' || unit === 'tsp') {
    volumeMl = amount * (unit === 'tbsp' ? 15 : 5);
    label = `${amount} ${unit}`;
  }
  // Three things arrive here meaning "this many helpings" rather than a volume:
  // a bare count ("2 paneer"), a stated serving the food has no serving size
  // for ("a packet of crisps"), and a vessel for a food with no published
  // density ("a bowl of bhujia"). Counting helpings is what the speaker meant
  // in every one of those, and it is far better than stopping to ask for an
  // amount they have already given in their own words.
  //
  // An absolute volume is deliberately excluded: "250 ml of paneer" states a
  // real volume the catalog cannot convert, and reading the 250 as a count of
  // helpings would log twenty-five kilograms.
  if (!volumeMl || !food.density) {
    // A vessel of a countable food is the one case still worth a question: a
    // bowl of idli holds several, and quietly logging one would understate it
    // by two thirds. Every other shape here has a helping to fall back on.
    const vesselOfCountable = food.pieceG && !food.servingG && VOLUME_UNITS.has(unit);
    const countsHelpings = !vesselOfCountable && (food.density
      ? unit === 'unknown' || unit === 'serving'
      : VESSEL_UNITS.has(unit));
    const one = countsHelpings ? defaultPortion(food, context) : null;
    if (one) {
      return {
        grams: one.grams * amount,
        low: one.low * amount,
        high: one.high * amount,
        label: amount === 1 ? one.label : `${amount} × ${one.label}`,
        assumed: true,
      };
    }
    // A real volume for a food with no published density, or a unit nothing in
    // the catalog can convert. Here the question is the honest answer.
    return amountQuestion(food, alias);
  }
  const grams = volumeMl * food.density;
  const variance = food.densityVariance ?? 0.12;
  return { grams, low: grams * (1 - variance), high: grams * (1 + variance), label };
}

function estimateFood(
  food: FoodReference,
  quantity: Quantity,
  text: string,
  context: EstimationContext,
  alias: string,
  suppressOilAssumption = false,
  resolved?: Resolved,
): Omit<MealItem, 'id' | 'slot' | 'loggedAt'> | Missing {
  const measured = gramsFor(quantity, food, context, alias, resolved);
  if ('question' in measured) return measured;
  const scale = measured.grams / 100;
  // A typical-composition entry is uncertain in two independent ways: how much
  // was eaten, and what a portion of that food actually contains. Only the
  // first applies to a measured USDA record.
  const spread = food.calorieVariance ?? 0;
  let calories = food.calories * scale;
  let low = food.calories * measured.low / 100 * (1 - spread);
  let high = food.calories * measured.high / 100 * (1 + spread);
  let fat = food.fat * scale;
  const assumptions: string[] = [];
  // Boiled lentils and beans are USDA records for the plain ingredient, so a
  // bowl of dal is that plus the fat it was cooked in. A typical-tier dish is
  // the opposite: "dal makhani" is already a published figure for the finished
  // thing, cream and all, and adding oil on top billed it twice — 100 g of dal
  // tadka came back a third higher than the entry's own per-100 g figure.
  const isCurryBase = food.tier !== 'typical'
    && /rajma|dal|daal|chole|chana|curry/.test(text)
    && /rajma|dal|daal|chole|chana|chickpea|kidney|lentil/i.test(food.name + food.aliases.join(' '));
  if (isCurryBase && !suppressOilAssumption && !/plain|boiled|dry/.test(text)) {
    // Proportional to the helping. A flat allowance meant a kilo of dal and a
    // spoonful were assumed to have been cooked in the same two teaspoons.
    const oilKcal = 4.6 * 8.84 * scale;
    calories += oilKcal;
    low += oilKcal * 0.5;
    high += oilKcal * 2;
    fat += 4.6 * scale;
    assumptions.push('home curry range assumes ½–2 tsp oil per 100 g');
  }
  const exactGrams = quantity.unit === 'g' || quantity.unit === 'kg';
  const typical = food.tier === 'typical';
  if (measured.assumed) {
    assumptions.unshift('amount not stated — assumed one usual serving');
  }
  if (typical) {
    assumptions.push('typical figure for this kind of food, not a measured record');
  }
  const confidence = typical || measured.assumed || measured.label.includes('bowl') || assumptions.length
    ? 'low'
    : exactGrams ? 'high' : 'medium';
  return {
    name: food.name,
    quantity: measured.label,
    calories: Math.round(calories),
    calorieLow: Math.round(low),
    calorieHigh: Math.round(high),
    protein: Math.round(food.protein * scale * 10) / 10,
    carbs: Math.round(food.carbs * scale * 10) / 10,
    fat: Math.round(fat * 10) / 10,
    source: typical ? 'typical' : 'usda',
    sourceLabel: typical ? 'Typical composition' : 'USDA FoodData Central',
    sourceId: typical ? 'category reference · wide range' : `FDC ${food.fdcId}`,
    confidence,
    basis: `${measured.label} · ${food.calories} kcal/100 g${assumptions[0] ? ` · ${assumptions[0]}` : ''}`,
    assumptions,
  };
}

type Activity = { name: string; aliases: RegExp; mets: [number, number, number]; sourceId: string };
/**
 * Activity names are anchored to the start of a word. Without the anchor,
 * `run` matches "c-run-ches" and a set of crunches is logged as a run.
 */
const activities: Activity[] = [
  { name: 'Walking', aliases: /\bwalk/, mets: [2.8, 3.8, 4.8], sourceId: 'walking' },
  { name: 'Running', aliases: /\brun|\bran\b|\bjog|\bsprint/, mets: [6.5, 8.5, 11], sourceId: 'running' },
  { name: 'Cycling', aliases: /\bcycl|\bbike|\bbiking|\bspin class/, mets: [4.3, 7, 9], sourceId: 'bicycling' },
  { name: 'Swimming', aliases: /\bswim|\bswam\b/, mets: [4.8, 7, 10], sourceId: 'swimming' },
  { name: 'Strength training', aliases: /\bstrength|\bweight training|\bweights\b|\blifting|\bgym\b|\bresistance training/, mets: [3.5, 5, 6], sourceId: 'conditioning-exercise' },
  { name: 'HIIT', aliases: /\bhiit\b|\bhigh intensity interval/, mets: [7, 9, 11], sourceId: 'conditioning-exercise' },
  { name: 'Yoga', aliases: /\byoga\b|\bvinyasa\b|\bhatha\b/, mets: [2.3, 2.7, 4], sourceId: 'conditioning-exercise' },
];

function estimateWorkout(
  text: string,
  context: EstimationContext,
  resolved?: Resolved,
): LogOperation | ParsedCommand | null {
  const activity = activities.find((candidate) => candidate.aliases.test(text));
  if (!activity) return null;
  const spokenDuration = firstNumber(text, /(\d+(?:\.\d+)?)\s*(?:min|minute)/);
  const spokenHours = firstNumber(text, /(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b/);
  const duration = resolved?.minutes ?? (spokenDuration || spokenHours * 60);
  if (!duration) {
    return clarification(
      text,
      `How many minutes did you do ${activity.name.toLowerCase()}?`,
      ['15 min', '30 min', '45 min', '60 min'],
      { kind: 'workoutMinutes', activity: activity.name },
    );
  }
  if (!context.weightKg) {
    return clarification(
      text,
      'What is your current body weight? I need it to estimate active calories.',
      ['60 kg', '70 kg', '80 kg', '90 kg'],
      { kind: 'bodyWeight' },
    );
  }
  const spokenIntensity = /hard|intense|vigorous/.test(text)
    ? 'hard'
    : /easy|light|gentle|casual/.test(text)
      ? 'light'
      : /moderate|steady|brisk/.test(text)
        ? 'moderate'
        : undefined;
  const intensity = resolved?.intensity ?? spokenIntensity;
  if (!intensity) {
    return clarification(
      text,
      `How hard was the ${activity.name.toLowerCase()}?`,
      ['Light', 'Moderate', 'Hard'],
      { kind: 'workoutIntensity', activity: activity.name },
    );
  }
  const intensityIndex = intensity === 'hard' ? 2 : intensity === 'light' ? 0 : 1;
  const met = activity.mets[intensityIndex];
  const metLow = intensityIndex ? activity.mets[intensityIndex - 1] : met * 0.9;
  const metHigh = intensityIndex < 2 ? activity.mets[intensityIndex + 1] : met * 1.1;
  const activeKcal = (value: number) => Math.max(0, value - 1) * 3.5 * context.weightKg! / 200 * duration;
  return {
    type: 'workout',
    action: 'add',
    name: activity.name,
    durationMin: duration,
    calories: Math.round(activeKcal(met)),
    calorieLow: Math.round(activeKcal(metLow)),
    calorieHigh: Math.round(activeKcal(metHigh)),
    intensity,
    met,
    sourceLabel: '2024 Adult Compendium of Physical Activities',
    sourceId: activity.sourceId,
    confidence: 'low',
    basis: `${met} MET · ${context.weightKg} kg · ${duration} min · resting energy excluded`,
  };
}

export function parseCommandLocally(
  text: string,
  preferredSlot: MealSlot | undefined,
  rawContext: EstimationContext,
  resolved?: Resolved,
): ParsedCommand {
  // Details the user has already supplied outrank anything inferred, and a
  // bowl size or body weight given as an answer is true for the whole update.
  const context: EstimationContext = {
    ...rawContext,
    bowlMl: resolved?.bowlMl ?? rawContext.bowlMl,
    weightKg: resolved?.weightKg ?? rawContext.weightKg,
  };
  const lowered = text.toLowerCase().trim();
  const operations: LogOperation[] = [];
  // "coconut water" is a drink, not a glass of water. Only treat the word as
  // hydration when it is not part of a food this catalog knows.
  const wateryFood = foods.some((food) => food.aliases.some(
    (alias) => alias.includes('water') && new RegExp(`\\b${escapeRegex(alias)}(?:es|s)?\\b`).test(lowered),
  ));
  const drankWater = !wateryFood && (
    WATER_WORDS.test(lowered)
    || (/\bglass(?:es)?\b/.test(lowered)
      && !foods.some((food) => food.aliases.some((alias) => lowered.includes(alias))))
  );
  if (drankWater) operations.push({ type: 'water', action: 'add', amount: waterAmount(lowered) });
  const steps = firstNumber(lowered, /(\d[\d,]*)\s*steps?/);
  if (steps) operations.push({ type: 'steps', action: 'set', amount: steps });
  // "I walked 10000 steps" is a step count, not a walk of unknown duration.
  // Without this it matched the Walking activity and asked for minutes.
  const stepsOnly = Boolean(steps) && !/\b(?:min|minute|hour|hr)/.test(lowered);
  // `\D` rather than `.` between the word and the number: a greedy `.{0,12}`
  // swallowed "7." and captured the "5", so "slept 7.5 hours" logged 5 hours.
  const sleep = firstNumber(lowered, /(?:slept|sleep)\D{0,12}(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\D{0,12}sleep/);
  if (/sleep|slept/.test(lowered) && sleep) operations.push({ type: 'sleep', action: 'set', amount: sleep });
  const hasWeightIntent = /\b(?:weight|weigh|weighed)\b/.test(lowered);
  const explicitWeight = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*kg/);
  const weight = hasWeightIntent
    ? (parseWeightInput(lowered)?.kg ?? explicitWeight)
    : explicitWeight;
  const hasWorkoutLanguage = activities.some((activity) => activity.aliases.test(lowered));
  if ((hasWeightIntent || (hasWorkoutLanguage && !context.weightKg)) && weight) {
    operations.push({ type: 'weight', action: 'set', amount: weight });
  }

  const effectiveContext = { ...context, weightKg: (context.weightKg ?? weight) || undefined };
  const workout = stepsOnly ? null : estimateWorkout(lowered, effectiveContext, resolved);
  if (workout && !('type' in workout)) return workout;
  if (workout) operations.push(workout);

  // A food the user taught us wins outright: their own verified figure beats
  // both a generic USDA record and a category estimate.
  const taught = learnedMatch(lowered, context.learned);

  // Match foods, then keep only the most specific alias when one match's
  // alias is contained in another (e.g. "brown rice" beats "rice",
  // "egg white" beats "egg", "peanut butter" beats "butter").
  // Composite dishes used to be blocked from matching, because "butter chicken"
  // would find butter and chicken breast separately. They are catalog entries
  // in their own right now, and the longest-alias rule below keeps the parts
  // from winning against the whole.
  const rawMatches = foods
    // The *longest* matching alias, not the first. "chicken nuggets and fries"
    // matched the nuggets entry on its short alias "nuggets", which does not
    // contain "chicken", so the plain chicken-breast entry survived the
    // specificity filter below and the meal was logged twice.
    .map((food) => ({
      food,
      alias: food.aliases
        .filter((candidate) => aliasSaid(lowered, candidate))
        .sort((a, b) => b.length - a.length)[0],
    }))
    .filter((match): match is { food: FoodReference; alias: string } => Boolean(match.alias));
  const matches = rawMatches.filter(({ alias }) => !rawMatches.some(
    (other) => other.alias !== alias && other.alias.length > alias.length && other.alias.includes(alias),
  ));
  const hasExplicitOil = matches.some(({ food }) => food.name === 'Olive oil');

  const looksLikeMeal = MEAL_WORDS.test(lowered) || matches.length > 0 || Boolean(taught);
  if (looksLikeMeal) {
    if (taught) {
      operations.push({
        type: 'meal',
        action: 'add',
        slot: preferredSlot ?? inferMealSlot(lowered),
        description: text.trim(),
        items: [estimateLearned(taught.food, lowered, taught.alias)],
      });
    } else if (!matches.length) {
      const declaredCalories = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*(?:kcal|calories?)/);
      if (declaredCalories) {
        operations.push({
          type: 'meal',
          action: 'add',
          slot: preferredSlot ?? inferMealSlot(lowered),
          description: text.trim(),
          items: [{
            name: text.replace(/\d+(?:\.\d+)?\s*(?:kcal|calories?)/i, '').replace(/\bfrom (?:the )?label\b/i, '').trim() || 'Packaged food',
            quantity: 'amount described by user',
            calories: Math.round(declaredCalories),
            calorieLow: Math.round(declaredCalories * 0.95),
            calorieHigh: Math.round(declaredCalories * 1.05),
            protein: 0,
            carbs: 0,
            fat: 0,
            source: 'label',
            sourceLabel: LABEL_SOURCE,
            sourceId: 'declared serving',
            confidence: 'medium',
            basis: `${declaredCalories} kcal declared for the amount consumed`,
            assumptions: ['label rounding and serving accuracy still apply'],
          }],
        });
      } else if (!operations.length) {
        // Only claim not to know the food when nothing else in the sentence was
        // understood. This used to return unconditionally, throwing away the
        // water, steps or workout already read out of the same sentence — so
        // "I had 2 glasses of water" lost the water and asked about a food that
        // was never there, because "had" alone made it look like a meal.
        return clarification(
          text,
          'I don’t have a verified reference for that food yet. Give its label calories, or log the main parts with amounts.',
          ['e.g. 350 kcal from the label', 'e.g. 150 g rice and 1 bowl dal', 'e.g. 2 rotis and 100 g paneer'],
          { kind: 'unknownFood' },
        );
      }
    }
    const items: Omit<MealItem, 'id' | 'slot' | 'loggedAt'>[] = [];
    for (const { food, alias } of taught ? [] : matches) {
      const quantity = resolved?.amounts.get(alias)
        ?? quantityNear(lowered, alias, food, matches.length === 1);
      const estimate = estimateFood(food, quantity, lowered, context, alias, hasExplicitOil, resolved);
      if ('question' in estimate) {
        return clarification(text, estimate.question, estimate.suggestions, estimate.target);
      }
      items.push(estimate);
    }
    if (items.length) {
      operations.push({
        type: 'meal',
        action: 'add',
        slot: preferredSlot ?? inferMealSlot(lowered),
        description: text.trim(),
        items,
      });
    }
  }
  if (!operations.length) {
    return clarification(
      text,
      'What would you like me to log?',
      ['A meal', 'Water', 'A workout', 'Weight'],
      { kind: 'intent' },
    );
  }
  return {
    transcript: text.trim(),
    confirmation: operations.length === 1 ? '1 evidence-backed update ready' : `${operations.length} evidence-backed updates ready`,
    operations,
    source: 'local',
  };
}

function requestPayload(context: EstimationContext, pending?: PendingClarification) {
  return {
    weight_kg: context.weightKg,
    bowl_ml: context.bowlMl,
    cup_ml: context.cupMl,
    previous_transcript: pending?.transcript,
    clarification_question: pending?.target.kind,
  };
}

/** Answered details that belong on the profile rather than only this entry. */
function profileUpdatesFrom(answers: ClarificationAnswer[]) {
  const updates: ParsedCommand['profileUpdates'] = {};
  for (const answer of answers) {
    if (answer.kind === 'bowlMl') updates.bowlMl = answer.ml;
    if (answer.kind === 'bodyWeight') updates.weightKg = answer.kg;
  }
  return Object.keys(updates).length ? updates : undefined;
}

/**
 * Reached only if a question comes back that an answer should already have
 * settled. That is a bug rather than a user error, so it says something true
 * and offers a way forward instead of asking a fourth time.
 */
function stalled(transcript: string): ParsedCommand {
  return clarification(
    transcript,
    'I still cannot pin that down. Say it again in one line with the amount included, for example “2 rotis and 150 g rajma”.',
    ['Start over'],
    { kind: 'intent' },
  );
}

/**
 * Every logged update — typed or dictated — is parsed by the local reviewed
 * catalog. The remote AI service is a disabled-by-default extra: it is only
 * consulted when the local parser could not recognize a food or part of the
 * update, and only when `EXPO_PUBLIC_ENABLE_AI_PARSING=1`. The local result
 * is always the fallback, and it asks a clarifying question instead of
 * inventing a number.
 */
export async function parseFitnessCommand(
  text: string,
  preferredSlot: MealSlot | undefined,
  context: EstimationContext,
  pending?: PendingClarification,
): Promise<ParsedCommand> {
  // The utterance being parsed is always the original one. Answers are applied
  // as resolved details, never concatenated — see ClarificationTarget.
  const utterance = pending?.transcript ?? text;
  const resolved = resolveAnswers(pending?.answers);
  const local = parseCommandLocally(utterance, preferredSlot, context, resolved);

  if (pending?.answers.length && local.clarification) {
    // A question we have already answered must never come back.
    const repeat = pending.answers.some((answer) => answers(answer, local.clarification!.target));
    if (repeat) return { ...stalled(utterance), profileUpdates: profileUpdatesFrom(pending.answers) };
  }

  const withProfile: ParsedCommand = pending?.answers.length
    ? { ...local, profileUpdates: { ...profileUpdatesFrom(pending.answers), ...local.profileUpdates } }
    : local;

  const fragments = utterance
    .split(/\b(?:and|plus|aur|with|then)\b|,/i)
    .map((fragment) => fragment.trim())
    .filter((fragment) => fragment.length > 2);
  const fragmentUnknown = fragments.some((fragment) => {
    const fragmentResult = parseCommandLocally(fragment, preferredSlot, context, resolved);
    return fragmentResult.clarification?.question.includes(UNKNOWN_FOOD_MARK)
      || (!fragmentResult.operations.length && !fragmentResult.clarification);
  });
  const localUnknown = local.clarification?.question.includes(UNKNOWN_FOOD_MARK) ?? false;
  const needsAi = localUnknown || (Boolean(local.operations.length) && fragmentUnknown);
  if (!needsAi) return withProfile;
  // Off by default. The local result already asks the user for the missing
  // detail rather than guessing, so the app costs nothing to run and works
  // offline. The branch below stays here so the service can be switched back
  // on with EXPO_PUBLIC_ENABLE_AI_PARSING=1.
  if (!AI_PARSING_ENABLED) return withProfile;
  if (!apiUrl()) return withProfile;
  try {
    const session = await readSession();
    if (!session) return withProfile;
    const response = await apiRequest<ParsedCommand>('/v1/parse-command', {
      method: 'POST',
      body: JSON.stringify({
        text: utterance,
        preferred_slot: preferredSlot,
        ...requestPayload(context, pending),
      }),
    }, session);
    return { ...response, source: 'ai' } as ParsedCommand;
  } catch {
    return withProfile;
  }
}

/**
 * Advances a clarification by one round.
 *
 * `unknownFood` and `intent` are not single missing values, so their replies
 * are treated as ordinary text: an unknown food keeps the original wording so
 * "rajma chawal thali" still names the entry when label calories arrive, while
 * an unclear intent starts fresh from whatever the user just said.
 */
export async function advanceClarification(
  pending: PendingClarification,
  reply: string,
  preferredSlot: MealSlot | undefined,
  context: EstimationContext,
): Promise<Conversation> {
  const trimmed = reply.trim();
  if (!trimmed) return conversation(stalled(pending.transcript), []);

  // A restart: the reply is ordinary text, not a value, so nothing carries over.
  if (pending.target.kind === 'unknownFood') {
    return conversation(
      await parseFitnessCommand(`${pending.transcript}. ${trimmed}`, preferredSlot, context),
      [],
    );
  }
  if (pending.target.kind === 'intent') {
    return conversation(await parseFitnessCommand(trimmed, preferredSlot, context), []);
  }

  const answer = interpretClarificationAnswer(pending.target, trimmed);
  if (!answer) {
    // Say why once, keep the question, and keep everything already answered.
    return {
      result: {
        ...clarification(
          pending.transcript,
          rejection(pending.target),
          rejectionSuggestions(pending.target),
          pending.target,
        ),
        profileUpdates: profileUpdatesFrom(pending.answers),
      },
      pending,
    };
  }

  const settled = [...pending.answers.filter((existing) => !answers(existing, pending.target)), answer];
  const result = await parseFitnessCommand(pending.transcript, preferredSlot, context, {
    ...pending,
    answers: settled,
  });
  return conversation(result, settled);
}

/** Where a logging conversation stands: what to show, and what is still open. */
export type Conversation = { result: ParsedCommand; pending: PendingClarification | null };

function conversation(result: ParsedCommand, settled: ClarificationAnswer[]): Conversation {
  return { result, pending: openQuestion(result, settled) };
}

/** The open question after a parse, or null when there is nothing left to ask. */
export function openQuestion(
  result: ParsedCommand,
  settled: ClarificationAnswer[] = [],
): PendingClarification | null {
  return result.clarification
    ? { transcript: result.transcript, target: result.clarification.target, answers: settled }
    : null;
}

/** Chips for a rejected reply, in units the thing being asked about accepts. */
function rejectionSuggestions(target: ClarificationTarget) {
  if (target.kind === 'workoutIntensity') return ['Light', 'Moderate', 'Hard'];
  if (target.kind === 'workoutMinutes') return ['15 min', '30 min', '45 min', '60 min'];
  if (target.kind === 'bodyWeight') return ['60 kg', '70 kg', '80 kg', '90 kg'];
  if (target.kind === 'bowlMl') return ['150 ml', '200 ml', '250 ml', '300 ml'];
  if (target.kind === 'pieceGrams') return ['30 g', '50 g', '75 g', '100 g'];
  if (target.kind === 'foodAmount') {
    const food = foodFor(target.alias);
    if (food) return amountQuestion(food, target.alias).suggestions;
  }
  return ['100 g', '1 bowl', '1 piece', '1 cup'];
}

/** Said once, in the same words as the question, when a reply does not fit. */
function rejection(target: ClarificationTarget) {
  if (target.kind === 'workoutIntensity') return 'Was that light, moderate or hard?';
  if (target.kind === 'bodyWeight') return 'I need a weight, like “72 kg” or “160 lb”.';
  if (target.kind === 'workoutMinutes') return 'I need the time in minutes, like “30 min”.';
  if (target.kind === 'bowlMl') return 'I need a size in millilitres, like “250 ml”.';
  if (target.kind === 'pieceGrams') return 'I need a weight in grams, like “50 g”.';
  if (target.kind === 'foodAmount') {
    const food = foodFor(target.alias);
    if (food && food.pieceG && !food.density) {
      return `${food.name} is counted in pieces — how many did you have, or how many grams?`;
    }
    if (food?.density && !food.pieceG) {
      return `I need an amount for the ${spoken(food.name)}, like “250 ml”, “1 glass” or “150 g”.`;
    }
    return 'I need an amount, like “1 bowl”, “2 pieces” or “150 g”.';
  }
  return 'I need an amount, like “1 bowl”, “2 pieces” or “150 g”.';
}
