import assert from 'node:assert/strict';
import test from 'node:test';

import {
  advanceClarification,
  openQuestion,
  parseFitnessCommand,
} from '../src/lib/nutrition.ts';

/**
 * Exercises the logging parser the way people actually talk to it.
 *
 * Every case is a whole conversation: an opening utterance, then the replies a
 * user would give to whatever the app asks back. Two invariants are asserted on
 * every case regardless of what else it checks, because breaking either is what
 * made the app unusable:
 *
 *   1. the conversation ends — a question already answered is never re-asked
 *   2. the transcript never grows — an answer is applied to the detail it
 *      answers, not glued onto the end of what the user said
 *
 * The rounds are driven through `advanceClarification`, the same function the
 * logging screen calls, so a passing case means the screen behaves.
 */

const BASE = { cupMl: 200 };
const MAX_ROUNDS = 6;

const ops = (result) => result.operations.map((operation) => operation.type);
const meal = (result) => result.operations.find((operation) => operation.type === 'meal');
const kcal = (result) => {
  const entry = meal(result);
  return entry ? entry.items.reduce((sum, item) => sum + item.calories, 0) : 0;
};
const names = (result) => meal(result)?.items.map((item) => item.name) ?? [];
const grams = (result, name) => {
  const item = meal(result)?.items.find((candidate) => candidate.name.toLowerCase().includes(name));
  return item ? item.quantity : '(not found)';
};

/** Every case: what to say, what to answer, and what must be true at the end. */
const CASES = [
  {
    name: 'two rotis and a measured bowl of rajma',
    say: '2 rotis and one 200 ml bowl rajma',
    context: { bowlMl: 250 },
    check: (r) => [
      [ops(r).join() === 'meal', `expected one meal, got ${ops(r).join() || 'nothing'}`],
      [meal(r)?.items.length === 2, `expected 2 items, got ${meal(r)?.items.length}`],
      [grams(r, 'roti').includes('40 g'), `roti measured as ${grams(r, 'roti')}`],
    ],
  },
  {
    name: 'exact grams for two foods',
    say: '100 g paneer and 150 g rice',
    check: (r) => [
      [meal(r)?.items.length === 2, `expected 2 items, got ${meal(r)?.items.length}`],
      [meal(r)?.items.every((item) => item.confidence === 'high'), 'exact grams should be high confidence'],
    ],
  },
  {
    // "Brisk" is the standard description of a moderate walking pace — roughly
    // 5 km/h — not a hard effort. This test asserted 'hard' and so protected a
    // real mistake: it overstated a half-hour walk by a third.
    name: 'brisk walk with a known body weight',
    say: '30 minute brisk walk',
    context: { weightKg: 72 },
    check: (r) => [
      [ops(r).join() === 'workout', `expected a workout, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.intensity === 'moderate', `brisk should read as moderate, got ${r.operations[0]?.intensity}`],
      [r.operations[0]?.durationMin === 30, `expected 30 min, got ${r.operations[0]?.durationMin}`],
      [r.operations[0]?.calories === 106, `expected 106 kcal for 30 min at 72 kg, got ${r.operations[0]?.calories}`],
    ],
  },
  {
    name: 'water in glasses',
    say: 'I drank 2 glasses of water',
    check: (r) => [
      [ops(r).join() === 'water', `expected water, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.amount === 500, `expected 500 ml, got ${r.operations[0]?.amount}`],
    ],
  },
  {
    name: 'steps',
    say: '10000 steps today',
    check: (r) => [[r.operations[0]?.amount === 10000, `expected 10000 steps, got ${r.operations[0]?.amount}`]],
  },
  {
    name: 'sleep',
    say: 'I slept 7 hours',
    check: (r) => [
      [ops(r).includes('sleep'), `expected sleep, got ${ops(r).join() || 'nothing'}`],
      [r.operations.find((o) => o.type === 'sleep')?.amount === 7, 'expected 7 hours'],
    ],
  },
  {
    name: 'weight spoken without a unit',
    say: 'my weight is 78.4',
    check: (r) => [
      [ops(r).join() === 'weight', `expected weight, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.amount === 78.4, `expected 78.4, got ${r.operations[0]?.amount}`],
    ],
  },
  {
    name: 'a countable food with a count',
    say: 'I had 3 idlis for breakfast',
    slot: 'breakfast',
    check: (r) => [
      [meal(r)?.slot === 'breakfast', `expected breakfast, got ${meal(r)?.slot}`],
      [grams(r, 'idli').includes('50 g'), `idli measured as ${grams(r, 'idli')}`],
      [kcal(r) === 192, `3 idlis should be 192 kcal, got ${kcal(r)}`],
    ],
  },
  {
    // Sahil's screenshot. The recogniser heard "kne" for "one", so no amount
    // could be read at all — which used to stop the entry dead. It now assumes
    // one idli and says so, and the user can correct it on the review screen.
    name: 'garbled dictation still logs, with the assumption stated',
    say: 'i had a one bowl of italy. i had kne bowl of idli',
    context: { bowlMl: 250 },
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [kcal(r) === 64, `one assumed idli should be 64 kcal, got ${kcal(r)}`],
      [(meal(r)?.items[0]?.assumptions ?? []).some((a) => a.includes('assumed one usual serving')),
        'the assumption must be on the entry'],
      [meal(r)?.items[0]?.confidence === 'low', 'an assumed amount cannot be high confidence'],
    ],
  },
  {
    // A bowl of idli genuinely cannot be converted — idli has a piece weight
    // and no density — so this is the case that still has to ask.
    name: 'an unconvertible unit still asks, and the answer settles it',
    say: 'i had 1 bowl of idli',
    context: { bowlMl: 250 },
    replies: ['2 pieces'],
    check: (r) => [[kcal(r) === 128, `expected 128 kcal, got ${kcal(r)}`]],
  },
  {
    name: 'bowl size asked once, then remembered on the profile',
    say: '1 bowl dal',
    replies: ['250 ml'],
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [r.profileUpdates?.bowlMl === 250, `bowl size should be saved, got ${r.profileUpdates?.bowlMl}`],
    ],
  },
  {
    name: 'workout with neither duration nor intensity',
    say: 'I went for a walk',
    context: { weightKg: 72 },
    replies: ['30 min', 'moderate'],
    check: (r) => [
      [ops(r).join() === 'workout', `expected a workout, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.durationMin === 30, `expected 30 min, got ${r.operations[0]?.durationMin}`],
      [r.operations[0]?.intensity === 'moderate', `expected moderate, got ${r.operations[0]?.intensity}`],
    ],
  },
  {
    name: 'workout for someone who has never entered a weight',
    say: '30 minute run',
    replies: ['70 kg', 'hard'],
    check: (r) => [
      [ops(r).join() === 'workout', `expected a workout, got ${ops(r).join() || 'nothing'}`],
      [r.profileUpdates?.weightKg === 70, `weight should be saved, got ${r.profileUpdates?.weightKg}`],
      [r.operations[0]?.calories > 0, 'a hard 30 minute run should burn something'],
    ],
  },
  {
    name: 'a food the catalog does not know, resolved by label calories',
    say: 'I had a bowl of undhiyu',
    replies: ['350 kcal'],
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [kcal(r) === 350, `expected 350 kcal, got ${kcal(r)}`],
      [meal(r)?.items[0]?.source === 'label', 'should be recorded as a label figure'],
    ],
  },
  {
    name: 'nonsense opening, then a real one',
    say: 'hello',
    replies: ['2 rotis'],
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [r.transcript === '2 rotis', `transcript should restart, got "${r.transcript}"`],
    ],
  },
  {
    name: 'an answer that is not an answer, then a real one',
    say: '1 bowl rice',
    replies: ['banana', '200 ml'],
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [r.profileUpdates?.bowlMl === 200, `expected 200 ml bowl, got ${r.profileUpdates?.bowlMl}`],
    ],
  },
  {
    name: 'two eggs',
    say: '2 eggs',
    check: (r) => [[kcal(r) === 155, `2 × 50 g egg should be 155 kcal, got ${kcal(r)}`]],
  },
  {
    name: 'three foods in one sentence',
    say: '150 g paneer and 2 rotis and 1 cup rice',
    context: { bowlMl: 250 },
    check: (r) => [[meal(r)?.items.length === 3, `expected 3 items, got ${meal(r)?.items.length}`]],
  },
  {
    name: 'water in millilitres',
    say: '500 ml water',
    check: (r) => [[r.operations[0]?.amount === 500, `expected 500 ml, got ${r.operations[0]?.amount}`]],
  },
  {
    name: 'light yoga',
    say: '45 minute yoga light',
    context: { weightKg: 72 },
    check: (r) => [
      [r.operations[0]?.intensity === 'light', `expected light, got ${r.operations[0]?.intensity}`],
      [r.operations[0]?.durationMin === 45, `expected 45 min, got ${r.operations[0]?.durationMin}`],
    ],
  },
  {
    name: 'katori is a bowl',
    say: '1 katori dal',
    context: { bowlMl: 250 },
    check: (r) => [[grams(r, 'lentil').includes('250 ml bowl'), `dal measured as ${grams(r, 'lentil')}`]],
  },
  {
    name: 'hinglish',
    say: 'aaj maine 2 roti aur 1 bowl dal khaya',
    context: { bowlMl: 250 },
    check: (r) => [[meal(r)?.items.length === 2, `expected 2 items, got ${meal(r)?.items.length}`]],
  },
  {
    name: 'half a bowl',
    say: 'half bowl rice',
    context: { bowlMl: 300 },
    check: (r) => [[grams(r, 'rice').startsWith('0.5 ×'), `rice measured as ${grams(r, 'rice')}`]],
  },
  {
    name: 'a banana',
    say: 'I ate a banana',
    check: (r) => [[kcal(r) === 105, `one banana should be 105 kcal, got ${kcal(r)}`]],
  },
  {
    name: 'peanut butter beats butter',
    say: '2 tbsp peanut butter',
    check: (r) => [
      [meal(r)?.items.length === 1, `expected 1 item, got ${meal(r)?.items.length}`],
      [meal(r)?.items[0]?.name.includes('peanut'), `matched ${meal(r)?.items[0]?.name}`],
    ],
  },
  {
    name: 'milk by volume',
    say: 'I had 250 ml milk',
    check: (r) => [[kcal(r) > 0, 'milk should produce calories']],
  },
  {
    name: 'a food with neither a piece weight nor a density falls back to a helping',
    say: 'I had some chicken',
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [grams(r, 'chicken').startsWith('100 g helping'), `chicken measured as ${grams(r, 'chicken')}`],
    ],
  },
  {
    name: 'a second food with no amount does not kill the whole sentence',
    say: '2 rotis and dal',
    context: { bowlMl: 250 },
    check: (r) => [
      [meal(r)?.items.length === 2, `expected 2 items, got ${meal(r)?.items.length}`],
      [(meal(r)?.items ?? []).some((i) => i.quantity.includes('assumed')), 'the dal should be an assumed helping'],
    ],
  },
  {
    // "a" used to match the last letter of "thod-a", so this read as one
    // unnameable unit of paneer and asked how much.
    name: 'a word ending in a is not a quantity',
    say: 'thoda paneer khaya',
    check: (r) => [[ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`]],
  },
  {
    name: 'a step count is not a walk of unknown length',
    say: 'I walked 10000 steps',
    context: { weightKg: 72 },
    check: (r) => [
      [ops(r).join() === 'steps', `expected steps only, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.amount === 10000, `expected 10000, got ${r.operations[0]?.amount}`],
    ],
  },
  {
    name: 'an hour of cycling',
    say: 'I cycled for 1 hour, moderate',
    context: { weightKg: 72 },
    check: (r) => [[r.operations[0]?.durationMin === 60, `expected 60 min, got ${r.operations[0]?.durationMin}`]],
  },
  {
    name: 'meal and workout in one breath',
    say: '2 rotis and a 30 minute moderate walk',
    context: { bowlMl: 250, weightKg: 72 },
    check: (r) => [
      [ops(r).includes('meal') && ops(r).includes('workout'), `got ${ops(r).join() || 'nothing'}`],
    ],
  },
  {
    // "crunches" contains "run"; "moderate" and "water" contain "ate".
    name: 'crunches are not a run',
    say: 'I did 20 crunches',
    context: { weightKg: 72 },
    replies: ['2 rotis'],
    check: (r) => [[!ops(r).includes('workout'), 'crunches should not be logged as running']],
  },
  {
    name: 'a moderate walk is a walk, not an unknown food',
    say: 'I did a moderate 30 minute walk',
    context: { weightKg: 72 },
    check: (r) => [
      [ops(r).join() === 'workout', `expected a workout, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.intensity === 'moderate', `expected moderate, got ${r.operations[0]?.intensity}`],
    ],
  },
  {
    name: 'glasses of milk are food, not water',
    say: 'I drank 2 glasses of milk',
    check: (r) => [
      [!ops(r).includes('water'), 'milk should not be counted as water'],
      [ops(r).includes('meal'), `expected a meal, got ${ops(r).join() || 'nothing'}`],
    ],
  },
  {
    // Sahil's beer: the recogniser produced two sentences and put three words
    // between "500ml" and "beer". A stated unit is allowed to reach across
    // those, because the sentence names exactly one food.
    name: 'a drink whose amount is separated from its name',
    say: 'I had one 500 ML of whole garden beer. 500ml hoegarden flavoured beer',
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [kcal(r) === 215, `500 ml of beer should be 215 kcal, got ${kcal(r)}`],
      [meal(r)?.items[0]?.source === 'typical', 'beer is a typical value, not a USDA record'],
    ],
  },
  {
    name: 'a typical value says so, and carries a wide range',
    say: '2 slices of pizza',
    check: (r) => {
      const item = meal(r)?.items[0];
      const spread = item ? (item.calorieHigh - item.calorieLow) / item.calories : 0;
      return [
        [item?.sourceLabel === 'Typical composition', `labelled ${item?.sourceLabel}`],
        [item?.sourceId?.includes('category') ?? false, 'should not claim an FDC record'],
        [item?.confidence === 'low', `a category figure is low confidence, got ${item?.confidence}`],
        [spread > 0.8, `expected a wide range, got ${Math.round(spread * 100)}% of the midpoint`],
        [(item?.assumptions ?? []).some((a) => a.includes('typical figure')), 'should state the assumption'],
      ];
    },
  },
  {
    name: 'a verified record still says USDA and stays tight',
    say: '150 g paneer',
    check: (r) => {
      const item = meal(r)?.items[0];
      const spread = item ? (item.calorieHigh - item.calorieLow) / item.calories : 1;
      return [
        [item?.source === 'usda', `expected usda, got ${item?.source}`],
        [item?.sourceId?.startsWith('FDC ') ?? false, `expected an FDC id, got ${item?.sourceId}`],
        [spread < 0.1, `a weighed USDA food should be tight, got ${Math.round(spread * 100)}%`],
      ];
    },
  },
  {
    name: 'a composite dish beats its parts',
    say: '1 bowl butter chicken',
    context: { bowlMl: 250 },
    check: (r) => [
      [meal(r)?.items.length === 1, `expected 1 item, got ${meal(r)?.items.length}`],
      [meal(r)?.items[0]?.name === 'Butter chicken', `matched ${meal(r)?.items[0]?.name}`],
    ],
  },
  {
    // "glass" ends in an s. The old unit normaliser stripped it blindly and
    // produced "glas", which matched no branch, so every "a glass of X" asked
    // how much X you had.
    name: 'a glass of something is a glass',
    say: 'I had a glass of beer',
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [(meal(r)?.items[0]?.quantity ?? '').includes('250 ml glass'), `measured as ${meal(r)?.items[0]?.quantity}`],
    ],
  },
  {
    name: 'a drink counted rather than measured',
    say: 'two beers',
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [kcal(r) === 284, `two 330 ml beers should be 284 kcal, got ${kcal(r)}`],
    ],
  },
  {
    name: 'one of a thing, with no number at all',
    say: 'I had a beer',
    check: (r) => [[kcal(r) === 142, `one beer should be 142 kcal, got ${kcal(r)}`]],
  },
  {
    name: 'pints and pegs are real measures',
    say: 'a pint of beer',
    check: (r) => [
      [(meal(r)?.items[0]?.quantity ?? '').includes('568 ml'), `measured as ${meal(r)?.items[0]?.quantity}`],
    ],
  },
  {
    // "had" made this look like a meal, so the unknown-food branch fired and
    // returned, discarding the water it had already understood.
    name: 'water in a sentence that also looks like a meal',
    say: 'I had 2 glasses of water',
    check: (r) => [
      [ops(r).join() === 'water', `expected water only, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.amount === 500, `expected 500 ml, got ${r.operations[0]?.amount}`],
    ],
  },
  {
    // The litre pattern required the word to end there, so "litres" matched
    // nothing and two litres silently became one 250 ml glass.
    name: 'litres of water, plural and singular',
    say: 'I drank 2 litres of water',
    check: (r) => [[r.operations[0]?.amount === 2000, `expected 2000 ml, got ${r.operations[0]?.amount}`]],
  },
  {
    name: 'a litre with no digit',
    say: 'I drank a litre of water',
    check: (r) => [[r.operations[0]?.amount === 1000, `expected 1000 ml, got ${r.operations[0]?.amount}`]],
  },
  {
    // A greedy gap swallowed "7." and captured the 5.
    name: 'a fractional number is not read from its decimal',
    say: 'slept 7.5 hours',
    check: (r) => [[r.operations[0]?.amount === 7.5, `expected 7.5 hours, got ${r.operations[0]?.amount}`]],
  },
  {
    name: 'a workout and a drink in one sentence',
    say: 'I walked 30 minutes moderate and had a beer',
    context: { weightKg: 72 },
    check: (r) => [
      [ops(r).includes('workout') && ops(r).includes('meal'), `got ${ops(r).join() || 'nothing'}`],
    ],
  },
  {
    name: 'food and water together',
    say: 'I had 2 rotis and 500 ml water',
    check: (r) => [
      [ops(r).includes('meal') && ops(r).includes('water'), `got ${ops(r).join() || 'nothing'}`],
      [r.operations.find((o) => o.type === 'water')?.amount === 500, 'expected 500 ml of water'],
    ],
  },

  /* ------------------------------------------------------------------ *
   * Found by logging a hundred realistic entries and reading the numbers
   * rather than the operation types. Every case below was wrong in a way
   * no structural test could see: the entry appeared, and it was false.
   * ------------------------------------------------------------------ */
  {
    // "tomato sauce" is not a tomato. The sauce is already inside the dish.
    name: 'a compound name does not also log its ingredient',
    say: 'pasta with tomato sauce',
    check: (r) => [
      [names(r).length === 1, `expected one item, got ${names(r).join(' + ') || 'nothing'}`],
      [names(r)[0]?.includes('Pasta'), `expected the pasta dish, got ${names(r)[0]}`],
    ],
  },
  {
    name: 'coconut water is not a glass of water',
    say: 'coconut water',
    check: (r) => [
      [!ops(r).includes('water'), 'coconut water should not log drinking water'],
      [names(r)[0] === 'Coconut water', `got ${names(r)[0]}`],
    ],
  },
  {
    // The nuggets entry matched on its short alias "nuggets", which does not
    // contain "chicken", so a phantom chicken breast survived alongside it.
    name: 'the longest alias wins, so a dish is not double-counted',
    say: 'chicken nuggets and fries',
    check: (r) => [
      [names(r).length === 2, `expected 2 items, got ${names(r).join(' + ')}`],
      [!names(r).some((n) => n.includes('breast')), `phantom chicken breast in ${names(r).join(' + ')}`],
    ],
  },
  {
    name: 'a mango lassi is one drink, not a lassi and a mango',
    say: 'a mango lassi',
    check: (r) => [[names(r).length === 1, `expected 1 item, got ${names(r).join(' + ')}`]],
  },
  {
    // A packet is a stated amount. Asking for it is asking twice.
    name: 'a packet is a helping, not an unanswered question',
    say: 'a packet of chips',
    check: (r) => [
      [!r.clarification, `asked instead of logging: ${r.clarification?.question}`],
      [kcal(r) > 120 && kcal(r) < 260, `a packet of crisps should be roughly 190 kcal, got ${kcal(r)}`],
    ],
  },
  {
    name: 'a bowl of something with no density is still a helping',
    say: 'a bowl of namkeen',
    check: (r) => [[!r.clarification, `asked instead of logging: ${r.clarification?.question}`]],
  },
  {
    // One almond is 1.2 g, so assuming "one piece" logged 7 kcal.
    name: 'a handful of almonds is a handful',
    say: 'a handful of almonds',
    check: (r) => [[kcal(r) > 140 && kcal(r) < 210, `expected about 174 kcal, got ${kcal(r)}`]],
  },
  {
    // Nobody eats a 250 ml bowl of ghee, and the generic bowl fallback said
    // 2048 kcal. Condiments and fats need an explicit spoon-sized helping.
    name: 'a fat defaults to a spoonful, not a bowlful',
    say: 'ghee',
    check: (r) => [[kcal(r) < 150, `a helping of ghee should be a spoon, got ${kcal(r)} kcal`]],
  },
  {
    name: 'a glass of wine is a wine glass',
    say: 'a glass of red wine',
    check: (r) => [[kcal(r) > 110 && kcal(r) < 145, `expected about 126 kcal for 150 ml, got ${kcal(r)}`]],
  },
  {
    name: 'a glass of milk is still a tumbler',
    say: 'a glass of milk',
    check: (r) => [[kcal(r) > 140 && kcal(r) < 170, `expected about 155 kcal for 250 ml, got ${kcal(r)}`]],
  },
  {
    name: 'a latte is a latte cup, not a beer bottle',
    say: 'a latte',
    check: (r) => [[kcal(r) > 115 && kcal(r) < 150, `expected about 132 kcal for 240 ml, got ${kcal(r)}`]],
  },
  {
    // "tomatoes" is not "tomatos", so the bare s? plural matched nothing and
    // the sentence understood no food at all.
    name: 'an -es plural still names its food',
    say: '2 tomatoes',
    check: (r) => [
      [!r.clarification, `asked instead of logging: ${r.clarification?.question}`],
      [names(r)[0] === 'Tomato', `got ${names(r)[0]}`],
    ],
  },
  {
    // A bowl of idli holds several. Logging one would understate it badly, and
    // this is the one shape where the question is the honest answer.
    name: 'a vessel of a countable food still asks',
    say: 'i had 1 bowl of idli',
    context: { bowlMl: 250 },
    replies: ['3 pieces'],
    check: (r) => [[kcal(r) === 192, `expected 3 idlis at 192 kcal, got ${kcal(r)}`]],
  },
];


function describeTarget(target) {
  return 'alias' in target ? `${target.kind}:${target.alias}` : target.kind;
}

/** Plays a conversation to its end, collecting anything that went wrong. */
async function converse(testCase) {
  const context = { ...BASE, ...testCase.context };
  const replies = [...(testCase.replies ?? [])];
  const problems = [];
  const asked = [];

  let result = await parseFitnessCommand(testCase.say, testCase.slot, context);
  let pending = openQuestion(result);
  let expectedTranscript = result.transcript;
  let rounds = 0;

  while (pending && rounds < MAX_ROUNDS) {
    asked.push(describeTarget(pending.target));
    if (!replies.length) {
      problems.push(`ran out of replies while it still asked "${result.clarification.question}"`);
      break;
    }
    const reply = replies.shift();
    // 'intent' and 'unknownFood' legitimately restart from the reply.
    const restarts = pending.target.kind === 'intent' || pending.target.kind === 'unknownFood';
    const previous = pending;
    const step = await advanceClarification(pending, reply, testCase.slot, context);
    result = step.result;
    rounds += 1;

    if (restarts) {
      expectedTranscript = result.transcript;
    } else if (result.transcript !== expectedTranscript) {
      problems.push(
        `transcript grew after answering ${describeTarget(previous.target)}: `
        + `"${expectedTranscript}" became "${result.transcript}"`,
      );
      expectedTranscript = result.transcript;
    }

    pending = step.pending;
    if (!pending) break;
  }

  if (pending) problems.push(`still asking after ${rounds} rounds: ${asked.join(' → ')}`);
  if (result.clarification?.question.startsWith('I still cannot pin that down')) {
    problems.push(`gave up: ${asked.join(' → ')}`);
  }
  for (const [ok, message] of testCase.check(result)) {
    if (!ok) problems.push(message);
  }
  return problems;
}

for (const testCase of CASES) {
  test(testCase.name, async () => {
    const problems = await converse(testCase);
    assert.deepEqual(problems, [], `\n  · ${problems.join('\n  · ')}\n`);
  });
}

test('a question that cannot be settled gives up instead of looping', async () => {
  const stuck = await parseFitnessCommand('i had kne bowl of idli', undefined, { ...BASE, bowlMl: 250 }, {
    transcript: 'i had kne bowl of idli',
    target: { kind: 'foodAmount', alias: 'idli', foodName: 'Idli' },
    // A bowl of idli is unconvertible, so this answer cannot settle the question.
    answers: [{ kind: 'foodAmount', alias: 'idli', amount: 1, unit: 'bowl' }],
  });
  assert.ok(stuck.clarification, 'expected it to stop and say so');
  assert.match(stuck.clarification.question, /^I still cannot pin that down/);
});
