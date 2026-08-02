/**
 * Throws several hundred realistic phrasings at the parser at once and groups
 * what breaks by cause rather than by instance.
 *
 * The tests in tests/ are a safety net for bugs already found. This is the
 * opposite tool: it exists to find the ones nobody has hit yet, because every
 * failure so far has been discovered by a person using the app rather than by
 * me imagining a sentence. A grouped report is the point — twenty broken cases
 * are usually two broken rules, and fixing rules is what stops the next twenty.
 *
 *   node --import ./tests/register-alias.mjs scripts/phrasing-sweep.mjs
 *   node --import ./tests/register-alias.mjs scripts/phrasing-sweep.mjs --verbose
 */
import { parseFitnessCommand } from '../src/lib/nutrition.ts';

const CTX = { cupMl: 200, bowlMl: 250, weightKg: 72 };
const verbose = process.argv.includes('--verbose');

/** say → the operation types this sentence must produce. */
const cases = [];
const add = (group, say, want, opts = {}) => cases.push({ group, say, want: [].concat(want), ...opts });

/* ---------------------------------------------------------------- units --- */
// Every way a person says the same measurement, against a food that supports it.
const GRAM_WORDS = ['100 g', '100g', '100 gm', '100 gms', '100 gram', '100 grams'];
for (const amount of GRAM_WORDS) add('grams', `I had ${amount} of paneer`, 'meal');
for (const amount of ['0.2 kg', '0.2 kilo', '0.2 kilos', '0.2 kilograms', '200 grams']) {
  add('kilos', `I ate ${amount} chicken`, 'meal');
}
for (const amount of ['250 ml', '250ml', '250 millilitres', '250 milliliters', '0.25 l', '0.25 litre', '0.25 liters']) {
  add('millilitres', `I drank ${amount} of milk`, 'meal');
}
for (const unit of ['bowl', 'bowls', 'katori', 'katoris']) add('bowls', `I had 2 ${unit} of dal`, 'meal');
for (const unit of ['cup', 'cups', 'mug', 'mugs']) add('cups', `I had 1 ${unit} of rice`, 'meal');
for (const unit of ['glass', 'glasses']) add('glasses', `I had 2 ${unit} of milk`, 'meal');
for (const unit of ['piece', 'pieces', 'slice', 'slices']) add('pieces', `I ate 2 ${unit} of pizza`, 'meal');
for (const unit of ['bottle', 'bottles', 'can', 'cans', 'pint', 'pints']) add('vessels', `I had 1 ${unit} of beer`, 'meal');
for (const unit of ['peg', 'pegs', 'shot', 'shots']) add('vessels', `I had 2 ${unit} of whisky`, 'meal');
for (const unit of ['tbsp', 'tsp']) add('spoons', `I had 2 ${unit} of peanut butter`, 'meal');
for (const unit of ['plate', 'plates', 'serving', 'servings']) add('servings', `I had 1 ${unit} of fried rice`, 'meal');

/* ------------------------------------------------------------- counting --- */
for (const number of ['a', 'an', 'one', '1', 'two', '2', 'three', '3', 'four', 'half a', 'a couple of', 'a few']) {
  add('counting', `I had ${number} banana`, 'meal');
}
for (const drink of ['beer', 'coffee', 'chai', 'coke', 'lassi']) {
  add('counting-drinks', `I had a ${drink}`, 'meal');
  add('counting-drinks', `I had two ${drink}s`, 'meal');
}

/* ------------------------------------------------------------ fragments --- */
// Dictation drops verbs constantly, and people type shorthand.
for (const say of [
  '2 rotis', '2 rotis and dal', '500 ml beer', 'beer 500 ml', '100g paneer',
  'paneer 100 g', 'rice and dal', 'banana', '3 idlis', 'chai', 'one beer',
  'dal chawal', '2 eggs and toast',
]) add('fragments', say, 'meal');

/* -------------------------------------------------------------- hinglish -- */
for (const say of [
  'maine 2 roti khaya', 'aaj maine dal chawal khaya', 'ek katori dal khayi',
  'do roti aur sabzi', 'teen idli khaye', 'maine ek glass doodh piya',
  'aadha bowl rice khaya', 'thoda paneer khaya', 'maine chai pi',
  '2 roti aur 1 katori rajma khaya',
]) add('hinglish', say, 'meal');

/* ------------------------------------------------------------ multi-item -- */
for (const say of [
  '2 rotis and 1 bowl dal', '2 rotis, 1 bowl dal', '2 rotis plus a bowl of dal',
  '2 rotis with dal', 'rice, dal and salad', '2 eggs, 2 toast and a coffee',
  '150 g paneer and 2 rotis and 1 cup rice', 'pizza and a coke',
  'chicken curry with rice and a beer', 'dal, rice, roti and curd',
]) add('multi-item', say, 'meal');

/* ------------------------------------------------------------ meal slots -- */
for (const slot of ['breakfast', 'lunch', 'dinner', 'snack']) {
  add('slots', `for ${slot} I had 2 rotis and dal`, 'meal');
  add('slots', `${slot}: 2 rotis and dal`, 'meal');
}

/* ----------------------------------------------------------------- water -- */
// The amount matters as much as the type. An earlier sweep passed every one of
// these while "2 litres" silently logged a single 250 ml glass, because it only
// checked that *a* water entry appeared.
for (const [say, ml] of [
  ['I drank 500 ml of water', 500], ['500 ml water', 500],
  ['I had 2 glasses of water', 500], ['two glasses of water', 500],
  ['I drank a litre of water', 1000], ['drank 1 l water', 1000],
  ['I drank 2 litres of water', 2000], ['3 liters of water', 3000],
  ['half a litre of water', 500], ['I had a bottle of water', 500],
  ['I had 3 glasses of pani', 750], ['maine 2 glass paani piya', 500],
  ['a glass of water', 250], ['I drank water', 250],
]) add('water', say, 'water', { amount: ml });

/* ----------------------------------------------------------------- steps -- */
for (const [say, count] of [
  ['10000 steps', 10000], ['I walked 10000 steps', 10000],
  ['I did 8000 steps today', 8000], ['12,000 steps', 12000],
]) add('steps', say, 'steps', { amount: count });

/* ----------------------------------------------------------------- sleep -- */
for (const [say, hours] of [
  ['I slept 7 hours', 7], ['slept 7.5 hours', 7.5],
  ['I got 6 hours of sleep', 6], ['8 hours sleep', 8],
]) add('sleep', say, 'sleep', { amount: hours });

/* ---------------------------------------------------------------- weight -- */
for (const [say, kg] of [
  ['my weight is 78.4', 78.4], ['I weigh 78.4 kg', 78.4], ['weight 78', 78],
  ['I weighed 78.4 kilos today', 78.4], ['my weight today is 80', 80],
  ['78.4 kg weight', 78.4],
]) add('weight', say, 'weight', { amount: kg });

/* -------------------------------------------------------------- workouts -- */
for (const say of [
  '30 minute brisk walk', 'I walked for 30 minutes moderate',
  'I ran for 20 minutes hard', '45 min moderate cycling',
  'I did yoga for 45 minutes light', 'gym for 1 hour moderate',
  'I did a 30 min easy jog', 'went for a moderate 40 minute walk',
  'strength training 45 minutes moderate', '1 hour moderate swim',
]) add('workouts', say, 'workout', { allowAsk: /swim/.test(say) });

/* ----------------------------------------------------------------- mixed -- */
add('mixed', 'I had 2 rotis and drank 500 ml water', ['meal', 'water']);
add('mixed', 'I walked 30 minutes moderate and had a beer', ['workout', 'meal']);
add('mixed', '10000 steps and I slept 7 hours', ['steps', 'sleep']);
add('mixed', 'I had a beer and my weight is 78.4', ['meal', 'weight']);
add('mixed', 'breakfast was 3 idlis and I drank a glass of water', ['meal', 'water']);

/* --------------------------------------------------------- dictation mess - */
for (const say of [
  'i had one 500 ML of whole garden beer. 500ml hoegarden flavoured beer',
  'i had kne bowl of idli',
  'i had a one bowl of italy. i had kne bowl of idli',
  'i had two rotis i had two rotis',
  'so i had like 2 rotis and some dal',
  'um i had a beer',
  'i ate 2 rotis and uh 1 bowl of dal',
  'today i had 3 idlis for breakfast and then a coffee',
  'i had 200 grams of chicken breast grilled',
]) add('dictation', say, 'meal', { allowAsk: /idli|italy/.test(say) });

/* ------------------------------------------------------------- qualifiers - */
for (const say of [
  'about 200 g of rice', 'roughly 2 rotis', 'around 100 grams paneer',
  'approximately 500 ml beer', 'nearly 2 bowls of dal', 'just 1 roti',
  'only 100 g rice', 'i had like 2 eggs',
]) add('qualifiers', say, 'meal');

/* ----------------------------------------------------------------- brands - */
for (const say of [
  'I had a maggi', 'i drank a red bull', 'i had a dairy milk',
  'i had 2 parle g biscuits', 'i had a coke zero', 'i ate kfc chicken',
]) add('brands', say, 'meal');

/* -------------------------------------------------------------- unknowns -- */
for (const say of ['I had undhiyu', 'i ate my mums special curry', 'i had a thali']) {
  add('unknown-food', say, [], { expectAsk: 'unknownFood' });
}

/* ---------------------------------------------------------------- runner -- */
const results = [];
for (const testCase of cases) {
  const parsed = await parseFitnessCommand(testCase.say, undefined, CTX);
  const got = parsed.operations.map((operation) => operation.type);
  const asked = parsed.clarification?.target.kind;

  let problem = null;
  if (testCase.expectAsk) {
    if (asked !== testCase.expectAsk) problem = `expected to ask ${testCase.expectAsk}, ${asked ? `asked ${asked}` : `logged ${got.join()}`}`;
  } else if (asked && !testCase.allowAsk) {
    problem = `asked ${asked}`;
  } else if (!asked) {
    const missing = testCase.want.filter((type) => !got.includes(type));
    if (missing.length) {
      problem = `missing ${missing.join()} (got ${got.join() || 'nothing'})`;
    } else if (testCase.amount !== undefined) {
      const operation = parsed.operations.find((candidate) => candidate.type === testCase.want[0]);
      if (operation?.amount !== testCase.amount) {
        problem = `expected ${testCase.amount}, logged ${operation?.amount}`;
      }
    }
  }
  results.push({ ...testCase, got, asked, problem });
}

const failures = results.filter((result) => result.problem);
const byGroup = new Map();
for (const failure of failures) {
  const list = byGroup.get(failure.group) ?? [];
  list.push(failure);
  byGroup.set(failure.group, list);
}

console.log();
for (const [group, list] of [...byGroup.entries()].sort((a, b) => b[1].length - a[1].length)) {
  const total = results.filter((result) => result.group === group).length;
  console.log(`${group}  —  ${list.length}/${total} failing`);
  for (const failure of list.slice(0, verbose ? 99 : 6)) {
    console.log(`    ${JSON.stringify(failure.say).padEnd(52)} ${failure.problem}`);
  }
  if (!verbose && list.length > 6) console.log(`    … and ${list.length - 6} more`);
  console.log();
}

const passed = results.length - failures.length;
console.log(`${passed}/${results.length} phrasings handled (${Math.round((passed / results.length) * 100)}%)`);
process.exit(failures.length ? 1 : 0);
