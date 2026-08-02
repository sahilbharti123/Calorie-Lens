/**
 * Says every food in the catalog, every ordinary way, and checks what comes
 * back.
 *
 * The hundred-entry script proves a hundred sentences I thought of are right.
 * This one is the machine version: 167 foods against a dozen phrasings each is
 * about two thousand sentences, which is more than I will ever imagine by hand
 * and is where the remaining bugs actually live. Every failure so far has been
 * found by a person using the app, and the point of this file is to stop that
 * being the only way.
 *
 * Four things are checked on every sentence:
 *   1. it logs at all, rather than asking a question
 *   2. it logs the food that was named, and not a second phantom one
 *   3. the number is plausible for the amount described
 *   4. a food in company is still found — "X and rice" must give two items
 *
 *   node --import ./tests/register-alias.mjs scripts/catalog-sweep.mjs
 *   node --import ./tests/register-alias.mjs scripts/catalog-sweep.mjs --verbose
 */
import { FOODS } from '../src/lib/food-catalog.ts';
import { parseFitnessCommand } from '../src/lib/nutrition.ts';

const CTX = { cupMl: 200, bowlMl: 250, weightKg: 72 };
const verbose = process.argv.includes('--verbose');

/**
 * How a person names an amount. `wants` is the plausible calorie band as a
 * multiple of one helping — deliberately loose, because this is checking for
 * absurdity rather than grading the portion.
 */
const PHRASINGS = [
  { group: 'bare', say: (f) => f, low: 0.2, high: 4 },
  { group: 'i had', say: (f) => `I had ${f}`, low: 0.2, high: 4 },
  { group: 'i ate', say: (f) => `I ate some ${f}`, low: 0.2, high: 4 },
  { group: 'for lunch', say: (f) => `${f} for lunch`, low: 0.2, high: 4 },
  { group: 'counted', say: (f) => `2 ${f}`, low: 0.3, high: 9 },
  // Checked against the food's own per-100 g figure rather than against a
  // helping: 100 g of sugar really is 387 kcal, and comparing that to a
  // two-teaspoon helping would flag a correct answer.
  { group: 'weighed', say: (f) => `100 g of ${f}`, per100g: true },
  { group: 'hinglish', say: (f) => `maine ${f} khaya`, low: 0.2, high: 4 },
  { group: 'dictation', say: (f) => `um i had like some ${f}`, low: 0.2, high: 4 },
];

/** A food said alongside a second, unrelated one. Both must survive. */
const COMPANIONS = ['rice', 'roti'];

const failures = [];
const note = (group, say, problem) => failures.push({ group, say, problem });

let checked = 0;

for (const food of FOODS) {
  const name = food.aliases[0];
  // The helping this food logs from its own name, used as the yardstick for
  // every other phrasing of it.
  const plain = await parseFitnessCommand(name, undefined, CTX);
  const plainItems = plain.operations.flatMap((o) => (o.type === 'meal' ? o.items : []));
  const helping = plainItems.reduce((sum, item) => sum + item.calories, 0);

  for (const phrasing of PHRASINGS) {
    const say = phrasing.say(name);
    checked += 1;
    const parsed = await parseFitnessCommand(say, undefined, CTX);
    if (parsed.clarification) {
      note(phrasing.group, say, `asks: ${parsed.clarification.question}`);
      continue;
    }
    const items = parsed.operations.flatMap((o) => (o.type === 'meal' ? o.items : []));
    if (!items.length) {
      note(phrasing.group, say, 'logged nothing');
      continue;
    }
    if (!items.some((item) => item.name === food.name)) {
      note(phrasing.group, say, `logged ${items.map((i) => i.name).join(' + ')} instead`);
      continue;
    }
    if (items.length > 1) {
      note(phrasing.group, say, `logged an extra item: ${items.map((i) => i.name).join(' + ')}`);
      continue;
    }
    const total = items[0].calories;
    if (phrasing.per100g) {
      const expected = food.calories;
      if (Math.abs(total - expected) > Math.max(3, expected * 0.05)) {
        note(phrasing.group, say, `${total} kcal for 100 g of a ${expected} kcal/100 g food`);
      }
    } else if (helping >= 15 && (total < helping * phrasing.low || total > helping * phrasing.high)) {
      // Foods under 15 kcal a helping — black coffee, diet drinks, lettuce —
      // cannot be checked as a ratio without every rounding step looking like
      // a failure.
      note(phrasing.group, say, `${total} kcal against a ${Math.round(helping)} kcal helping (${items[0].quantity})`);
    }
  }

  // In company. A food that only parses when it is alone is not much use.
  for (const companion of COMPANIONS) {
    // Skip a companion the food's own name already contains — "ghee roti and
    // roti" is a sentence about one thing said twice, not a test of anything.
    if (food.aliases.some((alias) => alias.includes(companion))) continue;
    if (food.name.toLowerCase().includes(companion)) continue;
    const say = `I had ${name} and ${companion}`;
    checked += 1;
    const parsed = await parseFitnessCommand(say, undefined, CTX);
    if (parsed.clarification) {
      note('in company', say, `asks: ${parsed.clarification.question}`);
      continue;
    }
    const items = parsed.operations.flatMap((o) => (o.type === 'meal' ? o.items : []));
    if (!items.some((item) => item.name === food.name)) {
      note('in company', say, `lost the ${name}: got ${items.map((i) => i.name).join(' + ') || 'nothing'}`);
    } else if (items.length < 2) {
      note('in company', say, `lost the ${companion}: got only ${items.map((i) => i.name).join(' + ')}`);
    }
  }
}

const byGroup = new Map();
for (const failure of failures) {
  byGroup.set(failure.group, [...(byGroup.get(failure.group) ?? []), failure]);
}

console.log();
for (const [group, list] of [...byGroup.entries()].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`${group}  —  ${list.length} failing`);
  for (const failure of list.slice(0, verbose ? 999 : 8)) {
    console.log(`    ${JSON.stringify(failure.say).padEnd(46)} ${failure.problem}`);
  }
  if (!verbose && list.length > 8) console.log(`    … and ${list.length - 8} more`);
  console.log();
}

const passed = checked - failures.length;
console.log(`${passed}/${checked} sentences handled (${((passed / checked) * 100).toFixed(1)}%)`);
process.exit(failures.length ? 1 : 0);
