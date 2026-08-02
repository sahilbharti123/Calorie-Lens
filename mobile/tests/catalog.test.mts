import assert from 'node:assert/strict';
import test from 'node:test';

import { FOODS } from '../src/lib/food-catalog.ts';
import { parseFitnessCommand } from '../src/lib/nutrition.ts';

/**
 * Checks the catalog against itself.
 *
 * Every bug this file guards against passed every other test in the project.
 * "Ghee" logged 2048 kcal, "a packet of chips" logged 545, and one chicken
 * nugget logged 49 — all of them structurally perfect entries with a number on
 * them that no person would recognise as their dinner. Nothing that looks at
 * operation types can see that, so this file looks at the numbers.
 *
 * The rule being enforced is simple: say the name of a food, get a plausible
 * helping of it. If that fails, the app is lying quietly, which is worse than
 * asking a question.
 */

const CONTEXT = { cupMl: 200, bowlMl: 250, weightKg: 72 };

/** What the app logs when the user says nothing but the food's own name. */
async function defaultHelping(alias: string) {
  const parsed = await parseFitnessCommand(alias, undefined, CONTEXT);
  const items = parsed.operations.flatMap((operation) => (operation.type === 'meal' ? operation.items : []));
  return { items, clarification: parsed.clarification };
}

test('every food logs from its own name without a question', async () => {
  const broken: string[] = [];
  for (const food of FOODS) {
    const { items, clarification } = await defaultHelping(food.aliases[0]);
    if (clarification) broken.push(`${food.aliases[0]} — asks "${clarification.question}"`);
    else if (!items.length) broken.push(`${food.aliases[0]} — understood nothing`);
  }
  assert.deepEqual(broken, [], `\n  · ${broken.join('\n  · ')}\n`);
});

test('a default helping is a plausible amount of food', async () => {
  // Wide on purpose. This is not a check that the portion is ideal — it is a
  // check that it is not absurd. A bowl of cooking oil and a single almond both
  // sit far outside it; a big restaurant meal sits inside.
  const MIN_KCAL = 10;
  const MAX_KCAL = 1300;
  const implausible: string[] = [];
  for (const food of FOODS) {
    const { items } = await defaultHelping(food.aliases[0]);
    const total = items.reduce((sum, item) => sum + item.calories, 0);
    // Water, black coffee and diet drinks are genuinely near zero.
    const nearlyFree = food.calories <= 15;
    if (!nearlyFree && total < MIN_KCAL) implausible.push(`${food.name} — ${total} kcal is too little for a helping`);
    if (total > MAX_KCAL) implausible.push(`${food.name} — ${total} kcal is too much for one helping (${items[0]?.quantity})`);
  }
  assert.deepEqual(implausible, [], `\n  · ${implausible.join('\n  · ')}\n`);
});

test('the macros on a default helping reconcile with its calories', async () => {
  // 4 kcal/g of protein and carbohydrate, 9 for fat. A catalog row with a fat
  // figure typed wrong passes every other test ever written and still shows the
  // user a number that contradicts itself.
  const off: string[] = [];
  for (const food of FOODS) {
    const { items } = await defaultHelping(food.aliases[0]);
    for (const item of items) {
      const implied = item.protein * 4 + item.carbs * 4 + item.fat * 9;
      // Alcohol carries its energy as ethanol at 7 kcal/g, which is not one of
      // the three macros, so beer and wine legitimately fail this arithmetic.
      const alcoholic = /beer|wine|whisky|vodka|gin|rum|cocktail|spirit/i.test(item.name);
      if (alcoholic || item.calories < 25 || !implied) continue;
      const drift = Math.abs(implied - item.calories) / item.calories;
      if (drift > 0.3) {
        off.push(`${item.name} — ${item.calories} kcal but macros imply ${Math.round(implied)}`);
      }
    }
  }
  assert.deepEqual(off, [], `\n  · ${off.join('\n  · ')}\n`);
});

test('no two foods claim the same alias', async () => {
  // Two entries answering to one word means the winner is decided by catalog
  // order, which is nobody's intent and changes whenever a food is added.
  const owners = new Map<string, string[]>();
  for (const food of FOODS) {
    for (const alias of food.aliases) {
      owners.set(alias, [...(owners.get(alias) ?? []), food.name]);
    }
  }
  const clashes = [...owners.entries()]
    .filter(([, names]) => names.length > 1)
    .map(([alias, names]) => `"${alias}" claimed by ${names.join(' and ')}`);
  assert.deepEqual(clashes, [], `\n  · ${clashes.join('\n  · ')}\n`);
});

test('a verified record carries an FDC id and a typical one never does', async () => {
  const wrong = FOODS
    .filter((food) => (food.tier === 'typical' ? Boolean(food.fdcId) : !food.fdcId))
    .map((food) => `${food.name} — tier ${food.tier ?? 'usda'}, fdcId ${food.fdcId ?? 'none'}`);
  assert.deepEqual(wrong, [], `\n  · ${wrong.join('\n  · ')}\n`);
});
