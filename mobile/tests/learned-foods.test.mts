import assert from 'node:assert/strict';
import test from 'node:test';

import {
  aliasesForName,
  learnableFrom,
  learnableName,
  parseFitnessCommand,
} from '../src/lib/nutrition.ts';

const BASE = { cupMl: 200, bowlMl: 250 };

/** The whole point: tell it once, and it knows the next time. */
async function teach(said: string, reply: string) {
  const asked = await parseFitnessCommand(said, undefined, BASE);
  assert.ok(asked.clarification, `expected a question for "${said}"`);
  assert.equal(asked.clarification.target.kind, 'unknownFood');

  const logged = await parseFitnessCommand(`${said}. ${reply}`, undefined, BASE);
  const meal = logged.operations.find((operation) => operation.type === 'meal');
  assert.ok(meal, 'expected the label figure to log');

  const learned = learnableFrom(logged.transcript, meal.items[0]);
  assert.ok(learned, 'expected it to be learnable');
  return { ...learned, id: 'x', createdAt: 'now', updatedAt: 'now' };
}

test('a name drops amounts, filler and dictation repeats', () => {
  const named = learnableName('I had one 500 ML of whole garden beer. 500ml hoegarden flavoured beer');
  assert.ok(named);
  assert.ok(!/\d/.test(named.name), `name still has a number: "${named.name}"`);
  assert.ok(!named.name.includes('had'), `name still has filler: "${named.name}"`);
  // "beer" is said twice; it belongs in the name once.
  assert.equal(named.name.split(' ').filter((word) => word === 'beer').length, 1);
});

test('a taught food is recognised next time, with no question', async () => {
  const food = await teach('I had a bowl of undhiyu', '350 kcal');
  const again = await parseFitnessCommand('I had undhiyu', undefined, { ...BASE, learned: [food] });
  assert.equal(again.clarification, undefined, 'it should not ask a second time');
  const meal = again.operations.find((operation) => operation.type === 'meal');
  assert.equal(meal?.items[0]?.calories, 350);
  assert.equal(meal?.items[0]?.sourceLabel, 'Your own figure');
});

test('a taught drink scales when a different amount is said', async () => {
  const food = await teach('I had 500 ml of grandmas kanji', '240 kcal');
  assert.equal(food.servingAmount, 500);
  assert.equal(food.servingUnit, 'ml');

  const half = await parseFitnessCommand('I had 250 ml of grandmas kanji', undefined, { ...BASE, learned: [food] });
  const meal = half.operations.find((operation) => operation.type === 'meal');
  assert.equal(meal?.items[0]?.calories, 120, 'half the amount is half the calories');
  assert.match(meal?.items[0]?.quantity ?? '', /scaled from 500 ml/);
});

test('the user figure beats the shipped catalog', async () => {
  const food = {
    id: 'x',
    name: 'my protein shake',
    aliases: ['my protein shake', 'protein shake'],
    calories: 180,
    protein: 30,
    carbs: 4,
    fat: 2,
    servingAmount: 1,
    servingUnit: 'glass',
    createdAt: 'now',
    updatedAt: 'now',
  };
  const result = await parseFitnessCommand('I had a protein shake', undefined, { ...BASE, learned: [food] });
  const meal = result.operations.find((operation) => operation.type === 'meal');
  assert.equal(meal?.items.length, 1, 'the taught food replaces the catalog match, not adds to it');
  assert.equal(meal?.items[0]?.calories, 180);
});

test('an entry that came from a saved food is not re-learned', () => {
  const fromSaved = {
    calories: 350,
    protein: 0,
    carbs: 0,
    fat: 0,
    sourceLabel: 'Your own figure',
  };
  assert.equal(learnableFrom('I had undhiyu', fromSaved), null);
});

test('renaming regenerates what the app listens for', async () => {
  // The derived name is only as good as the dictation, so a rename has to
  // change matching too — otherwise the corrected food is still findable only
  // under the garbled name.
  const garbled = learnableName('I had one 500 ML of whole garden beer. 500ml hoegarden beer');
  assert.ok(garbled);
  assert.ok(garbled.aliases.includes('hoegarden'), 'the garbled word is what it listens for');

  const renamed = aliasesForName('Hoegaarden');
  assert.deepEqual(renamed, ['hoegaarden']);

  const food = {
    id: 'x',
    name: 'Hoegaarden',
    aliases: renamed,
    calories: 240,
    protein: 2,
    carbs: 18,
    fat: 0,
    servingAmount: 500,
    servingUnit: 'ml',
    createdAt: 'now',
    updatedAt: 'now',
  };
  const result = await parseFitnessCommand('I had 500 ml hoegaarden', undefined, { ...BASE, learned: [food] });
  const meal = result.operations.find((operation) => operation.type === 'meal');
  assert.equal(meal?.items[0]?.calories, 240, 'the renamed food is found under its new name');
  assert.equal(meal?.items[0]?.name, 'Hoegaarden');
});

test('a one-word name keeps matching, and short filler is not an alias', () => {
  assert.deepEqual(aliasesForName('Coke'), ['coke']);
  // Two-word name: the phrase plus any word long enough to stand alone.
  assert.deepEqual(aliasesForName('my kanji'), ['my kanji', 'kanji']);
});
