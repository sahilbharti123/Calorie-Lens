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
    name: 'brisk walk with a known body weight',
    say: '30 minute brisk walk',
    context: { weightKg: 72 },
    check: (r) => [
      [ops(r).join() === 'workout', `expected a workout, got ${ops(r).join() || 'nothing'}`],
      [r.operations[0]?.intensity === 'hard', `brisk should read as hard, got ${r.operations[0]?.intensity}`],
      [r.operations[0]?.durationMin === 30, `expected 30 min, got ${r.operations[0]?.durationMin}`],
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
    // Sahil's screenshot: the recogniser heard "kne" for "one", so no amount
    // could be read, and "1 bowl" is not answerable for a food with no density.
    name: 'garbled dictation, then a bowl answer that cannot work, then a count',
    say: 'i had a one bowl of italy. i had kne bowl of idli',
    context: { bowlMl: 250 },
    replies: ['1 bowl', '2'],
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [kcal(r) === 128, `2 idlis should be 128 kcal, got ${kcal(r)}`],
    ],
  },
  {
    name: 'the same garbled line answered in pieces first time',
    say: 'i had kne bowl of idli',
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
    say: 'I had shahi tukda',
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
    name: 'a food with neither a piece weight nor a density',
    say: 'I had some chicken',
    replies: ['1 bowl', '200 g'],
    check: (r) => [
      [ops(r).join() === 'meal', `expected a meal, got ${ops(r).join() || 'nothing'}`],
      [grams(r, 'chicken') === '200 g', `chicken measured as ${grams(r, 'chicken')}`],
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
    name: 'food and water together',
    say: 'I had 2 rotis and 500 ml water',
    check: (r) => [
      [ops(r).includes('meal') && ops(r).includes('water'), `got ${ops(r).join() || 'nothing'}`],
      [r.operations.find((o) => o.type === 'water')?.amount === 500, 'expected 500 ml of water'],
    ],
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
