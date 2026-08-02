import assert from 'node:assert/strict';
import test from 'node:test';

import {
  parseWeightInput,
  targetWeightError,
  weightToTargetCopy,
} from '../src/lib/weight.ts';

test('dedicated weight parsing recovers a number even when speech mishears weight', () => {
  assert.equal(parseWeightInput('Current rate is 89')?.kg, 89);
  assert.equal(parseWeightInput('My current weight is 82.4 kg')?.kg, 82.4);
});

test('weight parsing converts pounds and rejects implausible readings', () => {
  assert.equal(parseWeightInput('196 lb')?.kg, 88.9);
  assert.equal(parseWeightInput('9 kg'), null);
  assert.equal(parseWeightInput('no number here'), null);
});

test('target weight follows the selected goal direction', () => {
  assert.equal(targetWeightError('lose-fat', 89, 82), null);
  assert.match(targetWeightError('lose-fat', 89, 92) ?? '', /below/);
  assert.match(targetWeightError('build-muscle', 70, 66) ?? '', /above/);
  assert.match(targetWeightError('maintain', 70, 64) ?? '', /maintenance/);
  assert.equal(weightToTargetCopy(89, 82), '7.0 kg to lose toward target');
});
