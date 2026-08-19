import assert from 'node:assert/strict';
import test from 'node:test';

import {
  elapsedSetSeconds,
  pauseSetTimer,
  remainingSetSeconds,
  resumeSetTimer,
  startSetTimer,
} from '../src/lib/set-timer.ts';

const at = (seconds: number) => new Date(1_750_000_000_000 + seconds * 1000);

test('duration timer uses an absolute deadline and survives a remount', () => {
  const timer = startSetTimer('exercise-1', 'set-1', 30, at(0));
  assert.equal(remainingSetSeconds(timer, at(0)), 30);
  assert.equal(remainingSetSeconds(JSON.parse(JSON.stringify(timer)), at(12)), 18);
  assert.equal(remainingSetSeconds(timer, at(31)), 0);
});

test('duration timer pauses and resumes without losing time', () => {
  const running = startSetTimer('exercise-1', 'set-1', 45, at(0));
  const paused = pauseSetTimer(running, at(10));
  assert.equal(paused.pausedRemainingSec, 35);
  assert.equal(remainingSetSeconds(paused, at(100)), 35);

  const resumed = resumeSetTimer(paused, at(100));
  assert.equal(remainingSetSeconds(resumed, at(100)), 35);
  assert.equal(remainingSetSeconds(resumed, at(136)), 0);
});

test('finishing early reports the time actually performed', () => {
  const timer = startSetTimer('exercise-1', 'set-1', 60, at(0));
  assert.equal(elapsedSetSeconds(timer, at(17)), 17);
  assert.equal(elapsedSetSeconds(timer, at(70)), 60);
});
