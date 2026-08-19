import assert from 'node:assert/strict';
import test from 'node:test';

import { metricByNewest, resolveMealLocations } from '@/src/lib/sync-merge.ts';

test('a newer downward water correction beats a larger stale value', () => {
  assert.deepEqual(
    metricByNewest(1000, '2026-08-19T10:01:00.000Z', 2000, '2026-08-19T10:00:00.000Z'),
    { value: 1000, updatedAt: '2026-08-19T10:01:00.000Z' },
  );
});

test('a newer zero remains a valid steps or sleep correction', () => {
  assert.equal(
    metricByNewest(0, '2026-08-19T10:02:00.000Z', 8000, '2026-08-19T10:00:00.000Z').value,
    0,
  );
});

test('legacy values without timestamps keep the local device value', () => {
  assert.equal(metricByNewest(750, undefined, 500, undefined).value, 750);
});

test('a meal moved on one device cannot remain duplicated on its old day', () => {
  const old = {
    id: 'meal-1', name: 'Dal', quantity: '1 bowl', calories: 200,
    protein: 10, carbs: 30, fat: 4, slot: 'lunch' as const,
    loggedAt: '2026-08-18T12:00:00.000Z', updatedAt: '2026-08-18T12:00:00.000Z',
    source: 'manual' as const,
  };
  const moved = {
    ...old,
    dayKey: '2026-08-19',
    slot: 'dinner' as const,
    loggedAt: '2026-08-19T12:00:00.000Z',
    updatedAt: '2026-08-19T18:00:00.000Z',
  };
  const result = resolveMealLocations({ '2026-08-18': [old], '2026-08-19': [moved] });
  assert.equal(result.length, 1);
  assert.equal(result[0].date, '2026-08-19');
  assert.equal(result[0].meal.slot, 'dinner');
});
