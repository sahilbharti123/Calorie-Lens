import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import test from 'node:test';

import { AGENT_PERSONAS, FEATURE_AREAS } from '../qa/agent-personas.mts';

const featureRoutes: Partial<Record<typeof FEATURE_AREAS[number], string[]>> = {
  onboarding: ['app/onboarding.tsx'],
  auth: ['app/auth.tsx'],
  'password-recovery': ['app/auth-reset.tsx'],
  today: ['app/(tabs)/index.tsx'],
  'quick-log': ['app/quick-log.tsx'],
  food: ['app/(tabs)/food.tsx'],
  'learned-foods': ['app/taught-foods.tsx'],
  water: ['app/water-log.tsx'],
  weight: ['app/weight-log.tsx'],
  progress: ['app/(tabs)/progress.tsx'],
  coach: ['app/(tabs)/coach.tsx'],
  settings: ['app/settings.tsx'],
  routines: ['app/routine-editor.tsx'],
  'exercise-library': ['app/exercise-picker.tsx'],
  'live-workout': ['app/workout-session.tsx'],
  'workout-history': ['app/workout-history.tsx'],
  'apple-watch': ['ios/VigorlyWatch/WorkoutView.swift', 'ios/VigorlyWatch/WatchWorkoutManager.swift'],
};

test('the product cohort contains exactly 30 unique real-life agents', () => {
  assert.equal(AGENT_PERSONAS.length, 30);
  assert.equal(new Set(AGENT_PERSONAS.map((agent) => agent.id)).size, 30);
  assert.equal(new Set(AGENT_PERSONAS.map((agent) => agent.name)).size, 30);
});

test('the 30-agent cohort covers every declared product area', () => {
  const covered = new Set(AGENT_PERSONAS.flatMap((agent) => agent.features));
  assert.deepEqual(FEATURE_AREAS.filter((feature) => !covered.has(feature)), []);
});

for (const agent of AGENT_PERSONAS) {
  test(`${agent.id} ${agent.name}: journey is executable and asks for product feedback`, () => {
    assert.ok(agent.realLifeContext.length >= 20);
    assert.ok(agent.journey.length >= 40);
    assert.ok(agent.suggestionPrompt.endsWith('?'));
    assert.ok(agent.features.length > 0);
    for (const feature of agent.features) {
      for (const route of featureRoutes[feature] ?? []) {
        assert.ok(existsSync(route), `${feature} is missing ${route}`);
      }
    }
  });
}
