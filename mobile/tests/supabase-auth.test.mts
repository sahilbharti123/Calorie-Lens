import assert from 'node:assert/strict';
import test from 'node:test';

import { parseAuthLink, toAuthSession } from '../src/lib/supabase-auth.ts';

test('Supabase auth links parse query-string tokens', () => {
  assert.deepEqual(
    parseAuthLink('vigorly://auth-reset?access_token=access-1&refresh_token=refresh-1&type=recovery'),
    { accessToken: 'access-1', refreshToken: 'refresh-1', type: 'recovery' },
  );
});

test('Supabase auth links parse fragment tokens and reject incomplete links', () => {
  assert.deepEqual(
    parseAuthLink('vigorly://auth#access_token=access-2&refresh_token=refresh-2&type=signup'),
    { accessToken: 'access-2', refreshToken: 'refresh-2', type: 'signup' },
  );
  assert.equal(parseAuthLink('vigorly://auth-reset?access_token=missing-refresh'), null);
  assert.equal(parseAuthLink('not a url'), null);
});

test('Supabase sessions map to the app account model', () => {
  const mapped = toAuthSession({
    access_token: 'access',
    refresh_token: 'refresh',
    token_type: 'bearer',
    expires_in: 3600,
    expires_at: 1_800_000_000,
    user: {
      id: 'user-1',
      email: 'person@example.com',
      created_at: '2026-08-02T00:00:00.000Z',
      user_metadata: { display_name: 'Sahil' },
      app_metadata: {},
      aud: 'authenticated',
    },
  } as never);

  assert.equal(mapped.token, 'access');
  assert.equal(mapped.user.id, 'user-1');
  assert.equal(mapped.user.displayName, 'Sahil');
  assert.equal(mapped.user.email, 'person@example.com');
});
