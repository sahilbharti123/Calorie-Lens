import { gcm } from '@noble/ciphers/aes.js';
import { bytesToHex, bytesToUtf8, hexToBytes, utf8ToBytes } from '@noble/ciphers/utils.js';
import * as Crypto from 'expo-crypto';
import * as SecureStore from 'expo-secure-store';
import Storage from 'expo-sqlite/kv-store';

/**
 * Persisted-storage keys keep the pre-rename `calorie-lens.` prefix on purpose.
 * They identify data already written to the device — renaming them would
 * orphan every existing vault, session and backup rather than migrate it.
 * The name a user sees is set in app.json; these are invisible.
 */
const DEVICE_KEY_NAME = 'calorie-lens.device-data-key.v1';
const SECURE_OPTIONS: SecureStore.SecureStoreOptions = {
  keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

/**
 * Records this launch found but could not decrypt. Read once during hydration
 * so the app can tell the user their local copy was reset rather than silently
 * appearing to have lost their history.
 */
const unreadable = new Set<string>();

async function deviceKey() {
  let encoded = await SecureStore.getItemAsync(DEVICE_KEY_NAME, SECURE_OPTIONS);
  if (!encoded) {
    encoded = bytesToHex(Crypto.getRandomBytes(32));
    await SecureStore.setItemAsync(DEVICE_KEY_NAME, encoded, SECURE_OPTIONS);
  }
  return hexToBytes(encoded);
}

/** Drops a record this device can no longer read, so it cannot fail again. */
async function discard(storageKey: string) {
  unreadable.add(storageKey);
  try {
    await Storage.removeItem(storageKey);
  } catch {
    // Leaving the record in place is survivable — the next read discards again.
  }
}

/**
 * Reads and decrypts a stored record, or returns null.
 *
 * This never throws on unreadable data, and that is deliberate. The device key
 * is stored `WHEN_UNLOCKED_THIS_DEVICE_ONLY`, which keeps it out of iCloud and
 * iTunes backups on purpose — but the SQLite file holding the ciphertext *is*
 * backed up. Restoring a backup onto a new phone therefore hands this device
 * ciphertext it can never decrypt, and the same happens whenever the keychain
 * item is cleared while the database survives.
 *
 * That is an expected state, not corruption. Throwing turned it into an
 * unhandled rejection that stopped the app from finishing hydration, so instead
 * the record is discarded and reported through `takeUnreadableRecords()`.
 */
export async function readEncryptedJson<T>(storageKey: string): Promise<T | null> {
  const encoded = await Storage.getItem(storageKey);
  if (!encoded) return null;

  if (!encoded.startsWith('aesgcm1:')) {
    // A pre-encryption record from an older build.
    try {
      return JSON.parse(encoded) as T;
    } catch {
      await discard(storageKey);
      return null;
    }
  }

  const [, nonceHex, ciphertextHex] = encoded.split(':');
  if (!nonceHex || !ciphertextHex) {
    await discard(storageKey);
    return null;
  }

  try {
    const plaintext = gcm(
      await deviceKey(),
      hexToBytes(nonceHex),
      utf8ToBytes(storageKey),
    ).decrypt(hexToBytes(ciphertextHex));
    return JSON.parse(bytesToUtf8(plaintext)) as T;
  } catch {
    // Wrong key (restored backup, cleared keychain) or a truncated write.
    await discard(storageKey);
    return null;
  }
}

export async function writeEncryptedJson(storageKey: string, value: unknown) {
  const nonce = Crypto.getRandomBytes(12);
  const ciphertext = gcm(
    await deviceKey(),
    nonce,
    utf8ToBytes(storageKey),
  ).encrypt(utf8ToBytes(JSON.stringify(value)));
  await Storage.setItem(
    storageKey,
    `aesgcm1:${bytesToHex(nonce)}:${bytesToHex(ciphertext)}`,
  );
  unreadable.delete(storageKey);
}

export async function removeEncryptedJson(storageKey: string) {
  unreadable.delete(storageKey);
  await Storage.removeItem(storageKey);
}

/**
 * Returns the records discarded because this device could not decrypt them,
 * and clears the list. Call after hydration to decide whether to tell the user.
 */
export function takeUnreadableRecords() {
  const keys = [...unreadable];
  unreadable.clear();
  return keys;
}
