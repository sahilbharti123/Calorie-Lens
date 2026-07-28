import { gcm } from '@noble/ciphers/aes.js';
import { bytesToHex, bytesToUtf8, hexToBytes, utf8ToBytes } from '@noble/ciphers/utils.js';
import * as Crypto from 'expo-crypto';
import * as SecureStore from 'expo-secure-store';
import Storage from 'expo-sqlite/kv-store';

const DEVICE_KEY_NAME = 'calorie-lens.device-data-key.v1';
const SECURE_OPTIONS: SecureStore.SecureStoreOptions = {
  keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

async function deviceKey() {
  let encoded = await SecureStore.getItemAsync(DEVICE_KEY_NAME, SECURE_OPTIONS);
  if (!encoded) {
    encoded = bytesToHex(Crypto.getRandomBytes(32));
    await SecureStore.setItemAsync(DEVICE_KEY_NAME, encoded, SECURE_OPTIONS);
  }
  return hexToBytes(encoded);
}

export async function readEncryptedJson<T>(storageKey: string): Promise<T | null> {
  const encoded = await Storage.getItem(storageKey);
  if (!encoded) return null;
  if (!encoded.startsWith('aesgcm1:')) {
    return JSON.parse(encoded) as T;
  }
  const [, nonceHex, ciphertextHex] = encoded.split(':');
  if (!nonceHex || !ciphertextHex) throw new Error('The local fitness vault is damaged.');
  const plaintext = gcm(
    await deviceKey(),
    hexToBytes(nonceHex),
    utf8ToBytes(storageKey),
  ).decrypt(hexToBytes(ciphertextHex));
  return JSON.parse(bytesToUtf8(plaintext)) as T;
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
}

export async function removeEncryptedJson(storageKey: string) {
  await Storage.removeItem(storageKey);
}
