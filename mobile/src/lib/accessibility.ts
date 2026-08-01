import { useSyncExternalStore } from 'react';
import { AccessibilityInfo } from 'react-native';

let reducedMotion = true;
let started = false;
const listeners = new Set<() => void>();

function publish(next: boolean) {
  if (next === reducedMotion) return;
  reducedMotion = next;
  for (const listener of listeners) listener();
}

function startListening() {
  if (started) return;
  started = true;
  void AccessibilityInfo.isReduceMotionEnabled().then(publish);
  AccessibilityInfo.addEventListener('reduceMotionChanged', publish);
}

function subscribe(listener: () => void) {
  startListening();
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Mirrors the operating-system Reduce Motion preference. */
export function useReducedMotion() {
  // Default to reduced motion until the asynchronous native preference is
  // known, so an opted-out user never sees an entrance animation start.
  return useSyncExternalStore(subscribe, () => reducedMotion, () => true);
}
