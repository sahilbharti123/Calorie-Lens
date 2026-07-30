/**
 * On-device dictation.
 *
 * Speech is transcribed by the operating system's own recogniser (Apple's
 * Speech framework on iOS, the platform recogniser on Android). Nothing is
 * uploaded to a Calorie Lens server and there is no per-use cost, which is why
 * voice logging works without an account, without an API key, and offline
 * wherever the OS has an on-device model installed.
 *
 * The transcript is then handed to the deterministic parser in `nutrition.ts`
 * — the same code path typed input uses. No model ever produces a calorie.
 */

import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
} from 'expo-speech-recognition';
import { useCallback, useEffect, useRef, useState } from 'react';

import { EXERCISES } from '@/src/lib/exercises';
import { FOODS } from '@/src/lib/food-catalog';

/** Longest a single dictation may run before it is closed automatically. */
const MAX_LISTEN_MS = 40_000;
/** Silence after speech has been heard that ends the dictation. */
const SILENCE_MS = 2_600;

export type DictationState = 'idle' | 'listening' | 'finishing';

/**
 * Domain vocabulary handed to the recogniser. Indian dish names and gym
 * movements are exactly the words a general dictation model gets wrong, and
 * iOS weights these strings during recognition.
 */
function buildVocabulary() {
  const words = new Set<string>();
  for (const food of FOODS) {
    words.add(food.name);
    for (const alias of food.aliases) words.add(alias);
  }
  for (const exercise of EXERCISES) {
    words.add(exercise.name);
  }
  for (const unit of ['katori', 'roti', 'chapati', 'paratha', 'grams', 'millilitres', 'scoop', 'bowl']) {
    words.add(unit);
  }
  // iOS ignores very long contextual lists; keep the highest-value terms.
  return [...words].filter((word) => word.length <= 28).slice(0, 300);
}

let cachedVocabulary: string[] | null = null;
function vocabulary() {
  cachedVocabulary ??= buildVocabulary();
  return cachedVocabulary;
}

export function speechAvailable() {
  try {
    return ExpoSpeechRecognitionModule.isRecognitionAvailable();
  } catch {
    return false;
  }
}

export async function ensureSpeechPermission() {
  const current = await ExpoSpeechRecognitionModule.getPermissionsAsync();
  if (current.granted) return { granted: true, canAskAgain: current.canAskAgain };
  const asked = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
  return { granted: asked.granted, canAskAgain: asked.canAskAgain };
}

/**
 * Drives one dictation session.
 *
 * `transcript` holds the best text so far — interim while speaking, final once
 * the recogniser settles — so the UI can show words appearing live.
 */
export function useDictation({ onFinal }: { onFinal?: (transcript: string) => void } = {}) {
  const [state, setState] = useState<DictationState>('idle');
  const [transcript, setTranscript] = useState('');
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const latest = useRef('');
  const heardSpeech = useRef(false);
  const maxTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const silenceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const finalHandler = useRef(onFinal);
  finalHandler.current = onFinal;

  const clearTimers = useCallback(() => {
    if (maxTimer.current) clearTimeout(maxTimer.current);
    if (silenceTimer.current) clearTimeout(silenceTimer.current);
    maxTimer.current = null;
    silenceTimer.current = null;
  }, []);

  const stop = useCallback(() => {
    clearTimers();
    setState((current) => (current === 'listening' ? 'finishing' : current));
    try {
      ExpoSpeechRecognitionModule.stop();
    } catch {
      setState('idle');
    }
  }, [clearTimers]);

  const armSilenceTimer = useCallback(() => {
    if (silenceTimer.current) clearTimeout(silenceTimer.current);
    silenceTimer.current = setTimeout(() => {
      if (heardSpeech.current) stop();
    }, SILENCE_MS);
  }, [stop]);

  useSpeechRecognitionEvent('start', () => {
    setState('listening');
    setError(null);
  });

  useSpeechRecognitionEvent('result', (event) => {
    const best = event.results?.[0]?.transcript?.trim() ?? '';
    if (best) {
      heardSpeech.current = true;
      latest.current = best;
      setTranscript(best);
    }
    if (event.isFinal) {
      clearTimers();
      const finalText = best || latest.current;
      if (finalText) {
        // Cleared first: `end` fires straight after a final result, and it
        // must not hand the same sentence to the parser a second time.
        heardSpeech.current = false;
        finalHandler.current?.(finalText);
        // One tap is one update. Android can deliver a final mid-session, and
        // without this the microphone would stay open with no timers left.
        stop();
      }
    } else {
      armSilenceTimer();
    }
  });

  useSpeechRecognitionEvent('volumechange', (event) => {
    // -2 (silence) … 10 (loud) → 0 … 1
    const normalised = Math.max(0, Math.min(1, (event.value + 2) / 12));
    setLevel(normalised);
  });

  useSpeechRecognitionEvent('error', (event) => {
    clearTimers();
    setState('idle');
    setLevel(0);
    if (event.error === 'no-speech') {
      setError("I didn't catch that. Tap the mic and try again.");
    } else if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
      setError('Microphone or speech access is off. Turn it on in Settings to log by voice.');
    } else if (event.error === 'network') {
      setError('Speech recognition needs a connection right now. Use the keyboard instead.');
    } else {
      setError('Voice input stopped unexpectedly. Try again or use the keyboard.');
    }
  });

  useSpeechRecognitionEvent('end', () => {
    clearTimers();
    setState('idle');
    setLevel(0);
    const finalText = latest.current.trim();
    if (finalText && heardSpeech.current) {
      heardSpeech.current = false;
      finalHandler.current?.(finalText);
    }
  });

  const start = useCallback(async () => {
    setError(null);
    setTranscript('');
    latest.current = '';
    heardSpeech.current = false;

    if (!speechAvailable()) {
      setError('This device cannot transcribe speech. Use the keyboard to log.');
      return false;
    }

    const permission = await ensureSpeechPermission();
    if (!permission.granted) {
      setError(
        permission.canAskAgain
          ? 'Calorie Lens needs microphone and speech access to log by voice.'
          : 'Microphone or speech access is off. Turn it on in Settings to log by voice.',
      );
      return false;
    }

    try {
      ExpoSpeechRecognitionModule.start({
        lang: 'en-US',
        interimResults: true,
        continuous: true,
        addsPunctuation: false,
        contextualStrings: vocabulary(),
        // Off by default in the module, and the level meter needs it.
        volumeChangeEventOptions: { enabled: true, intervalMillis: 100 },
      });
    } catch {
      setError('Voice input could not start. Use the keyboard instead.');
      return false;
    }

    setState('listening');
    maxTimer.current = setTimeout(stop, MAX_LISTEN_MS);
    armSilenceTimer();
    return true;
  }, [armSilenceTimer, stop]);

  const reset = useCallback(() => {
    clearTimers();
    latest.current = '';
    heardSpeech.current = false;
    setTranscript('');
    setError(null);
    setLevel(0);
    setState('idle');
  }, [clearTimers]);

  useEffect(() => () => {
    clearTimers();
    try {
      ExpoSpeechRecognitionModule.abort();
    } catch {
      // The module is already idle.
    }
  }, [clearTimers]);

  return { state, transcript, level, error, start, stop, reset, setError };
}
