export type AppleWorkoutActivityKey =
  | 'crossTraining'
  | 'running'
  | 'cycling'
  | 'elliptical'
  | 'rowing'
  | 'stairClimbing'
  | 'walking'
  | 'yoga'
  | 'traditionalStrengthTraining';

const CARDIO = /treadmill|\brun(?:ning)?\b|stationary bike|exercise bike|spin|cycling|\bcycle\b|elliptical|cross trainer|rowing|rower|stair|stepmill|\bwalk(?:ing)?\b/i;

/** Maps a session to one honest HealthKit category, including mixed sessions. */
export function classifyAppleWorkout(names: string[]): AppleWorkoutActivityKey {
  const hasCardio = names.some((name) => CARDIO.test(name));
  const hasStrength = names.some((name) => !CARDIO.test(name) && !/\byoga\b/i.test(name));
  const joined = names.join(' ').toLowerCase();
  if (hasCardio && hasStrength) return 'crossTraining';
  if (/treadmill|\brun(?:ning)?\b/.test(joined)) return 'running';
  if (/stationary bike|exercise bike|spin|cycling|\bcycle\b/.test(joined)) return 'cycling';
  if (/elliptical|cross trainer/.test(joined)) return 'elliptical';
  if (/rowing|rower/.test(joined)) return 'rowing';
  if (/stair|stepmill/.test(joined)) return 'stairClimbing';
  if (/\bwalk(?:ing)?\b/.test(joined)) return 'walking';
  if (/\byoga\b/.test(joined)) return 'yoga';
  return 'traditionalStrengthTraining';
}
