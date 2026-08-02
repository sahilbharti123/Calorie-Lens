import type { PrimaryGoal } from '@/src/types';

const KG_PER_POUND = 0.45359237;
const MIN_WEIGHT_KG = 25;
const MAX_WEIGHT_KG = 350;

export type ParsedWeight = {
  kg: number;
  sourceUnit: 'kg' | 'lb';
  sourceValue: number;
};

function roundToTenth(value: number) {
  return Math.round(value * 10) / 10;
}

/**
 * Parses a scale reading after the surrounding UI has already established
 * that the user is logging weight. This intentionally does not require the
 * recogniser to hear the word "weight" correctly; "current rate is 89" is
 * still recoverable on the dedicated weight screen.
 */
export function parseWeightInput(value: string): ParsedWeight | null {
  const normalized = value.toLowerCase().replace(/(\d),(\d)/g, '$1.$2');
  const number = normalized.match(/\d+(?:\.\d+)?/);
  if (!number) return null;

  const sourceValue = Number.parseFloat(number[0]);
  if (!Number.isFinite(sourceValue)) return null;
  const sourceUnit = /\b(?:lb|lbs|pounds?)\b/.test(normalized) ? 'lb' : 'kg';
  const kg = roundToTenth(sourceUnit === 'lb' ? sourceValue * KG_PER_POUND : sourceValue);
  if (kg < MIN_WEIGHT_KG || kg > MAX_WEIGHT_KG) return null;
  return { kg, sourceUnit, sourceValue };
}

export function targetWeightError(
  goal: PrimaryGoal | null | undefined,
  currentWeightKg: number | undefined,
  targetWeightKg: number | undefined,
) {
  if (!targetWeightKg || targetWeightKg < 35 || targetWeightKg > 250) {
    return 'Enter a target between 35 and 250 kg.';
  }
  if (!currentWeightKg) return null;
  if (goal === 'lose-fat' && targetWeightKg >= currentWeightKg) {
    return 'For fat loss, choose a target below your current weight.';
  }
  if (goal === 'build-muscle' && targetWeightKg <= currentWeightKg) {
    return 'For muscle gain, choose a target above your current weight.';
  }
  if (goal === 'maintain' && Math.abs(targetWeightKg - currentWeightKg) > 1) {
    return 'For maintenance, keep the target close to your current weight or choose a gain/loss goal.';
  }
  return null;
}

export function weightToTargetCopy(currentWeightKg?: number, targetWeightKg?: number) {
  if (!currentWeightKg || !targetWeightKg) return null;
  const difference = roundToTenth(Math.abs(currentWeightKg - targetWeightKg));
  if (difference < 0.1) return 'At target weight';
  return `${difference.toFixed(1)} kg ${currentWeightKg > targetWeightKg ? 'to lose' : 'to gain'} toward target`;
}
