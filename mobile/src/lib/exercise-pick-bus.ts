/**
 * One-shot hand-off between a screen that needs exercises (routine editor,
 * live workout) and the exercise picker modal. Route params can only carry
 * strings, so the requesting screen registers a callback here right before
 * pushing the picker.
 */

type PickHandler = (exerciseIds: string[]) => void;

let handler: PickHandler | null = null;

export function onNextExercisePick(next: PickHandler) {
  handler = next;
}

export function emitExercisePick(exerciseIds: string[]) {
  const current = handler;
  handler = null;
  if (current && exerciseIds.length) current(exerciseIds);
}

export function clearExercisePick() {
  handler = null;
}
