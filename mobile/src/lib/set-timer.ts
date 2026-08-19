import type { ActiveSetTimer } from '@/src/types';

export function startSetTimer(
  sessionExerciseId: string,
  setId: string,
  targetSec: number,
  now = new Date(),
): ActiveSetTimer {
  const safeTarget = Math.max(1, Math.round(targetSec));
  return {
    sessionExerciseId,
    setId,
    targetSec: safeTarget,
    startedAt: now.toISOString(),
    endsAt: new Date(now.getTime() + safeTarget * 1000).toISOString(),
  };
}

export function remainingSetSeconds(timer: ActiveSetTimer, now = new Date()) {
  if (timer.pausedRemainingSec != null) return Math.max(0, timer.pausedRemainingSec);
  return Math.max(0, Math.ceil((new Date(timer.endsAt).getTime() - now.getTime()) / 1000));
}

export function pauseSetTimer(timer: ActiveSetTimer, now = new Date()): ActiveSetTimer {
  return { ...timer, pausedRemainingSec: remainingSetSeconds(timer, now) };
}

export function resumeSetTimer(timer: ActiveSetTimer, now = new Date()): ActiveSetTimer {
  const remaining = Math.max(1, timer.pausedRemainingSec ?? remainingSetSeconds(timer, now));
  return {
    ...timer,
    endsAt: new Date(now.getTime() + remaining * 1000).toISOString(),
    pausedRemainingSec: undefined,
  };
}

export function elapsedSetSeconds(timer: ActiveSetTimer, now = new Date()) {
  return Math.min(timer.targetSec, Math.max(0, timer.targetSec - remainingSetSeconds(timer, now)));
}
