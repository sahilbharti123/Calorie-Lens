import { dateKey } from '@/src/lib/date';
import type { AppData } from '@/src/types';

/** A day counts toward the streak once anything real is logged on it. */
function isActive(data: AppData, key: string) {
  const day = data.days[key];
  if (!day) return false;
  return day.meals.length > 0 || day.workouts.length > 0 || day.steps > 0 || day.waterMl > 0;
}

/** Everything eaten on a day. Zero for a day that has no meals (or no record). */
function dayCalories(data: AppData, key: string) {
  const day = data.days[key];
  if (!day) return 0;
  return day.meals.reduce((sum, meal) => sum + meal.calories, 0);
}

/** `YYYY-MM-DD` → a local `Date`. Built from parts so it does not parse as UTC. */
function fromKey(key: string) {
  const [year, month, day] = key.split('-').map(Number);
  return new Date(year, month - 1, day);
}

/** True when `key` is the calendar day immediately after `previous`. */
function isNextDay(previous: string, key: string) {
  const date = fromKey(previous);
  date.setDate(date.getDate() + 1);
  return dateKey(date) === key;
}

/**
 * How a day's intake sat against the calorie target.
 *
 * `none` is "nothing eaten", not "nothing logged" — a day with only a walk on it
 * still keeps the streak alive but has no intake to judge.
 */
export type IntakeStatus = 'none' | 'on' | 'over';

/** Past this multiple of the target a day reads as genuinely over, not rounding. */
const OVER_TARGET = 1.15;

function intakeStatus(calories: number, target: number): IntakeStatus {
  if (calories <= 0) return 'none';
  if (target > 0 && calories > target * OVER_TARGET) return 'over';
  return 'on';
}

/**
 * Consecutive logged days ending today. Today not being logged yet does not
 * break the streak — the run is measured from yesterday so the number does not
 * reset to zero every midnight.
 */
export function loggingStreak(data: AppData) {
  const cursor = new Date();
  let streak = 0;

  if (isActive(data, dateKey(cursor))) streak += 1;
  cursor.setDate(cursor.getDate() - 1);

  while (isActive(data, dateKey(cursor))) {
    streak += 1;
    cursor.setDate(cursor.getDate() - 1);
    if (streak > 3650) break;
  }

  return streak;
}

/**
 * The longest run of consecutive logged days on record, including the run in
 * progress — so a current streak that ties the record reads as a personal best
 * rather than as one day short of one.
 */
export function bestStreak(data: AppData) {
  const keys = Object.keys(data.days)
    .filter((key) => isActive(data, key))
    .sort();

  let best = 0;
  let run = 0;
  let previous = '';

  for (const key of keys) {
    run = previous && isNextDay(previous, key) ? run + 1 : 1;
    if (run > best) best = run;
    previous = key;
  }

  return best;
}

export type WeekDay = {
  key: string;
  label: string;
  /** Anything at all logged — this is what the streak counts. */
  active: boolean;
  isToday: boolean;
  /** Calories eaten that day. */
  calories: number;
  /** That day's intake judged against the target. */
  status: IntakeStatus;
};

/**
 * The last seven days, oldest first, each with the intake that landed on it.
 *
 * Every day is judged against the *current* target because that is the only one
 * stored — the strip is a shape-of-the-week read, not an audit trail.
 */
export function weekActivity(data: AppData): WeekDay[] {
  const days: WeekDay[] = [];
  const todayKey = dateKey();
  const target = data.goals.calories;

  for (let offset = 6; offset >= 0; offset -= 1) {
    const date = new Date();
    date.setDate(date.getDate() - offset);
    const key = dateKey(date);
    const calories = dayCalories(data, key);
    days.push({
      key,
      label: new Intl.DateTimeFormat(undefined, { weekday: 'narrow' }).format(date),
      active: isActive(data, key),
      isToday: key === todayKey,
      calories,
      status: intakeStatus(calories, target),
    });
  }

  return days;
}
