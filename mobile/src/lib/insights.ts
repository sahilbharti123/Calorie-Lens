/**
 * On-device coaching insights.
 *
 * Everything here is derived arithmetically from the user's own logs — no model
 * is involved and nothing leaves the phone. Each insight states the number it
 * is based on so the advice can be checked rather than trusted.
 */

import { dateKey } from '@/src/lib/date';
import { dayTotals } from '@/src/lib/stats';
import { loggingStreak } from '@/src/lib/streak';
import { weeklyTrainingSummary } from '@/src/lib/training';
import type { AppData, DayLog } from '@/src/types';

/** Where an insight can send the user, when there is something to do about it. */
export type InsightAction = {
  label: string;
  route: '/quick-log' | '/settings';
};

export type Insight = {
  id: string;
  icon: 'flame' | 'target' | 'muscle' | 'water' | 'timer' | 'trend' | 'scale' | 'bowl';
  tone: 'good' | 'watch' | 'neutral';
  title: string;
  /** One line, short enough to sit in a collapsed row under the title. */
  summary: string;
  /** The full reading, shown when the row is opened. */
  body: string;
  /** Higher sorts first; the top insight is the one a screen should lead with. */
  priority: number;
  /** Present only where the user can act on the insight immediately. */
  action?: InsightAction;
};

/**
 * Ranking, highest first. Something drifting outranks something holding, and a
 * pattern that can be acted on today outranks a slow trend — so whatever leads
 * the screen is the most useful thing these numbers know.
 */
const PRIORITY = {
  proteinLow: 90,
  calorieDrift: 80,
  /** Above the weekly readings: a blank day is the one thing fixable right now. */
  nothingLoggedToday: 75,
  trainingBehind: 70,
  estimatesWide: 60,
  hydrationLow: 50,
  weightTrend: 30,
  calorieSteady: 22,
  proteinGood: 20,
  trainingOnPlan: 18,
  streak: 10,
} as const;

/** One-decimal figure — litres and kilograms — grouped for the device locale. */
function oneDecimal(value: number) {
  return value.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

function recentDays(data: AppData, count: number) {
  const days: DayLog[] = [];
  for (let offset = 0; offset < count; offset += 1) {
    const date = new Date();
    date.setDate(date.getDate() - offset);
    const day = data.days[dateKey(date)];
    if (day) days.push(day);
  }
  return days;
}

/** Days with at least one meal — the only days an intake average is honest about. */
function loggedDays(data: AppData, count: number) {
  return recentDays(data, count).filter((day) => day.meals.length > 0);
}

/** How many of the last `count` days carry at least one meal. */
export function loggedDayCount(data: AppData, count = 7) {
  return loggedDays(data, count).length;
}

export function buildInsights(data: AppData, today: DayLog): Insight[] {
  const insights: Insight[] = [];
  const week = loggedDays(data, 7);
  const streak = loggingStreak(data);
  const todayTotals = dayTotals(today);

  /* ---- protein adherence, the number that moves body composition most ---- */
  if (week.length >= 3 && data.goals.protein > 0) {
    const averageProtein =
      week.reduce((sum, day) => sum + dayTotals(day).protein, 0) / week.length;
    const share = averageProtein / data.goals.protein;
    if (share < 0.8) {
      insights.push({
        id: 'protein-low',
        icon: 'muscle',
        tone: 'watch',
        priority: PRIORITY.proteinLow,
        title: 'Protein is running short',
        summary: `${Math.round(averageProtein).toLocaleString()} g a day against a ${data.goals.protein.toLocaleString()} g target`,
        body: `You have averaged ${Math.round(averageProtein).toLocaleString()} g a day over ${week.length.toLocaleString()} logged days against a ${data.goals.protein.toLocaleString()} g target. Adding one protein-led item to the meal you find easiest is usually enough to close it.`,
        action: { label: 'Log a protein-led item', route: '/quick-log' },
      });
    } else {
      insights.push({
        id: 'protein-good',
        icon: 'muscle',
        tone: 'good',
        priority: PRIORITY.proteinGood,
        title: 'Protein is holding',
        summary: `${Math.round(averageProtein).toLocaleString()} g a day against a ${data.goals.protein.toLocaleString()} g target`,
        body: `${Math.round(averageProtein).toLocaleString()} g a day across ${week.length.toLocaleString()} logged days, against a ${data.goals.protein.toLocaleString()} g target. This is the habit worth protecting when the week gets busy.`,
      });
    }
  }

  /* ---- calorie adherence ---- */
  if (week.length >= 3 && data.goals.calories > 0) {
    const averageCalories =
      week.reduce((sum, day) => sum + dayTotals(day).calories, 0) / week.length;
    const gap = averageCalories - data.goals.calories;
    const drift = Math.abs(gap) / data.goals.calories;
    if (drift > 0.12) {
      insights.push({
        id: 'calorie-drift',
        icon: 'target',
        tone: 'watch',
        priority: PRIORITY.calorieDrift,
        title: gap > 0 ? 'Intake is above the plan' : 'Intake is below the plan',
        summary: `${Math.round(averageCalories).toLocaleString()} kcal a day, ${Math.abs(Math.round(gap)).toLocaleString()} kcal ${gap > 0 ? 'above' : 'below'} target`,
        body: `Your logged average is ${Math.round(averageCalories).toLocaleString()} kcal against a ${data.goals.calories.toLocaleString()} kcal target — a gap of ${Math.abs(Math.round(gap)).toLocaleString()} kcal a day. Judge this on a two-to-four week trend before changing the target, and check that low-confidence entries are not skewing it.`,
        action: { label: 'Review your targets', route: '/settings' },
      });
    } else {
      insights.push({
        id: 'calorie-steady',
        icon: 'target',
        tone: 'good',
        priority: PRIORITY.calorieSteady,
        title: 'Intake is tracking the plan',
        summary: `${Math.round(averageCalories).toLocaleString()} kcal a day against a ${data.goals.calories.toLocaleString()} kcal target`,
        body: `${Math.round(averageCalories).toLocaleString()} kcal a day against a ${data.goals.calories.toLocaleString()} kcal target across ${week.length.toLocaleString()} logged days. Consistency at this level is what makes the weight trend readable.`,
      });
    }
  }

  /* ---- training volume against the plan ----
   * Read through the same helper the Train tab uses, from the same training
   * store, so the two screens cannot state different figures for one week. */
  const training = weeklyTrainingSummary(data.training);
  if (data.goals.weeklyWorkoutMinutes > 0) {
    const share = training.minutes / data.goals.weeklyWorkoutMinutes;
    const onPlan = share >= 0.8;
    const volume = `${training.minutes.toLocaleString()} of ${data.goals.weeklyWorkoutMinutes.toLocaleString()} planned minutes`;
    insights.push({
      id: 'training',
      icon: 'timer',
      tone: onPlan ? 'good' : 'watch',
      priority: onPlan ? PRIORITY.trainingOnPlan : PRIORITY.trainingBehind,
      title: onPlan ? 'Training volume is on plan' : 'Training is behind plan',
      summary: volume,
      body: `${volume} over the last seven days, across ${training.workouts.toLocaleString()} workout${training.workouts === 1 ? '' : 's'} finished in Train. ${
        onPlan
          ? 'Hold this rather than adding volume — recoverable consistency beats a heavy week followed by none.'
          : 'A shorter session you actually do is worth more than a full one you skip.'
      }`,
    });
  }

  /* ---- hydration, only when it is actually being tracked ---- */
  const hydrationDays = recentDays(data, 7).filter((day) => day.waterMl > 0);
  if (hydrationDays.length >= 3 && data.goals.waterMl > 0) {
    const averageWater =
      hydrationDays.reduce((sum, day) => sum + day.waterMl, 0) / hydrationDays.length;
    if (averageWater < data.goals.waterMl * 0.75) {
      insights.push({
        id: 'water',
        icon: 'water',
        tone: 'watch',
        priority: PRIORITY.hydrationLow,
        title: 'Hydration is trailing',
        summary: `About ${oneDecimal(averageWater / 1000)} L a day against a ${oneDecimal(data.goals.waterMl / 1000)} L target`,
        body: `About ${oneDecimal(averageWater / 1000)} L a day against a ${oneDecimal(data.goals.waterMl / 1000)} L target. Tying a glass to something you already do every day works better than remembering to drink.`,
        action: { label: 'Log a glass', route: '/quick-log' },
      });
    }
  }

  /* ---- estimate quality: the app's own honesty check ---- */
  const recentMeals = recentDays(data, 7).flatMap((day) => day.meals);
  const lowConfidence = recentMeals.filter((meal) => meal.confidence === 'low').length;
  if (recentMeals.length >= 5 && lowConfidence / recentMeals.length > 0.4) {
    insights.push({
      id: 'confidence',
      icon: 'bowl',
      tone: 'watch',
      priority: PRIORITY.estimatesWide,
      title: 'Your estimates are wide',
      summary: `${lowConfidence.toLocaleString()} of your last ${recentMeals.length.toLocaleString()} entries were low confidence`,
      body: `${lowConfidence.toLocaleString()} of your last ${recentMeals.length.toLocaleString()} entries were low confidence — usually a bowl or a home recipe with unknown oil. Saying a gram weight or a package serving for even one meal a day narrows the whole picture.`,
      action: { label: 'Log with a weight', route: '/quick-log' },
    });
  }

  /* ---- weight trend ---- */
  if (data.weights.length >= 4) {
    const sorted = [...data.weights].sort((a, b) => a.date.localeCompare(b.date));
    const latest = sorted.at(-1);
    const earliest = sorted[Math.max(0, sorted.length - 8)];
    if (latest && earliest && latest.date !== earliest.date) {
      const change = latest.kg - earliest.kg;
      const movement = `${change > 0 ? '+' : ''}${oneDecimal(change)} kg across your last ${Math.min(8, sorted.length).toLocaleString()} weigh-ins`;
      insights.push({
        id: 'weight',
        icon: 'scale',
        tone: 'neutral',
        priority: PRIORITY.weightTrend,
        title: Math.abs(change) < 0.3 ? 'Weight is stable' : change > 0 ? 'Weight is trending up' : 'Weight is trending down',
        summary: movement,
        body: `${movement}. Day-to-day swings are mostly water and gut content — only the direction over weeks is signal.`,
      });
    }
  }

  /* ---- streak, or the nudge to start one ---- */
  if (streak >= 3) {
    insights.push({
      id: 'streak',
      icon: 'flame',
      tone: 'good',
      priority: PRIORITY.streak,
      title: `${streak.toLocaleString()} days logged in a row`,
      summary: 'An over-target day logged still beats a blank one',
      body: 'The streak matters more than any single day being perfect. Logging an over-target day is worth more than logging nothing.',
    });
  } else if (todayTotals.calories === 0) {
    insights.push({
      id: 'start',
      icon: 'flame',
      tone: 'neutral',
      priority: PRIORITY.nothingLoggedToday,
      title: 'Nothing logged yet today',
      summary: 'One entry keeps the picture continuous',
      body: 'One entry is enough to keep the picture continuous. Log the meal you remember most clearly rather than trying to reconstruct the whole day.',
      action: { label: 'Log something', route: '/quick-log' },
    });
  }

  return insights.sort((a, b) => b.priority - a.priority);
}
