import type {
  AppData,
  DayLog,
  Goals,
  PersonalPlan,
  PersonalProfile,
  PrimaryGoal,
} from '@/src/types';

const activityFactors = {
  'mostly-seated': 1.2,
  'lightly-active': 1.375,
  active: 1.55,
  'very-active': 1.725,
} as const;

const goalNames: Record<PrimaryGoal, string> = {
  'lose-fat': 'Lose fat',
  'build-muscle': 'Build muscle',
  maintain: 'Maintain weight',
  'improve-fitness': 'Improve fitness',
  'build-habits': 'Build consistent habits',
};

const goalAdjustments = {
  'lose-fat': { gentle: -250, steady: -400, ambitious: -550 },
  'build-muscle': { gentle: 150, steady: 250, ambitious: 300 },
  maintain: { gentle: 0, steady: 0, ambitious: 0 },
  'improve-fitness': { gentle: 0, steady: 0, ambitious: 0 },
  'build-habits': { gentle: 0, steady: 0, ambitious: 0 },
} as const;

const proteinFactors: Record<PrimaryGoal, number> = {
  'lose-fat': 1.6,
  'build-muscle': 1.7,
  maintain: 1.2,
  'improve-fitness': 1.4,
  'build-habits': 1,
};

const stepTargets = {
  'mostly-seated': 6000,
  'lightly-active': 7500,
  active: 9000,
  'very-active': 10000,
} as const;

export function goalLabel(goal?: PrimaryGoal) {
  return goal ? goalNames[goal] : 'Build healthier routines';
}

export function hasCalculableProfile(profile: PersonalProfile) {
  return Boolean(
    profile.primaryGoal
    && profile.equationSex
    && profile.age
    && profile.heightCm
    && profile.weightKg
    && profile.activityLevel
    && profile.goalPace
    && profile.workoutPreference
    && profile.trainingDays != null
    && profile.availableMinutes,
  );
}

export function calculatePersonalTargets(profile: PersonalProfile): {
  goals: Goals;
  plan: PersonalPlan;
} {
  const goal = profile.primaryGoal ?? 'build-habits';
  const pace = profile.goalPace ?? 'gentle';
  const activity = profile.activityLevel ?? 'lightly-active';
  const weight = clamp(profile.weightKg ?? 70, 35, 250);
  const height = clamp(profile.heightCm ?? 170, 125, 230);
  const age = clamp(profile.age ?? 30, 18, 90);
  const sexOffset = profile.equationSex === 'male'
    ? 5
    : profile.equationSex === 'female'
      ? -161
      : -78;
  const restingCalories = Math.round((10 * weight) + (6.25 * height) - (5 * age) + sexOffset);
  const maintenanceCalories = roundTo(restingCalories * activityFactors[activity], 25);
  const requestedAdjustment = goalAdjustments[goal][pace];
  const floor = profile.equationSex === 'male' ? 1500 : profile.equationSex === 'female' ? 1200 : 1350;
  const rawCalories = maintenanceCalories + requestedAdjustment;
  const calories = roundTo(Math.max(floor, rawCalories), 50);
  const appliedAdjustment = calories - maintenanceCalories;

  const heightMetres = height / 100;
  const referenceWeight = Math.min(weight, 27.5 * heightMetres * heightMetres);
  const protein = roundTo(clamp(referenceWeight * proteinFactors[goal], 45, 220), 5);
  const fat = roundTo(Math.max(45, (calories * 0.28) / 9), 5);
  const carbs = roundTo(Math.max(100, (calories - (protein * 4) - (fat * 9)) / 4), 5);
  const activityWater = activity === 'active' || activity === 'very-active' ? 300 : 0;
  const waterMl = roundTo(clamp((weight * 32) + activityWater, 1800, 4500), 100);

  let steps: number = stepTargets[activity];
  if (goal === 'improve-fitness' || goal === 'lose-fat') steps += 1000;
  if (profile.workoutPreference === 'walking') steps += 1000;
  if (profile.workoutPreference === 'restarting') steps = Math.min(steps, 6000);
  steps = roundTo(clamp(steps, 5000, 12000), 500);

  const trainingDays = clamp(profile.trainingDays ?? 3, 1, 7);
  const availableMinutes = clamp(profile.availableMinutes ?? 30, 10, 120);
  const weeklyWorkoutMinutes = roundTo(trainingDays * availableMinutes, 5);
  const strengthDays = goal === 'build-muscle'
    ? Math.min(trainingDays, Math.max(2, trainingDays - 1))
    : profile.workoutPreference === 'gym' || profile.workoutPreference === 'mixed'
      ? Math.min(trainingDays, 2)
      : goal === 'improve-fitness'
        ? Math.min(trainingDays, 2)
        : Math.min(trainingDays, 1);

  const warnings: string[] = [];
  if (profile.equationSex === 'neutral') {
    warnings.push('Energy needs use the midpoint of the female and male Mifflin–St Jeor constants.');
  }
  if (calories !== roundTo(rawCalories, 50)) {
    warnings.push(`The requested pace reached the app’s ${floor} kcal safety floor, so the deficit was reduced.`);
  }
  if (pace === 'ambitious' && (goal === 'lose-fat' || goal === 'build-muscle')) {
    warnings.push('Ambitious pace is an estimate. Adjust using your 2–4 week weight and energy trend.');
  }
  const bmi = weight / (heightMetres * heightMetres);
  if (bmi < 18.5 || bmi >= 35) {
    warnings.push('General equations can be less accurate at this body size; consider clinician or dietitian guidance.');
  }

  const direction = appliedAdjustment < 0
    ? `${Math.abs(appliedAdjustment)} kcal below estimated maintenance`
    : appliedAdjustment > 0
      ? `${appliedAdjustment} kcal above estimated maintenance`
      : 'near estimated maintenance';

  return {
    goals: {
      calories,
      protein,
      carbs,
      fat,
      waterMl,
      steps,
      weeklyWorkoutMinutes,
      strengthDays,
    },
    plan: {
      maintenanceCalories,
      calorieAdjustment: appliedAdjustment,
      restingCalories,
      method: 'Mifflin–St Jeor estimate × activity factor',
      summary: `${goalLabel(goal)} · ${calories} kcal (${direction}) · ${protein} g protein`,
      warnings,
      updatedAt: new Date().toISOString(),
    },
  };
}

export function personalDailyNudge(data: AppData, day: DayLog) {
  const totals = day.meals.reduce(
    (sum, meal) => ({
      calories: sum.calories + meal.calories,
      protein: sum.protein + meal.protein,
    }),
    { calories: 0, protein: 0 },
  );
  const profile = data.profile;
  const goal = profile.primaryGoal ?? 'build-habits';
  const calorieGap = data.goals.calories - totals.calories;
  const proteinGap = Math.max(0, data.goals.protein - totals.protein);
  const waterGap = Math.max(0, data.goals.waterMl - day.waterMl);
  const hour = new Date().getHours();

  if (!day.meals.length) {
    const firstAction = goal === 'build-muscle'
      ? `Start with a protein anchor; aim for roughly ${Math.ceil(data.goals.protein / Math.max(3, profile.mealsPerDay ?? 3))} g in your first meal.`
      : goal === 'lose-fat'
        ? 'Log the first meal before changing it. Accurate awareness is more useful than a perfect menu.'
        : 'Log one normal meal. Today’s plan adapts from real data, not an ideal day.';
    return { title: `Today: ${goalLabel(goal).toLowerCase()}`, body: firstAction };
  }
  if (profile.mainChallenge === 'portions') {
    const uncertainMeal = [...day.meals].reverse().find((meal) => meal.confidence === 'low');
    if (uncertainMeal) {
      return {
        title: `Tighten the ${uncertainMeal.name} estimate`,
        body: 'Its range is wide. Add a gram weight, measured bowl volume or recipe detail before changing what you eat.',
      };
    }
  }
  if (profile.mainChallenge === 'cravings' && hour >= 16 && calorieGap > 150) {
    return {
      title: 'Plan the evening before hunger decides',
      body: `${Math.round(calorieGap)} kcal remain. Reserve a filling snack or dinner with protein and fibre instead of trying to avoid food.`,
    };
  }
  if (profile.mainChallenge === 'time' && calorieGap > 0) {
    return {
      title: 'Use the shortest useful option',
      body: `${proteinSuggestion(profile, Math.round(proteinGap))} Choose a version you can assemble in ten minutes.`,
    };
  }
  if (profile.mainChallenge === 'protein' && proteinGap > 20) {
    return {
      title: `${Math.round(proteinGap)} g protein remains`,
      body: proteinSuggestion(profile, Math.round(proteinGap)),
    };
  }
  if (proteinGap > data.goals.protein * 0.35) {
    return {
      title: `${Math.round(proteinGap)} g protein remains`,
      body: proteinSuggestion(profile, Math.round(proteinGap)),
    };
  }
  if (waterGap >= 750) {
    return {
      title: `${(waterGap / 1000).toFixed(1)} L hydration gap`,
      body: 'Add one glass now and another with your next meal. The target is a practical prompt, not a medical prescription.',
    };
  }
  if (day.steps < data.goals.steps * 0.6) {
    const remaining = Math.max(0, data.goals.steps - day.steps);
    return {
      title: `${remaining.toLocaleString()} steps remain`,
      body: profile.workoutPreference === 'restarting'
        ? 'A ten-minute walk is enough for the next win. Build the streak before increasing the target.'
        : 'Break this into two short walks instead of waiting for one long session.',
    };
  }
  return {
    title: calorieGap >= 0 ? `${Math.round(calorieGap)} kcal available` : `${Math.abs(Math.round(calorieGap))} kcal over target`,
    body: goal === 'build-muscle'
      ? 'Keep the next choice protein-led and judge progress from weekly strength and weight trends.'
      : goal === 'lose-fat'
        ? 'One day does not decide fat loss. Keep dinner ordinary and use the weekly trend.'
        : 'Your day is broadly aligned. Keep the next action simple enough to repeat tomorrow.',
  };
}

export function personalOfflineReply(message: string, data: AppData, day: DayLog) {
  const lower = message.toLowerCase();
  const totals = day.meals.reduce(
    (sum, meal) => ({
      calories: sum.calories + meal.calories,
      protein: sum.protein + meal.protein,
    }),
    { calories: 0, protein: 0 },
  );
  const profile = data.profile;
  const goal = profile.primaryGoal ?? 'build-habits';
  const proteinGap = Math.max(0, Math.round(data.goals.protein - totals.protein));
  const calorieGap = Math.round(data.goals.calories - totals.calories);

  if (/protein|meal|eat|food|dinner|lunch|breakfast/.test(lower)) {
    return coachTone(
      profile,
      `${goalLabel(goal)} plan: ${proteinGap} g protein and ${Math.max(0, calorieGap)} kcal remain today. ${proteinSuggestion(profile, proteinGap)}`,
    );
  }
  if (/workout|gym|train|exercise|walk|cardio|strength/.test(lower)) {
    const limitation = profile.injuries.length
      ? ` Keep your noted limitation (${profile.injuries.join(', ')}) in mind and stop if movement causes pain.`
      : '';
    return coachTone(profile, `${workoutSuggestion(profile, data.goals)}${limitation}`);
  }
  if (/calorie|target|deficit|maintenance|goal/.test(lower)) {
    return coachTone(
      profile,
      `Your ${data.goals.calories} kcal target supports “${goalLabel(goal)}.” Estimated maintenance is ${data.plan.maintenanceCalories ?? 'not yet calculated'} kcal; today you have ${calorieGap >= 0 ? `${calorieGap} kcal remaining` : `${Math.abs(calorieGap)} kcal above target`}. Use a 2–4 week trend before changing the plan.`,
    );
  }
  if (/water|hydration/.test(lower)) {
    return coachTone(
      profile,
      `Your starting hydration target is ${(data.goals.waterMl / 1000).toFixed(1)} L. You have logged ${(day.waterMl / 1000).toFixed(1)} L, leaving ${(Math.max(0, data.goals.waterMl - day.waterMl) / 1000).toFixed(1)} L. Increase gradually around heat and training.`,
    );
  }
  const nudge = personalDailyNudge(data, day);
  return coachTone(profile, `${nudge.title}. ${nudge.body}`);
}

function proteinSuggestion(profile: PersonalProfile, proteinGap: number) {
  const amount = Math.min(40, Math.max(20, proteinGap));
  const avoid = profile.allergies.join(' ').toLowerCase();
  const candidates = profile.dietStyle === 'vegan'
    ? ['tofu', 'soy chunks', 'lentils', 'beans', 'pea protein']
    : profile.dietStyle === 'vegetarian' || profile.dietStyle === 'home-indian'
      ? ['dal', 'curd', 'paneer', 'tofu', 'soy chunks', 'whey']
      : ['eggs', 'chicken', 'fish', 'curd', 'paneer', 'tofu', 'whey'];
  const safe = candidates.filter((candidate) => !allergyMatch(avoid, candidate)).slice(0, 4);
  const options = safe.length ? safe.join(', ') : 'a protein that is safe for your allergies';
  return `Build the next meal around ${options}; aim for about ${amount} g protein.`;
}

function workoutSuggestion(profile: PersonalProfile, goals: Goals) {
  const days = profile.trainingDays ?? Math.max(1, goals.strengthDays);
  const minutes = profile.availableMinutes ?? Math.round(goals.weeklyWorkoutMinutes / days);
  if (profile.primaryGoal === 'build-muscle') {
    const detail = profile.experienceLevel === 'new'
      ? 'Use a small exercise list and practise stable technique.'
      : profile.experienceLevel === 'experienced'
        ? 'Track loads or reps and progress only when recovery supports it.'
        : 'Repeat key movements and add reps or load gradually.';
    return `Your plan prioritizes ${goals.strengthDays} strength days. For the next ${minutes}-minute session, train major movement patterns and finish with reps still in reserve. ${detail}`;
  }
  if (profile.workoutPreference === 'walking' || profile.workoutPreference === 'restarting') {
    return `Use ${days} manageable ${minutes}-minute movement sessions this week. A walk counts; increase duration only after the routine feels repeatable.`;
  }
  return `Aim for ${goals.weeklyWorkoutMinutes} total minutes across ${days} days, including ${goals.strengthDays} strength ${goals.strengthDays === 1 ? 'day' : 'days'}. Choose ${profile.workoutPreference ?? 'familiar'} sessions you can repeat.`;
}

function coachTone(profile: PersonalProfile, message: string) {
  if (profile.coachingTone === 'gentle') return `You do not need a perfect day. ${message}`;
  if (profile.coachingTone === 'direct') return `Next action: ${message}`;
  return message;
}

function allergyMatch(avoid: string, candidate: string) {
  if (!avoid) return false;
  const groups: Record<string, string[]> = {
    curd: ['dairy', 'milk', 'lactose', 'curd'],
    paneer: ['dairy', 'milk', 'paneer'],
    whey: ['dairy', 'milk', 'whey'],
    tofu: ['soy', 'tofu'],
    'soy chunks': ['soy'],
    eggs: ['egg'],
    fish: ['fish', 'seafood'],
  };
  return (groups[candidate] ?? [candidate]).some((term) => avoid.includes(term));
}

function roundTo(value: number, increment: number) {
  return Math.round(value / increment) * increment;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
