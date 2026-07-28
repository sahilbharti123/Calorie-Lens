import type {
  EstimationContext,
  LogOperation,
  MealItem,
  MealSlot,
  ParsedCommand,
} from '@/src/types';
import { apiRequest } from '@/src/lib/api-client';
import { readSession } from '@/src/lib/session';

type FoodReference = {
  name: string;
  aliases: string[];
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fdcId: number;
  pieceG?: number;
  pieceVariance?: number;
  density?: number;
  densityVariance?: number;
};

const foods: FoodReference[] = [
  { name: 'Cooked kidney beans', aliases: ['kidney beans', 'kidney bean', 'rajma'], calories: 127, protein: 8.67, carbs: 22.8, fat: 0.5, fdcId: 175194, density: 0.75, densityVariance: 0.18 },
  { name: 'Cooked lentils', aliases: ['lentils', 'lentil', 'dhal', 'dal'], calories: 116, protein: 9.02, carbs: 20.1, fat: 0.38, fdcId: 172421, density: 0.75, densityVariance: 0.18 },
  { name: 'Cooked white rice', aliases: ['white rice', 'plain rice', 'cooked rice', 'rice'], calories: 130, protein: 2.69, carbs: 28.2, fat: 0.28, fdcId: 168878, density: 0.79, densityVariance: 0.12 },
  { name: 'Whole-wheat roti', aliases: ['chapati', 'chappati', 'phulka', 'roti'], calories: 299, protein: 7.85, carbs: 46.1, fat: 9.2, fdcId: 174075, pieceG: 40, pieceVariance: 0.2 },
  { name: 'Roasted chicken breast', aliases: ['chicken breast', 'grilled chicken', 'roasted chicken'], calories: 165, protein: 31, carbs: 0, fat: 3.57, fdcId: 171477 },
  { name: 'Boiled egg', aliases: ['hard boiled egg', 'boiled egg', 'egg'], calories: 155, protein: 12.6, carbs: 1.12, fat: 10.6, fdcId: 173424, pieceG: 50, pieceVariance: 0.12 },
  { name: 'Whole milk', aliases: ['whole milk', 'full fat milk', 'milk'], calories: 60, protein: 3.27, carbs: 4.63, fat: 3.2, fdcId: 746782, density: 1.03, densityVariance: 0.03 },
  { name: 'Plain whole-milk yogurt', aliases: ['plain yogurt', 'yogurt', 'curd', 'dahi'], calories: 61, protein: 3.47, carbs: 4.66, fat: 3.25, fdcId: 171284, density: 1.03, densityVariance: 0.06 },
  { name: 'Banana', aliases: ['banana', 'kela'], calories: 89, protein: 1.09, carbs: 22.8, fat: 0.33, fdcId: 173944, pieceG: 118, pieceVariance: 0.18 },
  { name: 'Whole-wheat bread', aliases: ['whole wheat bread', 'brown bread', 'bread', 'toast'], calories: 252, protein: 12.4, carbs: 42.7, fat: 3.5, fdcId: 172688, pieceG: 28, pieceVariance: 0.18 },
  { name: 'Smooth peanut butter', aliases: ['peanut butter'], calories: 598, protein: 22.2, carbs: 22.3, fat: 51.4, fdcId: 172470, density: 1.07, densityVariance: 0.08 },
  { name: 'Idli', aliases: ['idli'], calories: 128, protein: 6.36, carbs: 24.98, fat: 0.35, fdcId: 2708346, pieceG: 50, pieceVariance: 0.22 },
  { name: 'Plain dosa', aliases: ['plain dosa', 'dosa'], calories: 210, protein: 5.7, carbs: 37.04, fat: 4.05, fdcId: 2708347, pieceG: 100, pieceVariance: 0.3 },
];

const numberWords: Record<string, number> = {
  a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, half: 0.5,
};

type Quantity = { amount: number; unit: string };

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function numberValue(value?: string) {
  if (!value) return 0;
  return numberWords[value] ?? Number.parseFloat(value);
}

function quantityNear(text: string, alias: string, reference: FoodReference): Quantity {
  const units = 'kg|g|grams?|ml|l|litres?|liters?|bowls?|cups?|pieces?|tbsp|tsp';
  const number = '\\d+(?:\\.\\d+)?|a|an|one|two|three|four|five|half';
  const explicitVolume = text.match(
    new RegExp(`(\\d+(?:\\.\\d+)?)\\s*ml\\s*(?:bowl(?:\\s+of)?\\s*)?${escapeRegex(alias)}s?\\b`),
  );
  if (explicitVolume) return { amount: numberValue(explicitVolume[1]), unit: 'ml' };
  const before = text.match(new RegExp(`(${number})\\s*(${units})?\\s*${escapeRegex(alias)}s?\\b`));
  const after = text.match(new RegExp(`${escapeRegex(alias)}s?\\s*(${number})\\s*(${units})\\b`));
  const match = before ?? after;
  if (!match) return { amount: 0, unit: 'unknown' };
  const unit = (match[2] ?? (reference.pieceG ? 'piece' : 'unknown'))
    .replace(/s$/, '')
    .replace(/^gram$/, 'g')
    .replace(/^litre$|^liter$/, 'l');
  return { amount: numberValue(match[1]), unit };
}

function firstNumber(text: string, pattern: RegExp) {
  const match = text.match(pattern);
  if (!match) return 0;
  const raw = match.slice(1).find(Boolean)?.replaceAll(',', '');
  return raw ? Number.parseFloat(raw) : 0;
}

function clarification(transcript: string, question: string, suggestions: string[]): ParsedCommand {
  return {
    transcript,
    confirmation: 'One detail will improve this estimate',
    operations: [],
    source: 'local',
    clarification: { question, suggestions },
  };
}

export function inferMealSlot(text = ''): MealSlot {
  const lowered = text.toLowerCase();
  if (/breakfast|morning|nashta/.test(lowered)) return 'breakfast';
  if (/lunch|afternoon/.test(lowered)) return 'lunch';
  if (/snack|evening|chai/.test(lowered)) return 'snack';
  if (/dinner|night/.test(lowered)) return 'dinner';
  const hour = new Date().getHours();
  if (hour < 11) return 'breakfast';
  if (hour < 16) return 'lunch';
  if (hour < 19) return 'snack';
  return 'dinner';
}

function gramsFor(
  quantity: Quantity,
  food: FoodReference,
  context: EstimationContext,
): { grams: number; low: number; high: number; label: string } | { question: string; suggestions: string[] } {
  const amount = quantity.amount;
  const unit = quantity.unit;
  if (!amount) {
    return {
      question: `How much ${food.name.toLowerCase()} did you have?`,
      suggestions: ['100 g', '1 piece', '1 bowl', '1 cup'],
    };
  }
  if (unit === 'kg') return { grams: amount * 1000, low: amount * 980, high: amount * 1020, label: `${amount} kg` };
  if (unit === 'g') return { grams: amount, low: amount * 0.98, high: amount * 1.02, label: `${amount} g` };
  if (unit === 'piece') {
    if (!food.pieceG) return { question: `About how many grams was each ${food.name.toLowerCase()}?`, suggestions: ['30 g', '50 g', '75 g', '100 g'] };
    const grams = amount * food.pieceG;
    const variance = food.pieceVariance ?? 0.12;
    return { grams, low: grams * (1 - variance), high: grams * (1 + variance), label: `${amount} × ${food.pieceG} g standard piece` };
  }
  let volumeMl = 0;
  let label = '';
  if (unit === 'bowl') {
    if (!context.bowlMl) return { question: 'About how large is your usual bowl?', suggestions: ['150 ml', '200 ml', '250 ml', '300 ml'] };
    volumeMl = amount * context.bowlMl;
    label = `${amount} × ${context.bowlMl} ml bowl`;
  } else if (unit === 'cup') {
    volumeMl = amount * context.cupMl;
    label = `${amount} × ${context.cupMl} ml cup`;
  } else if (unit === 'ml') {
    volumeMl = amount;
    label = `${amount} ml`;
  } else if (unit === 'l') {
    volumeMl = amount * 1000;
    label = `${amount} l`;
  } else if (unit === 'tbsp' || unit === 'tsp') {
    volumeMl = amount * (unit === 'tbsp' ? 15 : 5);
    label = `${amount} ${unit}`;
  }
  if (!volumeMl || !food.density) {
    return { question: `Can you give the grams for ${food.name.toLowerCase()}?`, suggestions: ['50 g', '100 g', '150 g', '200 g'] };
  }
  const grams = volumeMl * food.density;
  const variance = food.densityVariance ?? 0.12;
  return { grams, low: grams * (1 - variance), high: grams * (1 + variance), label };
}

function estimateFood(
  food: FoodReference,
  quantity: Quantity,
  text: string,
  context: EstimationContext,
): Omit<MealItem, 'id' | 'slot' | 'loggedAt'> | { question: string; suggestions: string[] } {
  const measured = gramsFor(quantity, food, context);
  if ('question' in measured) return measured;
  const scale = measured.grams / 100;
  let calories = food.calories * scale;
  let low = food.calories * measured.low / 100;
  let high = food.calories * measured.high / 100;
  let fat = food.fat * scale;
  const assumptions: string[] = [];
  if ((/rajma|dal|curry/.test(text)) && !/plain|boiled|dry/.test(text)) {
    const oilKcal = 4.6 * 8.84;
    calories += oilKcal;
    low += oilKcal * 0.5;
    high += oilKcal * 2;
    fat += 4.6;
    assumptions.push('home curry range assumes ½–2 tsp oil in this portion');
  }
  const exactGrams = quantity.unit === 'g' || quantity.unit === 'kg';
  const confidence = measured.label.includes('bowl') || assumptions.length ? 'low' : exactGrams ? 'high' : 'medium';
  return {
    name: food.name,
    quantity: measured.label,
    calories: Math.round(calories),
    calorieLow: Math.round(low),
    calorieHigh: Math.round(high),
    protein: Math.round(food.protein * scale * 10) / 10,
    carbs: Math.round(food.carbs * scale * 10) / 10,
    fat: Math.round(fat * 10) / 10,
    source: 'usda',
    sourceLabel: 'USDA FoodData Central',
    sourceId: `FDC ${food.fdcId}`,
    confidence,
    basis: `${measured.label} · ${food.calories} kcal/100 g${assumptions[0] ? ` · ${assumptions[0]}` : ''}`,
    assumptions,
  };
}

type Activity = { name: string; aliases: RegExp; mets: [number, number, number]; sourceId: string };
const activities: Activity[] = [
  { name: 'Walking', aliases: /walk/, mets: [2.8, 3.8, 4.8], sourceId: 'walking' },
  { name: 'Running', aliases: /run|jog/, mets: [6.5, 8.5, 11], sourceId: 'running' },
  { name: 'Cycling', aliases: /cycl|bike/, mets: [4.3, 7, 9], sourceId: 'bicycling' },
  { name: 'Strength training', aliases: /strength|weight|lifting|gym/, mets: [3.5, 5, 6], sourceId: 'conditioning-exercise' },
  { name: 'HIIT', aliases: /hiit|high intensity interval/, mets: [7, 9, 11], sourceId: 'conditioning-exercise' },
  { name: 'Yoga', aliases: /yoga|vinyasa|hatha/, mets: [2.3, 2.7, 4], sourceId: 'conditioning-exercise' },
];

function estimateWorkout(text: string, context: EstimationContext): LogOperation | ParsedCommand | null {
  const activity = activities.find((candidate) => candidate.aliases.test(text));
  if (!activity) return null;
  const duration = firstNumber(text, /(\d+(?:\.\d+)?)\s*(?:min|minute)/);
  if (!duration) return clarification(text, `How many minutes did you do ${activity.name.toLowerCase()}?`, ['15 min', '30 min', '45 min', '60 min']);
  if (!context.weightKg) return clarification(text, 'What is your current body weight? I need it to estimate active calories.', ['60 kg', '70 kg', '80 kg', '90 kg']);
  const hasIntensity = /hard|intense|vigorous|brisk|moderate|easy|light/.test(text);
  if (!hasIntensity) return clarification(text, `How hard was the ${activity.name.toLowerCase()}?`, ['Light', 'Moderate', 'Hard', 'Give speed']);
  const intensityIndex = /hard|intense|vigorous|brisk/.test(text) ? 2 : /easy|light/.test(text) ? 0 : 1;
  const intensity = (['light', 'moderate', 'hard'] as const)[intensityIndex];
  const met = activity.mets[intensityIndex];
  const metLow = intensityIndex ? activity.mets[intensityIndex - 1] : met * 0.9;
  const metHigh = intensityIndex < 2 ? activity.mets[intensityIndex + 1] : met * 1.1;
  const activeKcal = (value: number) => Math.max(0, value - 1) * 3.5 * context.weightKg! / 200 * duration;
  return {
    type: 'workout',
    action: 'add',
    name: activity.name,
    durationMin: duration,
    calories: Math.round(activeKcal(met)),
    calorieLow: Math.round(activeKcal(metLow)),
    calorieHigh: Math.round(activeKcal(metHigh)),
    intensity,
    met,
    sourceLabel: '2024 Adult Compendium of Physical Activities',
    sourceId: activity.sourceId,
    confidence: 'low',
    basis: `${met} MET · ${context.weightKg} kg · ${duration} min · resting energy excluded`,
  };
}

export function parseCommandLocally(
  text: string,
  preferredSlot: MealSlot | undefined,
  context: EstimationContext,
): ParsedCommand {
  const lowered = text.toLowerCase().trim();
  const operations: LogOperation[] = [];
  const water = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*(?:ml|millilit)/);
  const waterLitres = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*(?:l|litre|liter)\b/);
  const glasses = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*glass/);
  if (/water|paani|pani|glass/.test(lowered)) {
    operations.push({ type: 'water', action: 'add', amount: water || waterLitres * 1000 || glasses * 250 || 250 });
  }
  const steps = firstNumber(lowered, /(\d[\d,]*)\s*steps?/);
  if (steps) operations.push({ type: 'steps', action: 'set', amount: steps });
  const sleep = firstNumber(lowered, /(?:slept|sleep).{0,12}(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:hours?|hrs?).{0,12}sleep/);
  if (/sleep|slept/.test(lowered) && sleep) operations.push({ type: 'sleep', action: 'set', amount: sleep });
  const weight = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*kg/);
  const hasWorkoutLanguage = activities.some((activity) => activity.aliases.test(lowered));
  if ((/weight|weigh/.test(lowered) || (hasWorkoutLanguage && !context.weightKg)) && weight) {
    operations.push({ type: 'weight', action: 'set', amount: weight });
  }

  const effectiveContext = { ...context, weightKg: (context.weightKg ?? weight) || undefined };
  const workout = estimateWorkout(lowered, effectiveContext);
  if (workout && !('type' in workout)) return workout;
  if (workout) operations.push(workout);

  const matches = foods
    .map((food) => ({ food, alias: food.aliases.find((alias) => new RegExp(`\\b${escapeRegex(alias)}s?\\b`).test(lowered)) }))
    .filter((match): match is { food: FoodReference; alias: string } => Boolean(match.alias));
  const looksLikeMeal = /ate|had|breakfast|lunch|dinner|snack|khaya|khayi/.test(lowered) || matches.length > 0;
  if (looksLikeMeal) {
    if (!matches.length) {
      const declaredCalories = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*(?:kcal|calories?)/);
      if (declaredCalories) {
        operations.push({
          type: 'meal',
          action: 'add',
          slot: preferredSlot ?? inferMealSlot(lowered),
          description: text.trim(),
          items: [{
            name: text.replace(/\d+(?:\.\d+)?\s*(?:kcal|calories?)/i, '').trim() || 'Packaged food',
            quantity: 'amount described by user',
            calories: Math.round(declaredCalories),
            calorieLow: Math.round(declaredCalories * 0.95),
            calorieHigh: Math.round(declaredCalories * 1.05),
            protein: 0,
            carbs: 0,
            fat: 0,
            source: 'label',
            sourceLabel: 'Food label supplied by user',
            sourceId: 'declared serving',
            confidence: 'medium',
            basis: `${declaredCalories} kcal declared for the amount consumed`,
            assumptions: ['label rounding and serving accuracy still apply'],
          }],
        });
      } else {
      return clarification(text, 'I do not have a verified reference for that food yet. Can you give its label calories or main ingredients?', ['Read label calories', 'List ingredients', 'Use a different food name']);
      }
    }
    const items: Omit<MealItem, 'id' | 'slot' | 'loggedAt'>[] = [];
    for (const { food, alias } of matches) {
      const estimate = estimateFood(food, quantityNear(lowered, alias, food), lowered, context);
      if ('question' in estimate) return clarification(text, estimate.question, estimate.suggestions);
      items.push(estimate);
    }
    if (items.length) {
      operations.push({
        type: 'meal',
        action: 'add',
        slot: preferredSlot ?? inferMealSlot(lowered),
        description: text.trim(),
        items,
      });
    }
  }
  if (!operations.length) return clarification(text, 'What would you like me to log?', ['A meal', 'Water', 'A workout', 'Weight']);
  return {
    transcript: text.trim(),
    confirmation: operations.length === 1 ? '1 evidence-backed update ready' : `${operations.length} evidence-backed updates ready`,
    operations,
    source: 'local',
  };
}

type ClarificationContext = {
  previousTranscript?: string;
  clarificationQuestion?: string;
};

function requestPayload(
  context: EstimationContext,
  clarification?: ClarificationContext,
) {
  return {
    weight_kg: context.weightKg,
    bowl_ml: context.bowlMl,
    cup_ml: context.cupMl,
    previous_transcript: clarification?.previousTranscript,
    clarification_question: clarification?.clarificationQuestion,
  };
}

export async function parseFitnessCommand(
  text: string,
  preferredSlot: MealSlot | undefined,
  context: EstimationContext,
  clarification?: ClarificationContext,
) {
  if (!process.env.EXPO_PUBLIC_API_URL) return parseCommandLocally(text, preferredSlot, context);
  try {
    const session = await readSession();
    const response = await apiRequest<ParsedCommand>('/v1/parse-command', {
      method: 'POST',
      body: JSON.stringify({
        text,
        preferred_slot: preferredSlot,
        ...requestPayload(context, clarification),
      }),
    }, session);
    return { ...response, source: 'ai' } as ParsedCommand;
  } catch {
    const combined = clarification?.previousTranscript
      ? `${clarification.previousTranscript}. ${text}`
      : text;
    return parseCommandLocally(combined, preferredSlot, context);
  }
}

export async function parseVoiceCommand(
  audioUri: string,
  preferredSlot: MealSlot | undefined,
  context: EstimationContext,
  clarification?: ClarificationContext,
) {
  if (!process.env.EXPO_PUBLIC_API_URL) throw new Error('Set EXPO_PUBLIC_API_URL to enable voice transcription.');
  const body = new FormData();
  body.append('audio', { uri: audioUri, name: 'quick-log.m4a', type: 'audio/mp4' } as unknown as Blob);
  if (preferredSlot) body.append('preferred_slot', preferredSlot);
  if (context.weightKg) body.append('weight_kg', String(context.weightKg));
  if (context.bowlMl) body.append('bowl_ml', String(context.bowlMl));
  body.append('cup_ml', String(context.cupMl));
  if (clarification?.previousTranscript) body.append('previous_transcript', clarification.previousTranscript);
  if (clarification?.clarificationQuestion) body.append('clarification_question', clarification.clarificationQuestion);
  const session = await readSession();
  const response = await apiRequest<ParsedCommand>(
    '/v1/parse-command/audio',
    { method: 'POST', body },
    session,
  );
  return { ...response, source: 'ai' } as ParsedCommand;
}
