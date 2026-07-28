import type { LogOperation, MealItem, MealSlot, ParsedCommand } from '@/src/types';

type FoodReference = {
  name: string;
  aliases: string[];
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  quantity: string;
};

const foodReferences: FoodReference[] = [
  { name: 'Roti', aliases: ['roti', 'chapati', 'phulka'], calories: 120, protein: 3.5, carbs: 18, fat: 3, quantity: '1 piece' },
  { name: 'Cooked rice', aliases: ['rice', 'plain rice'], calories: 205, protein: 4.3, carbs: 45, fat: 0.4, quantity: '1 cup' },
  { name: 'Dal', aliases: ['dal', 'dhal', 'lentils'], calories: 180, protein: 9, carbs: 24, fat: 4, quantity: '1 bowl' },
  { name: 'Paneer', aliases: ['paneer', 'paneer bhurji'], calories: 265, protein: 18, carbs: 6, fat: 20, quantity: '100 g' },
  { name: 'Chicken curry', aliases: ['chicken curry', 'butter chicken'], calories: 260, protein: 24, carbs: 8, fat: 14, quantity: '1 bowl' },
  { name: 'Chicken breast', aliases: ['chicken breast', 'grilled chicken'], calories: 165, protein: 31, carbs: 0, fat: 3.6, quantity: '100 g' },
  { name: 'Egg', aliases: ['egg', 'boiled egg'], calories: 78, protein: 6.3, carbs: 0.6, fat: 5.3, quantity: '1 egg' },
  { name: 'Omelette', aliases: ['omelette', 'omelet'], calories: 154, protein: 11, carbs: 2, fat: 11, quantity: '2 eggs' },
  { name: 'Idli', aliases: ['idli', 'idlis'], calories: 58, protein: 2, carbs: 12, fat: 0.4, quantity: '1 piece' },
  { name: 'Dosa', aliases: ['dosa', 'masala dosa'], calories: 170, protein: 4, carbs: 25, fat: 5, quantity: '1 medium' },
  { name: 'Poha', aliases: ['poha'], calories: 210, protein: 5, carbs: 34, fat: 6, quantity: '1 bowl' },
  { name: 'Upma', aliases: ['upma'], calories: 220, protein: 5, carbs: 32, fat: 8, quantity: '1 bowl' },
  { name: 'Bread', aliases: ['bread', 'toast'], calories: 80, protein: 3, carbs: 15, fat: 1, quantity: '1 slice' },
  { name: 'Peanut butter', aliases: ['peanut butter'], calories: 95, protein: 4, carbs: 3, fat: 8, quantity: '1 tbsp' },
  { name: 'Banana', aliases: ['banana'], calories: 105, protein: 1.3, carbs: 27, fat: 0.4, quantity: '1 medium' },
  { name: 'Apple', aliases: ['apple'], calories: 95, protein: 0.5, carbs: 25, fat: 0.3, quantity: '1 medium' },
  { name: 'Curd', aliases: ['curd', 'yogurt', 'dahi'], calories: 98, protein: 5, carbs: 7, fat: 5, quantity: '1 cup' },
  { name: 'Milk', aliases: ['milk'], calories: 120, protein: 6, carbs: 12, fat: 5, quantity: '250 ml' },
  { name: 'Chai', aliases: ['tea', 'chai'], calories: 70, protein: 2, carbs: 9, fat: 3, quantity: '1 cup' },
  { name: 'Coffee', aliases: ['coffee'], calories: 60, protein: 2, carbs: 7, fat: 2, quantity: '1 cup' },
  { name: 'Samosa', aliases: ['samosa'], calories: 250, protein: 4, carbs: 30, fat: 12, quantity: '1 piece' },
  { name: 'Salad', aliases: ['salad'], calories: 50, protein: 2, carbs: 10, fat: 0.5, quantity: '1 bowl' },
  { name: 'Protein shake', aliases: ['protein shake', 'whey shake', 'whey'], calories: 130, protein: 24, carbs: 4, fat: 2, quantity: '1 scoop' },
];

const numberWords: Record<string, number> = {
  a: 1,
  an: 1,
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  half: 0.5,
};

function quantityMultiplier(text: string, reference: FoodReference) {
  const alias = reference.aliases.find((candidate) => text.includes(candidate));
  if (!alias) return 1;
  const before = text.slice(0, text.indexOf(alias)).trim().split(/\s+/).at(-1) ?? '';
  const numeric = Number.parseFloat(before);
  if (Number.isFinite(numeric)) return numeric;
  return numberWords[before] ?? 1;
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

export function estimateMealLocally(text: string) {
  const lowered = text.toLowerCase();
  const matches = foodReferences.filter((food) => food.aliases.some((alias) => lowered.includes(alias)));
  const selected: FoodReference[] = matches.length ? matches : [{
    name: text.trim() || 'Meal',
    aliases: [],
    calories: 300,
    protein: 12,
    carbs: 35,
    fat: 12,
    quantity: 'estimated serving',
  }];

  return selected.map((food): Omit<MealItem, 'id' | 'slot' | 'loggedAt'> => {
    const multiplier = quantityMultiplier(lowered, food);
    return {
      name: food.name,
      quantity: multiplier === 1 ? food.quantity : `${multiplier} × ${food.quantity}`,
      calories: Math.round(food.calories * multiplier),
      protein: Math.round(food.protein * multiplier * 10) / 10,
      carbs: Math.round(food.carbs * multiplier * 10) / 10,
      fat: Math.round(food.fat * multiplier * 10) / 10,
      source: 'local',
    };
  });
}

function firstNumber(text: string, pattern: RegExp) {
  const match = text.match(pattern);
  if (!match) return 0;
  const raw = match.slice(1).find(Boolean)?.replaceAll(',', '');
  return raw ? Number.parseFloat(raw) : 0;
}

export function parseCommandLocally(text: string, preferredSlot?: MealSlot): ParsedCommand {
  const lowered = text.toLowerCase().trim();
  const operations: LogOperation[] = [];
  const water = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*(?:ml|millilit)/);
  const waterLitres = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*(?:l|litre|liter)\b/);
  const glasses = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*glass/);
  if (/water|paani|pani|glass/.test(lowered)) {
    operations.push({
      type: 'water',
      action: 'add',
      amount: water || waterLitres * 1000 || glasses * 250 || 250,
    });
  }

  const steps = firstNumber(lowered, /(\d[\d,]*)\s*steps?/);
  if (steps) operations.push({ type: 'steps', action: 'set', amount: steps });

  const sleep = firstNumber(lowered, /(?:slept|sleep).{0,12}(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:hours?|hrs?).{0,12}sleep/);
  if (/sleep|slept/.test(lowered) && sleep) operations.push({ type: 'sleep', action: 'set', amount: sleep });

  const weight = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*kg/);
  if (/weight|weigh/.test(lowered) && weight) operations.push({ type: 'weight', action: 'set', amount: weight });

  const duration = firstNumber(lowered, /(\d+(?:\.\d+)?)\s*(?:min|minute)/);
  const workoutMatch = lowered.match(/(?:did|workout|trained|exercise|walked|ran|cycled)\s+(.+?)(?:\s+for\s+\d+|\s+\d+\s*(?:min|minute)|$)/);
  if (workoutMatch || /workout|gym|walk|run|cycling|yoga|strength/.test(lowered)) {
    const name = workoutMatch?.[1]?.trim()
      || (lowered.includes('walk') ? 'Walk' : lowered.includes('run') ? 'Run' : 'Workout');
    const minutes = duration || 30;
    operations.push({
      type: 'workout',
      action: 'add',
      name: name.charAt(0).toUpperCase() + name.slice(1),
      durationMin: minutes,
      calories: Math.round(minutes * 6),
      intensity: /hard|intense|hiit/.test(lowered) ? 'hard' : /easy|light/.test(lowered) ? 'light' : 'moderate',
    });
  }

  const looksLikeMeal = /ate|had|breakfast|lunch|dinner|snack|khaya|khayi/.test(lowered)
    || foodReferences.some((food) => food.aliases.some((alias) => lowered.includes(alias)));
  if (looksLikeMeal) {
    const slot = preferredSlot ?? inferMealSlot(lowered);
    operations.push({
      type: 'meal',
      action: 'add',
      slot,
      description: text.trim(),
      items: estimateMealLocally(text),
    });
  }

  if (!operations.length) {
    const slot = preferredSlot ?? inferMealSlot(lowered);
    operations.push({
      type: 'meal',
      action: 'add',
      slot,
      description: text.trim(),
      items: estimateMealLocally(text),
    });
  }

  return {
    transcript: text.trim(),
    confirmation: operations.length === 1 ? '1 update ready to review' : `${operations.length} updates ready to review`,
    operations,
    source: 'local',
  };
}

export async function parseFitnessCommand(text: string, preferredSlot?: MealSlot) {
  const apiUrl = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');
  if (!apiUrl) return parseCommandLocally(text, preferredSlot);

  try {
    const response = await fetch(`${apiUrl}/v1/parse-command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, preferred_slot: preferredSlot }),
    });
    if (!response.ok) throw new Error(`Request failed with ${response.status}`);
    return { ...(await response.json()), source: 'ai' } as ParsedCommand;
  } catch {
    return parseCommandLocally(text, preferredSlot);
  }
}

export async function parseVoiceCommand(audioUri: string, preferredSlot?: MealSlot) {
  const apiUrl = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');
  if (!apiUrl) {
    throw new Error('Set EXPO_PUBLIC_API_URL to enable voice transcription. Typed logging already works offline.');
  }
  const body = new FormData();
  body.append('audio', {
    uri: audioUri,
    name: 'quick-log.m4a',
    type: 'audio/mp4',
  } as unknown as Blob);
  if (preferredSlot) body.append('preferred_slot', preferredSlot);
  const response = await fetch(`${apiUrl}/v1/parse-command/audio`, { method: 'POST', body });
  if (!response.ok) throw new Error('Voice could not be understood. Please try again or type the update.');
  return { ...(await response.json()), source: 'ai' } as ParsedCommand;
}
