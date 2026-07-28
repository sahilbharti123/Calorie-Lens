export type MealSlot = 'breakfast' | 'lunch' | 'snack' | 'dinner';
export type EstimateConfidence = 'high' | 'medium' | 'low';
export type EstimationContext = {
  weightKg?: number;
  bowlMl?: number;
  cupMl: number;
};

export type MealItem = {
  id: string;
  name: string;
  quantity: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  calorieLow?: number;
  calorieHigh?: number;
  confidence?: EstimateConfidence;
  basis?: string;
  sourceLabel?: string;
  sourceId?: string;
  assumptions?: string[];
  slot: MealSlot;
  loggedAt: string;
  source: 'usda' | 'label' | 'ai' | 'local' | 'manual';
};

export type Workout = {
  id: string;
  name: string;
  durationMin: number;
  calories: number;
  calorieLow?: number;
  calorieHigh?: number;
  confidence?: EstimateConfidence;
  basis?: string;
  sourceLabel?: string;
  sourceId?: string;
  met?: number;
  intensity: 'light' | 'moderate' | 'hard';
  loggedAt: string;
};

export type DayLog = {
  date: string;
  meals: MealItem[];
  workouts: Workout[];
  waterMl: number;
  steps: number;
  activeCalories: number;
  sleepHours: number;
};

export type Goals = { calories: number; protein: number; waterMl: number; steps: number };
export type WeightPoint = { date: string; kg: number };
export type EstimationProfile = { bowlMl?: number; cupMl: number };

export type AppData = {
  goals: Goals;
  estimation: EstimationProfile;
  days: Record<string, DayLog>;
  weights: WeightPoint[];
  lastHealthSync?: string;
};

export type LogOperation =
  | {
      type: 'meal';
      action: 'add';
      slot: MealSlot;
      description: string;
      items: Omit<MealItem, 'id' | 'slot' | 'loggedAt'>[];
    }
  | { type: 'water'; action: 'add' | 'set'; amount: number }
  | {
      type: 'workout';
      action: 'add';
      name: string;
      durationMin: number;
      calories: number;
      calorieLow?: number;
      calorieHigh?: number;
      confidence?: EstimateConfidence;
      basis?: string;
      sourceLabel?: string;
      sourceId?: string;
      met?: number;
      intensity: Workout['intensity'];
    }
  | { type: 'steps'; action: 'add' | 'set'; amount: number }
  | { type: 'sleep'; action: 'set'; amount: number }
  | { type: 'weight'; action: 'set'; amount: number };

export type ParsedCommand = {
  transcript: string;
  confirmation: string;
  operations: LogOperation[];
  source: 'ai' | 'local';
  clarification?: {
    question: string;
    suggestions: string[];
  };
  profileUpdates?: {
    weightKg?: number;
    bowlMl?: number;
  };
};

export type HealthSnapshot = {
  steps?: number;
  activeCalories?: number;
  sleepHours?: number;
  weightKg?: number;
  source: 'Apple Health' | 'Health Connect';
};
