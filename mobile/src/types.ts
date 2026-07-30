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

export type Goals = {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  waterMl: number;
  steps: number;
  weeklyWorkoutMinutes: number;
  strengthDays: number;
};
export type WeightPoint = { date: string; kg: number };
export type EstimationProfile = { bowlMl?: number; cupMl: number };

export type PrimaryGoal =
  | 'lose-fat'
  | 'build-muscle'
  | 'maintain'
  | 'improve-fitness'
  | 'build-habits';
export type EquationSex = 'female' | 'male' | 'neutral';
export type GoalPace = 'gentle' | 'steady' | 'ambitious';
export type ActivityLevel = 'mostly-seated' | 'lightly-active' | 'active' | 'very-active';
export type WorkoutPreference = 'gym' | 'walking' | 'home' | 'mixed' | 'restarting';
export type ExperienceLevel = 'new' | 'some' | 'experienced';
export type DietStyle = 'home-indian' | 'vegetarian' | 'vegan' | 'mixed' | 'high-protein';
export type MainChallenge = 'portions' | 'protein' | 'cravings' | 'time' | 'consistency';
export type CoachingTone = 'gentle' | 'direct' | 'data-led';

export type PersonalProfile = {
  primaryGoal?: PrimaryGoal;
  equationSex?: EquationSex;
  age?: number;
  heightCm?: number;
  weightKg?: number;
  targetWeightKg?: number;
  goalPace?: GoalPace;
  activityLevel?: ActivityLevel;
  workoutPreference?: WorkoutPreference;
  experienceLevel?: ExperienceLevel;
  trainingDays?: number;
  availableMinutes?: number;
  dietStyle?: DietStyle;
  mealsPerDay?: number;
  allergies: string[];
  injuries: string[];
  mainChallenge?: MainChallenge;
  coachingTone?: CoachingTone;
  updatedAt: string;
};

export type PersonalPlan = {
  maintenanceCalories?: number;
  calorieAdjustment?: number;
  restingCalories?: number;
  method: string;
  summary: string;
  warnings: string[];
  updatedAt: string;
};

export type CoachMemory = {
  dietaryPreferences: string[];
  injuries: string[];
  workoutPreferences: string[];
  coachingStyle: string;
  notes: string;
  updatedAt: string;
};

export type CoachMessage = {
  id: string;
  role: 'user' | 'coach';
  text: string;
  createdAt: string;
};

// ---------------------------------------------------------------------------
// Training (Hevy-style strength logging)
// ---------------------------------------------------------------------------

export type SetType = 'normal' | 'warmup' | 'failure' | 'drop';

export type WorkoutSet = {
  id: string;
  type: SetType;
  weightKg?: number;
  reps?: number;
  durationSec?: number;
  rpe?: number;
  completed: boolean;
  /** Records earned when this set was checked off, e.g. ['Heaviest weight']. */
  prFlags?: string[];
};

export type RoutineSetTemplate = {
  id: string;
  type: SetType;
  weightKg?: number;
  reps?: number;
  repsMin?: number;
  repsMax?: number;
  durationSec?: number;
};

export type RoutineExercise = {
  id: string;
  exerciseId: string;
  note?: string;
  restSec: number;
  supersetId?: string;
  sets: RoutineSetTemplate[];
};

export type Routine = {
  id: string;
  name: string;
  folder?: string;
  exercises: RoutineExercise[];
  createdAt: string;
  updatedAt: string;
  lastPerformedAt?: string;
};

export type SessionExercise = {
  id: string;
  exerciseId: string;
  note?: string;
  restSec: number;
  supersetId?: string;
  sets: WorkoutSet[];
};

export type WorkoutSession = {
  id: string;
  name: string;
  routineId?: string;
  startedAt: string;
  endedAt?: string;
  durationMin?: number;
  exercises: SessionExercise[];
  totalVolumeKg?: number;
  totalSets?: number;
  records?: number;
  calories?: number;
  calorieLow?: number;
  calorieHigh?: number;
  calorieBasis?: string;
  note?: string;
};

export type CustomExercise = {
  id: string;
  name: string;
  equipment: string;
  primaryMuscle: string;
  kind: 'weight-reps' | 'reps-only' | 'duration';
  createdAt: string;
};

export type TrainingData = {
  routines: Routine[];
  sessions: WorkoutSession[];
  activeSession: WorkoutSession | null;
  customExercises: CustomExercise[];
  deletedRoutineIds: string[];
  deletedSessionIds: string[];
  defaultRestSec: number;
  rpeEnabled: boolean;
  updatedAt: string;
};

export type AppData = {
  goals: Goals;
  profile: PersonalProfile;
  plan: PersonalPlan;
  estimation: EstimationProfile;
  days: Record<string, DayLog>;
  weights: WeightPoint[];
  coachMemory: CoachMemory;
  coachMessages: CoachMessage[];
  deletedMealIds: string[];
  deletedWorkoutIds: string[];
  training: TrainingData;
  lastHealthSync?: string;
};

export type AccountUser = {
  id: string;
  email: string;
  displayName: string;
  createdAt: string;
};

export type AuthSession = {
  token: string;
  expiresAt: string;
  user: AccountUser;
};

export type SyncState = 'offline' | 'syncing' | 'synced' | 'error';

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
