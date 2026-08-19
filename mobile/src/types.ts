export type MealSlot = 'breakfast' | 'lunch' | 'snack' | 'dinner';
export type EstimateConfidence = 'high' | 'medium' | 'low';
export type EstimationContext = {
  weightKg?: number;
  bowlMl?: number;
  cupMl: number;
  /** Foods this user has taught the app; checked before the shipped catalog. */
  learned?: LearnedFood[];
};

/**
 * A food the user told the app about once.
 *
 * The shipped catalog cannot contain a particular Belgian wheat beer or the
 * dish someone's mother makes, and asking for the same label calories every
 * single time is how an app teaches people to stop using it. Once a figure has
 * been given, it is kept and reused.
 *
 * The figure is stored exactly as it was given — "500 ml was 215 kcal" — rather
 * than normalised to 100 g, because the user verified *that serving*, and a
 * per-100 conversion would silently invent a density the label never stated.
 * A different amount later is scaled from it linearly, which is the only honest
 * thing to do with a single data point.
 */
export type LearnedFood = {
  id: string;
  /** What the user calls it. Shown on the entry and editable in settings. */
  name: string;
  /** Lowercased phrases that should match this food. */
  aliases: string[];
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  /** The amount that `calories` describes, e.g. 500 with unit 'ml'. */
  servingAmount?: number;
  servingUnit?: string;
  createdAt: string;
  updatedAt: string;
};

/** Ingredient evidence attached to one composed dish, never logged as a second meal item. */
export type RecipeIngredient = {
  name: string;
  quantity?: string;
  calories?: number;
  protein?: number;
  carbs?: number;
  fat?: number;
};

export type MealRecipe = {
  /** A broad finished-dish reference, or a calculation from this portion's ingredients. */
  mode: 'dish-estimate' | 'ingredient-sum';
  ingredients: RecipeIngredient[];
  /** The exact ingredient sentence, retained so the user can edit it again. */
  input?: string;
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
  updatedAt?: string;
  dayKey?: string;
  batchId?: string;
  recipe?: MealRecipe;
  source: 'usda' | 'typical' | 'label' | 'ai' | 'local' | 'manual';
};

export type SavedMeal = {
  id: string;
  name: string;
  slot: MealSlot;
  items: Omit<MealItem, 'id' | 'slot' | 'loggedAt'>[];
  createdAt: string;
  updatedAt: string;
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
  waterUpdatedAt?: string;
  steps: number;
  stepsUpdatedAt?: string;
  activeCalories: number;
  activeCaloriesUpdatedAt?: string;
  sleepHours: number;
  sleepUpdatedAt?: string;
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

/**
 * A duration set that is currently running. Absolute timestamps make the
 * countdown resilient to navigation, app backgrounding, and process restore.
 */
export type ActiveSetTimer = {
  sessionExerciseId: string;
  setId: string;
  targetSec: number;
  startedAt: string;
  endsAt: string;
  pausedRemainingSec?: number;
};

export type ActiveRestTimer = {
  endsAt: number;
  totalSec: number;
};

export type LiveWorkoutMetrics = {
  heartRateBpm?: number;
  activeCalories?: number;
  source: 'Apple Watch' | 'Apple Health';
  updatedAt: string;
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
  activeSetTimer?: ActiveSetTimer;
  activeRestTimer?: ActiveRestTimer;
  liveMetrics?: LiveWorkoutMetrics;
  /** Monotonic Watch-side revision used to reject stale/offline sync packets. */
  watchRevision?: number;
  watchUpdatedAt?: string;
};

export type CustomExercise = {
  id: string;
  name: string;
  equipment: string;
  primaryMuscle: string;
  kind: 'weight-reps' | 'reps-only' | 'duration';
  createdAt: string;
  updatedAt: string;
};

export type TrainingData = {
  routines: Routine[];
  sessions: WorkoutSession[];
  activeSession: WorkoutSession | null;
  customExercises: CustomExercise[];
  deletedCustomExerciseIds: string[];
  deletedRoutineIds: string[];
  deletedSessionIds: string[];
  defaultRestSec: number;
  rpeEnabled: boolean;
  /** A workout-specific tombstone; unrelated routine edits must never clear Watch work. */
  activeWorkoutTombstone?: { workoutId: string; clearedAt: string };
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
  deletedSavedMealIds: string[];
  savedMeals: SavedMeal[];
  learnedFoods: LearnedFood[];
  deletedLearnedFoodIds: string[];
  training: TrainingData;
  lastHealthSync?: string;
  healthSync?: HealthSyncRecord;
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
      batchId?: string;
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

/**
 * What a clarifying question is actually asking about.
 *
 * The parser used to ask its question in prose only, and the answer was glued
 * onto the end of the transcript and re-parsed. That could not work: an answer
 * like "1 bowl" carries no reference to the food it belongs to, and quantities
 * are only read next to the food they describe — so the same question came back
 * forever while the transcript grew. Naming the target lets the answer be
 * applied to the exact thing that was missing.
 */
export type ClarificationTarget =
  | { kind: 'foodAmount'; alias: string; foodName: string }
  | { kind: 'pieceGrams'; alias: string; foodName: string }
  | { kind: 'bowlMl' }
  | { kind: 'bodyWeight' }
  | { kind: 'workoutMinutes'; activity: string }
  | { kind: 'workoutIntensity'; activity: string }
  | { kind: 'unknownFood' }
  | { kind: 'intent' };

/** A clarifying question, resolved into a value the parser can use. */
export type ClarificationAnswer =
  | { kind: 'foodAmount'; alias: string; amount: number; unit: string }
  | { kind: 'pieceGrams'; alias: string; grams: number }
  | { kind: 'bowlMl'; ml: number }
  | { kind: 'bodyWeight'; kg: number }
  | { kind: 'workoutMinutes'; minutes: number }
  | { kind: 'workoutIntensity'; intensity: Workout['intensity'] };

/**
 * Carried by the logging screen between rounds. `transcript` is the original
 * utterance and never grows; every answered detail is accumulated in `answers`.
 */
export type PendingClarification = {
  transcript: string;
  target: ClarificationTarget;
  answers: ClarificationAnswer[];
};

export type ParsedCommand = {
  transcript: string;
  confirmation: string;
  operations: LogOperation[];
  source: 'ai' | 'local';
  clarification?: {
    question: string;
    suggestions: string[];
    target: ClarificationTarget;
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
  sampleCount?: number;
  sampleCounts?: {
    steps: number;
    activeCalories: number;
    sleep: number;
    weight: number;
  };
  latestSampleAt?: string;
  source: 'Apple Health' | 'Health Connect';
};

export type HealthSyncRecord = {
  source?: HealthSnapshot['source'];
  status: 'current' | 'empty' | 'error';
  lastAttemptAt: string;
  lastSuccessAt?: string;
  latestSampleAt?: string;
  sampleCounts?: HealthSnapshot['sampleCounts'];
  message?: string;
};
