import * as Device from 'expo-device';
import { requestRecordingPermissionsAsync } from 'expo-audio';
import { type Href, useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph, type GlyphName } from '@/src/components/glyph';
import { readServiceHealth } from '@/src/lib/api-client';
import { healthSetupCopy, syncNativeHealth } from '@/src/lib/health';
import { calculatePersonalTargets, goalLabel } from '@/src/lib/personalization';
import { useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, type } from '@/src/theme';
import type {
  ActivityLevel,
  CoachingTone,
  DietStyle,
  EquationSex,
  ExperienceLevel,
  GoalPace,
  MainChallenge,
  PersonalProfile,
  PrimaryGoal,
  WorkoutPreference,
} from '@/src/types';

const TOTAL_STEPS = 10;

type Choice<T extends string> = {
  value: T;
  title: string;
  body: string;
  icon?: GlyphName;
};

const goalChoices: Choice<PrimaryGoal>[] = [
  { value: 'lose-fat', title: 'Lose fat, steadily', body: 'Use a measured deficit and adjust from my trend—not a crash plan.', icon: 'chart' },
  { value: 'build-muscle', title: 'Build muscle and strength', body: 'Prioritize training, protein and a controlled energy surplus.', icon: 'dumbbell' },
  { value: 'maintain', title: 'Maintain my weight', body: 'Keep energy stable while improving food quality and routine.', icon: 'heart' },
  { value: 'improve-fitness', title: 'Improve fitness and energy', body: 'Build cardio, strength, sleep and daily movement together.', icon: 'steps' },
  { value: 'build-habits', title: 'Become consistent first', body: 'Start with targets I can repeat on an ordinary week.', icon: 'spark' },
];

const sexChoices: Choice<EquationSex>[] = [
  { value: 'female', title: 'Female equation', body: 'Uses the female Mifflin–St Jeor constant.' },
  { value: 'male', title: 'Male equation', body: 'Uses the male Mifflin–St Jeor constant.' },
  { value: 'neutral', title: 'Use a midpoint', body: 'Less precise, but does not require choosing either equation.' },
];

const paceChoices: Choice<GoalPace>[] = [
  { value: 'gentle', title: 'Gentle', body: 'Smaller calorie change; easiest to sustain and assess.' },
  { value: 'steady', title: 'Steady', body: 'A practical middle ground for most routines.' },
  { value: 'ambitious', title: 'Ambitious', body: 'Larger change; monitor hunger, recovery and performance.' },
];

const activityChoices: Choice<ActivityLevel>[] = [
  { value: 'mostly-seated', title: 'Mostly seated', body: 'Desk-based day with little walking outside planned exercise.' },
  { value: 'lightly-active', title: 'Some daily movement', body: 'Regular errands or walking, but much of the day is seated.' },
  { value: 'active', title: 'Active most days', body: 'A mobile job, frequent walking or regular training.' },
  { value: 'very-active', title: 'Very active', body: 'Physical work, long training sessions or high daily movement.' },
];

const workoutChoices: Choice<WorkoutPreference>[] = [
  { value: 'gym', title: 'Gym training', body: 'Weights, machines or structured classes.' },
  { value: 'walking', title: 'Walking and cardio', body: 'Steps and simple aerobic sessions fit best.' },
  { value: 'home', title: 'Home workouts', body: 'Bodyweight, yoga or dumbbells at home.' },
  { value: 'mixed', title: 'A flexible mix', body: 'Use whichever option fits the day.' },
  { value: 'restarting', title: 'I am restarting', body: 'Begin below the standard target and build momentum.' },
];

const experienceChoices: Choice<ExperienceLevel>[] = [
  { value: 'new', title: 'New', body: 'I need clear, simple starting points.' },
  { value: 'some', title: 'Some experience', body: 'I know the basics but want consistency.' },
  { value: 'experienced', title: 'Experienced', body: 'I track performance and can handle more detail.' },
];

const dietChoices: Choice<DietStyle>[] = [
  { value: 'home-indian', title: 'Mostly home-cooked Indian', body: 'Roti, rice, dal, sabzi, curd and recipe-based portions.' },
  { value: 'vegetarian', title: 'Vegetarian', body: 'Dairy and/or eggs are fine; keep protein practical.' },
  { value: 'vegan', title: 'Vegan', body: 'Use plant-only suggestions and protein sources.' },
  { value: 'mixed', title: 'Mixed diet', body: 'Home food, cafés, takeout, meat and vegetarian meals.' },
  { value: 'high-protein', title: 'Already protein-first', body: 'I regularly use eggs, meat, paneer, tofu, whey or soy.' },
];

const challengeChoices: Choice<MainChallenge>[] = [
  { value: 'portions', title: 'Portion uncertainty', body: 'I struggle to estimate home food and serving sizes.' },
  { value: 'protein', title: 'Getting enough protein', body: 'I need realistic protein choices across the day.' },
  { value: 'cravings', title: 'Cravings or snacking', body: 'Evenings or stressful days usually derail me.' },
  { value: 'time', title: 'Time and planning', body: 'Suggestions must work on busy days.' },
  { value: 'consistency', title: 'Staying consistent', body: 'I start well and then stop tracking.' },
];

const toneChoices: Choice<CoachingTone>[] = [
  { value: 'gentle', title: 'Gentle', body: 'Encourage me without guilt or pressure.' },
  { value: 'direct', title: 'Direct', body: 'Give me the clearest next action.' },
  { value: 'data-led', title: 'Data-led', body: 'Explain the numbers and tradeoffs briefly.' },
];

const goalNotes: Record<PrimaryGoal, string> = {
  'lose-fat': 'Your target begins below estimated maintenance and should be reviewed against a 2–4 week weight trend.',
  'build-muscle': 'Your target begins slightly above estimated maintenance, with strength training and protein prioritized.',
  maintain: 'Your target stays near estimated maintenance while consistency and body-weight trend guide adjustments.',
  'improve-fitness': 'Your target stays near maintenance while weekly movement and recovery become the primary signals.',
  'build-habits': 'Your target stays conservative so the routine can become stable before adding difficulty.',
};

export default function OnboardingScreen() {
  const router = useRouter();
  const {
    applyHealthSnapshot,
    data,
    hydrated,
    savePersonalization,
    updateCoachMemory,
  } = useApp();
  const { completeOnboarding } = useAuth();
  const [step, setStep] = useState(0);
  const [primaryGoal, setPrimaryGoal] = useState<PrimaryGoal | null>(data.profile.primaryGoal ?? null);
  const [equationSex, setEquationSex] = useState<EquationSex | null>(data.profile.equationSex ?? null);
  const [age, setAge] = useState(data.profile.age ? String(data.profile.age) : '');
  const [height, setHeight] = useState(data.profile.heightCm ? String(data.profile.heightCm) : '');
  const [weight, setWeight] = useState(data.profile.weightKg ? String(data.profile.weightKg) : '');
  const [targetWeight, setTargetWeight] = useState(data.profile.targetWeightKg ? String(data.profile.targetWeightKg) : '');
  const [goalPace, setGoalPace] = useState<GoalPace | null>(data.profile.goalPace ?? null);
  const [activityLevel, setActivityLevel] = useState<ActivityLevel | null>(data.profile.activityLevel ?? null);
  const [workoutPreference, setWorkoutPreference] = useState<WorkoutPreference | null>(data.profile.workoutPreference ?? null);
  const [experienceLevel, setExperienceLevel] = useState<ExperienceLevel | null>(data.profile.experienceLevel ?? null);
  const [trainingDays, setTrainingDays] = useState(String(data.profile.trainingDays ?? 3));
  const [availableMinutes, setAvailableMinutes] = useState(String(data.profile.availableMinutes ?? 30));
  const [dietStyle, setDietStyle] = useState<DietStyle | null>(data.profile.dietStyle ?? null);
  const [mealsPerDay, setMealsPerDay] = useState(String(data.profile.mealsPerDay ?? 3));
  const [allergies, setAllergies] = useState(data.profile.allergies.join(', '));
  const [injuries, setInjuries] = useState(data.profile.injuries.join(', '));
  const [mainChallenge, setMainChallenge] = useState<MainChallenge | null>(data.profile.mainChallenge ?? null);
  const [coachingTone, setCoachingTone] = useState<CoachingTone | null>(data.profile.coachingTone ?? null);
  const [bowl, setBowl] = useState(data.estimation.bowlMl ? String(data.estimation.bowlMl) : '');
  const [healthBusy, setHealthBusy] = useState(false);
  const [healthMessage, setHealthMessage] = useState('');
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [voiceMessage, setVoiceMessage] = useState('');
  const setup = healthSetupCopy();

  const profile = useMemo<PersonalProfile>(() => ({
    primaryGoal: primaryGoal ?? undefined,
    equationSex: equationSex ?? undefined,
    age: numeric(age),
    heightCm: numeric(height),
    weightKg: numeric(weight),
    targetWeightKg: numeric(targetWeight),
    goalPace: goalPace ?? undefined,
    activityLevel: activityLevel ?? undefined,
    workoutPreference: workoutPreference ?? undefined,
    experienceLevel: experienceLevel ?? undefined,
    trainingDays: numeric(trainingDays),
    availableMinutes: numeric(availableMinutes),
    dietStyle: dietStyle ?? undefined,
    mealsPerDay: numeric(mealsPerDay),
    allergies: list(allergies),
    injuries: list(injuries),
    mainChallenge: mainChallenge ?? undefined,
    coachingTone: coachingTone ?? undefined,
    updatedAt: '',
  }), [
    activityLevel,
    age,
    allergies,
    availableMinutes,
    coachingTone,
    dietStyle,
    equationSex,
    experienceLevel,
    goalPace,
    height,
    injuries,
    mainChallenge,
    mealsPerDay,
    primaryGoal,
    targetWeight,
    trainingDays,
    weight,
    workoutPreference,
  ]);
  const preview = useMemo(() => calculatePersonalTargets(profile), [profile]);

  const canContinue = step === 0
    || (step === 1 && Boolean(primaryGoal))
    || (step === 2
      && Boolean(equationSex)
      && inRange(age, 18, 90)
      && inRange(height, 125, 230)
      && inRange(weight, 35, 250))
    || (step === 3 && Boolean(goalPace) && validTarget(primaryGoal, weight, targetWeight))
    || (step === 4 && Boolean(activityLevel))
    || (step === 5
      && Boolean(workoutPreference)
      && Boolean(experienceLevel)
      && inRange(trainingDays, 1, 7)
      && inRange(availableMinutes, 10, 120))
    || (step === 6 && Boolean(dietStyle) && inRange(mealsPerDay, 2, 6))
    || (step === 7 && Boolean(mainChallenge) && Boolean(coachingTone))
    || step >= 8;

  async function checkVoice() {
    setVoiceBusy(true);
    setVoiceMessage('');
    try {
      if (Platform.OS === 'ios' && !Device.isDevice) {
        setVoiceMessage('The iOS Simulator cannot record a voice command. Typed logging works here; test the microphone in an iPhone development build.');
        return;
      }
      const permission = await requestRecordingPermissionsAsync();
      if (!permission.granted) {
        setVoiceMessage('Microphone access is off. Enable it in system settings, then return here.');
        return;
      }
      const service = await readServiceHealth();
      setVoiceMessage(
        service.ai_enabled
          ? 'Microphone and voice service are ready. Sign in on the next screen to use voice logging.'
          : 'Microphone works, but the local service has no Gemini key. Typed logging remains available.',
      );
    } catch (error) {
      const deviceHint = Device.isDevice
        ? 'Set EXPO_PUBLIC_API_URL to this Mac’s LAN address and keep both devices on the same Wi-Fi.'
        : 'Start the API on port 8000, then try again.';
      setVoiceMessage(`${error instanceof Error ? error.message : 'The local service is unreachable.'} ${deviceHint}`);
    } finally {
      setVoiceBusy(false);
    }
  }

  async function connectHealth() {
    setHealthBusy(true);
    setHealthMessage('');
    try {
      const snapshot = await syncNativeHealth();
      applyHealthSnapshot(snapshot);
      const hasSamples = Boolean(
        snapshot.steps || snapshot.activeCalories || snapshot.sleepHours || snapshot.weightKg,
      );
      setHealthMessage(
        hasSamples
          ? `Connected to ${snapshot.source} and imported today’s approved data.`
          : `Connected to ${snapshot.source}. No approved samples were found. Check Health sharing permissions; use a physical phone for real wearable records.`,
      );
    } catch (error) {
      setHealthMessage(error instanceof Error ? error.message : 'Health connection failed.');
    } finally {
      setHealthBusy(false);
    }
  }

  async function finish() {
    savePersonalization(profile, numeric(bowl));
    const diet = dietChoices.find((choice) => choice.value === dietStyle)?.title;
    const workout = workoutChoices.find((choice) => choice.value === workoutPreference)?.title;
    const challenge = challengeChoices.find((choice) => choice.value === mainChallenge)?.title;
    updateCoachMemory({
      dietaryPreferences: [
        ...(diet ? [diet] : []),
        ...profile.allergies.map((item) => `Avoid: ${item}`),
      ],
      injuries: profile.injuries,
      workoutPreferences: workout ? [workout] : [],
      coachingStyle: coachingTone === 'gentle'
        ? 'gentle, practical and never guilt-based'
        : coachingTone === 'data-led'
          ? 'concise and data-led; explain the basis for recommendations'
          : 'supportive, direct and focused on the next action',
      notes: [
        `Primary goal: ${goalLabel(primaryGoal ?? undefined)}.`,
        challenge ? `Main challenge: ${challenge}.` : '',
        profile.targetWeightKg ? `Target weight: ${profile.targetWeightKg} kg.` : '',
        `Usual meal pattern: ${profile.mealsPerDay ?? 3} meals per day.`,
      ].filter(Boolean).join(' '),
    });
    await completeOnboarding();
    router.replace('/auth' as Href);
  }

  if (!hydrated) {
    return <View style={styles.loading}><ActivityIndicator color={palette.forest} /></View>;
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <View style={styles.topBar}>
          {step > 0 ? (
            <Pressable accessibilityLabel="Go back" onPress={() => setStep((current) => current - 1)} style={styles.back}>
              <Text style={styles.backText}>←</Text>
            </Pressable>
          ) : <View style={styles.back} />}
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${((step + 1) / TOTAL_STEPS) * 100}%` }]} />
          </View>
          <Text style={styles.stepCount}>{step + 1}/{TOTAL_STEPS}</Text>
        </View>

        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
          {step === 0 ? <Welcome />
            : step === 1 ? (
              <Question
                eyebrow="THE OUTCOME"
                title="What should this plan help you change?"
                body="This becomes the basis for calories, macros, training priorities and daily suggestions."
                choices={goalChoices}
                selected={primaryGoal}
                onSelect={setPrimaryGoal}
              />
            ) : step === 2 ? (
              <BodyBasics
                age={age}
                equationSex={equationSex}
                height={height}
                setAge={setAge}
                setEquationSex={setEquationSex}
                setHeight={setHeight}
                setWeight={setWeight}
                weight={weight}
              />
            ) : step === 3 ? (
              <GoalDirection
                currentWeight={weight}
                goal={primaryGoal}
                pace={goalPace}
                setPace={setGoalPace}
                setTargetWeight={setTargetWeight}
                targetWeight={targetWeight}
              />
            ) : step === 4 ? (
              <Question
                eyebrow="A NORMAL DAY"
                title="How active are you outside planned workouts?"
                body="Choose your real routine. Workout sessions are handled separately on the next screen."
                choices={activityChoices}
                selected={activityLevel}
                onSelect={setActivityLevel}
              />
            ) : step === 5 ? (
              <TrainingSetup
                availableMinutes={availableMinutes}
                experience={experienceLevel}
                preference={workoutPreference}
                setAvailableMinutes={setAvailableMinutes}
                setExperience={setExperienceLevel}
                setPreference={setWorkoutPreference}
                setTrainingDays={setTrainingDays}
                trainingDays={trainingDays}
              />
            ) : step === 6 ? (
              <FoodSetup
                allergies={allergies}
                dietStyle={dietStyle}
                mealsPerDay={mealsPerDay}
                setAllergies={setAllergies}
                setDietStyle={setDietStyle}
                setMealsPerDay={setMealsPerDay}
              />
            ) : step === 7 ? (
              <SupportSetup
                challenge={mainChallenge}
                injuries={injuries}
                setChallenge={setMainChallenge}
                setInjuries={setInjuries}
                setTone={setCoachingTone}
                tone={coachingTone}
              />
            ) : step === 8 ? (
              <Calibration bowl={bowl} setBowl={setBowl} />
            ) : (
              <PlanPreview
                goal={primaryGoal}
                healthBusy={healthBusy}
                healthMessage={healthMessage}
                onConnectHealth={() => void connectHealth()}
                onTestVoice={() => void checkVoice()}
                plan={preview.plan}
                goals={preview.goals}
                setupDetail={setup.detail}
                setupTitle={setup.title}
                voiceBusy={voiceBusy}
                voiceMessage={voiceMessage}
              />
            )}
        </ScrollView>

        <View style={styles.footer}>
          {step < TOTAL_STEPS - 1 ? (
            <Pressable
              disabled={!canContinue}
              onPress={() => setStep((current) => current + 1)}
              style={({ pressed }) => [styles.primary, !canContinue && styles.disabled, pressed && styles.pressed]}>
              <Text style={styles.primaryText}>{step === 0 ? 'Build my plan' : 'Continue'}</Text>
            </Pressable>
          ) : (
            <Pressable onPress={() => void finish()} style={({ pressed }) => [styles.primary, pressed && styles.pressed]}>
              <Text style={styles.primaryText}>Save plan and continue</Text>
            </Pressable>
          )}
          {step === 8 ? <Text style={styles.footerHint}>Bowl size is optional and can be added later.</Text> : null}
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function Welcome() {
  return (
    <View style={styles.welcome}>
      <View style={styles.brandMark}><Glyph name="spark" color={palette.lime} size={28} /></View>
      <Text style={styles.welcomeKicker}>CALORIE LENS</Text>
      <Text style={styles.welcomeTitle}>Your plan should know who it is for.</Text>
      <Text style={styles.welcomeBody}>
        A few honest answers will set your calories, macros, movement targets and coaching priorities. Every answer stays editable.
      </Text>
      <View style={styles.promise}>
        <PromiseRow icon="chart" text="Targets calculated from your body, routine and goal" />
        <PromiseRow icon="dumbbell" text="Suggestions shaped around your time and training style" />
        <PromiseRow icon="bowl" text="Food guidance that respects how you actually eat" />
      </View>
      <Text style={styles.safetyCopy}>For adults 18+. Estimates support general wellness and do not replace medical or dietetic care.</Text>
    </View>
  );
}

function PromiseRow({ icon, text }: { icon: GlyphName; text: string }) {
  return (
    <View style={styles.promiseRow}>
      <View style={styles.promiseIcon}><Glyph name={icon} color={palette.forest} size={18} /></View>
      <Text style={styles.promiseText}>{text}</Text>
    </View>
  );
}

function SectionIntro({ body, eyebrow, title }: { body: string; eyebrow: string; title: string }) {
  return (
    <>
      <Text style={styles.eyebrow}>{eyebrow}</Text>
      <Text style={styles.questionTitle}>{title}</Text>
      <Text style={styles.questionBody}>{body}</Text>
    </>
  );
}

function Question<T extends string>({
  body,
  choices,
  eyebrow,
  onSelect,
  selected,
  title,
}: {
  body: string;
  choices: Choice<T>[];
  eyebrow: string;
  onSelect: (value: T) => void;
  selected: T | null;
  title: string;
}) {
  return (
    <View>
      <SectionIntro body={body} eyebrow={eyebrow} title={title} />
      <ChoiceList choices={choices} onSelect={onSelect} selected={selected} />
    </View>
  );
}

function ChoiceList<T extends string>({
  choices,
  onSelect,
  selected,
}: {
  choices: Choice<T>[];
  onSelect: (value: T) => void;
  selected: T | null;
}) {
  return (
    <View style={styles.choices}>
      {choices.map((choice) => {
        const active = selected === choice.value;
        return (
          <Pressable
            key={choice.value}
            onPress={() => onSelect(choice.value)}
            style={({ pressed }) => [styles.choice, active && styles.choiceActive, pressed && styles.pressed]}>
            {choice.icon ? (
              <View style={[styles.choiceIcon, active && styles.choiceIconActive]}>
                <Glyph name={choice.icon} color={palette.forest} size={19} />
              </View>
            ) : null}
            <View style={styles.choiceCopy}>
              <Text style={styles.choiceTitle}>{choice.title}</Text>
              <Text style={styles.choiceBody}>{choice.body}</Text>
            </View>
            <View style={[styles.radio, active && styles.radioActive]}>
              {active ? <View style={styles.radioDot} /> : null}
            </View>
          </Pressable>
        );
      })}
    </View>
  );
}

function BodyBasics({
  age,
  equationSex,
  height,
  setAge,
  setEquationSex,
  setHeight,
  setWeight,
  weight,
}: {
  age: string;
  equationSex: EquationSex | null;
  height: string;
  setAge: (value: string) => void;
  setEquationSex: (value: EquationSex) => void;
  setHeight: (value: string) => void;
  setWeight: (value: string) => void;
  weight: string;
}) {
  return (
    <View>
      <SectionIntro
        eyebrow="YOUR STARTING POINT"
        title="Calculate energy from your body—not a generic 2,200."
        body="Age, height and weight estimate resting energy. The equation option is about physiology used by the formula, not gender identity."
      />
      <View style={styles.formCard}>
        <NumberField label="Age" unit="years" value={age} onChange={setAge} placeholder="29" />
        <NumberField label="Height" unit="cm" value={height} onChange={setHeight} placeholder="172" />
        <NumberField label="Current weight" unit="kg" value={weight} onChange={setWeight} placeholder="74" />
      </View>
      <Text style={styles.groupLabel}>ENERGY EQUATION</Text>
      <ChoiceList choices={sexChoices} onSelect={setEquationSex} selected={equationSex} />
    </View>
  );
}

function GoalDirection({
  currentWeight,
  goal,
  pace,
  setPace,
  setTargetWeight,
  targetWeight,
}: {
  currentWeight: string;
  goal: PrimaryGoal | null;
  pace: GoalPace | null;
  setPace: (value: GoalPace) => void;
  setTargetWeight: (value: string) => void;
  targetWeight: string;
}) {
  const targetError = targetWeight && !validTarget(goal, currentWeight, targetWeight);
  return (
    <View>
      <SectionIntro
        eyebrow="DIRECTION, NOT A DEADLINE"
        title="How quickly should the starting plan move?"
        body={goal ? goalNotes[goal] : 'The pace controls the starting calorie adjustment.'}
      />
      <View style={styles.formCard}>
        <NumberField label="Target weight" unit="kg" value={targetWeight} onChange={setTargetWeight} placeholder="Optional" />
      </View>
      {targetError ? (
        <Text style={styles.errorText}>
          For this goal, choose a target in the intended direction or leave it blank.
        </Text>
      ) : <Text style={styles.helperText}>A target weight helps the coach frame progress, but it does not create a deadline.</Text>}
      <Text style={styles.groupLabel}>STARTING PACE</Text>
      <ChoiceList choices={paceChoices} onSelect={setPace} selected={pace} />
    </View>
  );
}

function TrainingSetup({
  availableMinutes,
  experience,
  preference,
  setAvailableMinutes,
  setExperience,
  setPreference,
  setTrainingDays,
  trainingDays,
}: {
  availableMinutes: string;
  experience: ExperienceLevel | null;
  preference: WorkoutPreference | null;
  setAvailableMinutes: (value: string) => void;
  setExperience: (value: ExperienceLevel) => void;
  setPreference: (value: WorkoutPreference) => void;
  setTrainingDays: (value: string) => void;
  trainingDays: string;
}) {
  return (
    <View>
      <SectionIntro
        eyebrow="A PLAN THAT FITS THE WEEK"
        title="What training can you realistically repeat?"
        body="These answers set weekly minutes, strength frequency and the kind of suggestions you receive."
      />
      <ChoiceList choices={workoutChoices} onSelect={setPreference} selected={preference} />
      <Text style={styles.groupLabel}>YOUR EXPERIENCE</Text>
      <ChoiceList choices={experienceChoices} onSelect={setExperience} selected={experience} />
      <View style={styles.formCard}>
        <NumberField label="Days available" unit="/ week" value={trainingDays} onChange={setTrainingDays} placeholder="3" />
        <NumberField label="Time per session" unit="minutes" value={availableMinutes} onChange={setAvailableMinutes} placeholder="30" />
      </View>
    </View>
  );
}

function FoodSetup({
  allergies,
  dietStyle,
  mealsPerDay,
  setAllergies,
  setDietStyle,
  setMealsPerDay,
}: {
  allergies: string;
  dietStyle: DietStyle | null;
  mealsPerDay: string;
  setAllergies: (value: string) => void;
  setDietStyle: (value: DietStyle) => void;
  setMealsPerDay: (value: string) => void;
}) {
  return (
    <View>
      <SectionIntro
        eyebrow="YOUR FOOD, NOT A TEMPLATE"
        title="What does eating normally look like?"
        body="The coach uses this to choose relevant protein sources, meal examples and portion advice."
      />
      <ChoiceList choices={dietChoices} onSelect={setDietStyle} selected={dietStyle} />
      <View style={styles.formCard}>
        <NumberField label="Meals most days" unit="/ day" value={mealsPerDay} onChange={setMealsPerDay} placeholder="3" />
        <TextField
          label="Allergies or foods to avoid"
          onChange={setAllergies}
          placeholder="e.g. peanuts, shellfish"
          value={allergies}
        />
      </View>
      <Text style={styles.helperText}>Separate multiple items with commas. Leave blank if none.</Text>
    </View>
  );
}

function SupportSetup({
  challenge,
  injuries,
  setChallenge,
  setInjuries,
  setTone,
  tone,
}: {
  challenge: MainChallenge | null;
  injuries: string;
  setChallenge: (value: MainChallenge) => void;
  setInjuries: (value: string) => void;
  setTone: (value: CoachingTone) => void;
  tone: CoachingTone | null;
}) {
  return (
    <View>
      <SectionIntro
        eyebrow="WHAT USUALLY GETS IN THE WAY"
        title="Make the advice useful on difficult days."
        body="Your main challenge determines which gap the app calls out first."
      />
      <ChoiceList choices={challengeChoices} onSelect={setChallenge} selected={challenge} />
      <Text style={styles.groupLabel}>HOW SHOULD THE COACH SPEAK?</Text>
      <ChoiceList choices={toneChoices} onSelect={setTone} selected={tone} />
      <View style={styles.formCard}>
        <TextField
          label="Injuries or movement limits"
          onChange={setInjuries}
          placeholder="e.g. sensitive left knee"
          value={injuries}
        />
      </View>
      <Text style={styles.helperText}>The app avoids suggesting around a noted limitation, but it cannot diagnose or rehabilitate an injury.</Text>
    </View>
  );
}

function Calibration({ bowl, setBowl }: { bowl: string; setBowl: (value: string) => void }) {
  return (
    <View>
      <SectionIntro
        eyebrow="ONE USEFUL MEASUREMENT"
        title="Make home-food estimates less random."
        body="Your usual bowl size changes calculations for dal, rajma, rice and other foods logged by volume."
      />
      <View style={styles.formCard}>
        <NumberField label="Your usual bowl" unit="ml" value={bowl} onChange={setBowl} placeholder="200" />
      </View>
      <View style={styles.tip}>
        <Glyph name="water" color={palette.limeDark} size={19} />
        <Text style={styles.tipText}>Fill the bowl with water once and pour it into a measuring jug. You can skip this and add it later.</Text>
      </View>
      <View style={styles.whyCard}>
        <Text style={styles.whyTitle}>What will stay approximate?</Text>
        <Text style={styles.whyBody}>Oil, recipes and restaurant portions still vary. The app shows a range and its assumptions before saving.</Text>
      </View>
    </View>
  );
}

function PlanPreview({
  goal,
  goals,
  healthBusy,
  healthMessage,
  onConnectHealth,
  onTestVoice,
  plan,
  setupDetail,
  setupTitle,
  voiceBusy,
  voiceMessage,
}: {
  goal: PrimaryGoal | null;
  goals: ReturnType<typeof calculatePersonalTargets>['goals'];
  healthBusy: boolean;
  healthMessage: string;
  onConnectHealth: () => void;
  onTestVoice: () => void;
  plan: ReturnType<typeof calculatePersonalTargets>['plan'];
  setupDetail: string;
  setupTitle: string;
  voiceBusy: boolean;
  voiceMessage: string;
}) {
  return (
    <View>
      <SectionIntro
        eyebrow="YOUR STARTING PLAN"
        title={`${goalLabel(goal ?? undefined)}—with numbers that have a reason.`}
        body="These are starting estimates. Your 2–4 week trend, energy and training performance should guide later adjustments."
      />
      <View style={styles.planCard}>
        <Text style={styles.planKicker}>DAILY TARGETS</Text>
        <Text style={styles.planCalories}>{goals.calories.toLocaleString()} <Text style={styles.planUnit}>kcal</Text></Text>
        <Text style={styles.planBasis}>{plan.summary}</Text>
        <View style={styles.metricGrid}>
          <PlanMetric label="Protein" value={`${goals.protein} g`} />
          <PlanMetric label="Water" value={`${(goals.waterMl / 1000).toFixed(1)} L`} />
          <PlanMetric label="Steps" value={goals.steps.toLocaleString()} />
          <PlanMetric label="Training" value={`${goals.weeklyWorkoutMinutes} min/wk`} />
        </View>
        <View style={styles.methodRow}>
          <Text style={styles.methodLabel}>METHOD</Text>
          <Text style={styles.methodText}>{plan.method} · maintenance ~{plan.maintenanceCalories} kcal</Text>
        </View>
      </View>
      {plan.warnings.map((warning) => <Text key={warning} style={styles.warningText}>• {warning}</Text>)}

      <Text style={styles.groupLabel}>OPTIONAL CONNECTION CHECKS</Text>
      <SetupCard
        body={setupDetail}
        busy={healthBusy}
        button="Connect"
        icon="heart"
        message={healthMessage}
        onPress={onConnectHealth}
        title={setupTitle}
      />
      <SetupCard
        body={Platform.OS === 'ios' && !Device.isDevice
          ? 'The iOS Simulator cannot record voice. Typed logging works here; test voice on an iPhone.'
          : 'Checks microphone permission and whether the local AI service is ready.'}
        busy={voiceBusy}
        button="Test voice"
        icon="mic"
        message={voiceMessage}
        onPress={onTestVoice}
        title="Voice logging"
      />
    </View>
  );
}

function PlanMetric({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.planMetric}>
      <Text style={styles.planMetricValue}>{value}</Text>
      <Text style={styles.planMetricLabel}>{label}</Text>
    </View>
  );
}

function SetupCard({
  body,
  busy,
  button,
  icon,
  message,
  onPress,
  title,
}: {
  body: string;
  busy: boolean;
  button: string;
  icon: GlyphName;
  message: string;
  onPress: () => void;
  title: string;
}) {
  return (
    <View style={styles.setupCard}>
      <View style={styles.setupRow}>
        <View style={styles.setupIcon}><Glyph name={icon} color={palette.forest} size={20} /></View>
        <View style={styles.choiceCopy}>
          <Text style={styles.choiceTitle}>{title}</Text>
          <Text style={styles.choiceBody}>{body}</Text>
        </View>
      </View>
      {message ? <Text style={styles.setupMessage}>{message}</Text> : null}
      <Pressable disabled={busy} onPress={onPress} style={({ pressed }) => [styles.setupButton, pressed && styles.pressed]}>
        {busy ? <ActivityIndicator color={palette.forest} size="small" /> : <Text style={styles.setupButtonText}>{button}</Text>}
      </Pressable>
    </View>
  );
}

function NumberField({
  label,
  onChange,
  placeholder,
  unit,
  value,
}: {
  label: string;
  onChange: (value: string) => void;
  placeholder: string;
  unit: string;
  value: string;
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <View style={styles.inputRow}>
        <TextInput
          keyboardType="decimal-pad"
          onChangeText={(next) => onChange(next.replace(/[^\d.]/g, ''))}
          placeholder={placeholder}
          placeholderTextColor="#929A93"
          selectTextOnFocus
          style={styles.numberInput}
          value={value}
        />
        <Text style={styles.fieldUnit}>{unit}</Text>
      </View>
    </View>
  );
}

function TextField({
  label,
  onChange,
  placeholder,
  value,
}: {
  label: string;
  onChange: (value: string) => void;
  placeholder: string;
  value: string;
}) {
  return (
    <View style={styles.textField}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput
        autoCapitalize="sentences"
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor="#929A93"
        style={styles.textInput}
        value={value}
      />
    </View>
  );
}

function numeric(value: string) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function inRange(value: string, min: number, max: number) {
  const parsed = numeric(value);
  return parsed != null && parsed >= min && parsed <= max;
}

function validTarget(goal: PrimaryGoal | null, current: string, target: string) {
  if (!target) return true;
  if (!inRange(target, 35, 250)) return false;
  const currentValue = numeric(current);
  const targetValue = numeric(target);
  if (!currentValue || !targetValue) return false;
  if (goal === 'lose-fat') return targetValue < currentValue;
  if (goal === 'build-muscle') return targetValue >= currentValue * 0.9;
  return true;
}

function list(value: string) {
  return value.split(',').map((item) => item.trim()).filter(Boolean).slice(0, 12);
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  fill: { flex: 1 },
  loading: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.canvas },
  topBar: { height: 52, paddingHorizontal: space.md, flexDirection: 'row', alignItems: 'center', gap: 12 },
  back: { width: 34, height: 34, alignItems: 'center', justifyContent: 'center' },
  backText: { color: palette.ink, fontFamily: type.demi, fontSize: 22 },
  progressTrack: { flex: 1, height: 4, borderRadius: 2, backgroundColor: palette.line, overflow: 'hidden' },
  progressFill: { height: '100%', borderRadius: 2, backgroundColor: palette.limeDark },
  stepCount: { width: 38, color: palette.muted, fontFamily: type.demi, fontSize: 9, textAlign: 'right' },
  content: { flexGrow: 1, paddingHorizontal: space.lg, paddingTop: 18, paddingBottom: 30 },
  footer: { paddingHorizontal: space.lg, paddingTop: 10, paddingBottom: 12, backgroundColor: palette.canvas },
  primary: { height: 56, borderRadius: radius.md, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  primaryText: { color: palette.lime, fontFamily: type.demi, fontSize: 14 },
  disabled: { opacity: 0.35 },
  pressed: { opacity: 0.84, transform: [{ scale: 0.99 }] },
  footerHint: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, textAlign: 'center', marginTop: 7 },
  welcome: { paddingTop: 28 },
  brandMark: { width: 62, height: 62, borderRadius: 21, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center', marginBottom: 22 },
  welcomeKicker: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.8 },
  welcomeTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 35, lineHeight: 39, letterSpacing: -1.5, marginTop: 8, maxWidth: 340 },
  welcomeBody: { color: palette.muted, fontFamily: type.regular, fontSize: 13, lineHeight: 20, marginTop: 13, maxWidth: 350 },
  promise: { marginTop: 30, gap: 12 },
  promiseRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  promiseIcon: { width: 38, height: 38, borderRadius: 13, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  promiseText: { flex: 1, color: palette.ink, fontFamily: type.medium, fontSize: 12, lineHeight: 17 },
  safetyCopy: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 15, marginTop: 28 },
  eyebrow: { color: palette.limeDark, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.5 },
  questionTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 29, lineHeight: 33, letterSpacing: -1, marginTop: 8 },
  questionBody: { color: palette.muted, fontFamily: type.regular, fontSize: 12.5, lineHeight: 19, marginTop: 9, marginBottom: 18 },
  choices: { gap: 9 },
  choice: { minHeight: 70, padding: 14, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, flexDirection: 'row', alignItems: 'center', gap: 11 },
  choiceActive: { borderColor: palette.limeDark, backgroundColor: '#F1F8E5' },
  choiceIcon: { width: 38, height: 38, borderRadius: 13, backgroundColor: '#EEF1EA', alignItems: 'center', justifyContent: 'center' },
  choiceIconActive: { backgroundColor: palette.softLime },
  choiceCopy: { flex: 1 },
  choiceTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 12.5 },
  choiceBody: { color: palette.muted, fontFamily: type.regular, fontSize: 10.5, lineHeight: 15, marginTop: 3 },
  radio: { width: 18, height: 18, borderRadius: 9, borderWidth: 1.5, borderColor: '#AFB7B0', alignItems: 'center', justifyContent: 'center' },
  radioActive: { borderColor: palette.limeDark },
  radioDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: palette.limeDark },
  groupLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.3, marginTop: 24, marginBottom: 9 },
  formCard: { marginTop: 18, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, paddingHorizontal: 15 },
  field: { minHeight: 66, borderBottomWidth: 1, borderBottomColor: palette.line, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  fieldLabel: { color: palette.ink, fontFamily: type.medium, fontSize: 12 },
  inputRow: { flexDirection: 'row', alignItems: 'baseline', gap: 6 },
  numberInput: { minWidth: 76, color: palette.ink, fontFamily: type.demi, fontSize: 18, textAlign: 'right', paddingVertical: 9 },
  fieldUnit: { width: 48, color: palette.muted, fontFamily: type.regular, fontSize: 9.5 },
  textField: { minHeight: 74, justifyContent: 'center' },
  textInput: { color: palette.ink, fontFamily: type.regular, fontSize: 12, paddingVertical: 7 },
  helperText: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 15, marginTop: 8, paddingHorizontal: 3 },
  errorText: { color: palette.coral, fontFamily: type.medium, fontSize: 9.5, lineHeight: 15, marginTop: 8, paddingHorizontal: 3 },
  tip: { flexDirection: 'row', gap: 11, backgroundColor: palette.softLime, padding: 14, borderRadius: radius.md, marginTop: 14 },
  tipText: { flex: 1, color: palette.ink, fontFamily: type.regular, fontSize: 10.5, lineHeight: 16 },
  whyCard: { backgroundColor: palette.paper, borderRadius: radius.md, padding: 16, marginTop: 12 },
  whyTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 12 },
  whyBody: { color: palette.muted, fontFamily: type.regular, fontSize: 10.5, lineHeight: 16, marginTop: 5 },
  planCard: { backgroundColor: palette.forest, borderRadius: radius.lg, padding: 19 },
  planKicker: { color: palette.lime, fontFamily: type.demi, fontSize: 9, letterSpacing: 1.4 },
  planCalories: { color: palette.white, fontFamily: type.demi, fontSize: 39, letterSpacing: -1.5, marginTop: 3 },
  planUnit: { color: '#AEB9B0', fontFamily: type.medium, fontSize: 12 },
  planBasis: { color: '#C7CEC8', fontFamily: type.regular, fontSize: 10.5, lineHeight: 16, marginTop: 3 },
  metricGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 17 },
  planMetric: { width: '48%', backgroundColor: '#243029', borderRadius: radius.sm, padding: 11 },
  planMetricValue: { color: palette.white, fontFamily: type.demi, fontSize: 14 },
  planMetricLabel: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 9, marginTop: 2 },
  methodRow: { borderTopWidth: 1, borderTopColor: '#344138', marginTop: 16, paddingTop: 12 },
  methodLabel: { color: palette.lime, fontFamily: type.demi, fontSize: 8, letterSpacing: 1.1 },
  methodText: { color: '#AEB9B0', fontFamily: type.regular, fontSize: 9.5, lineHeight: 14, marginTop: 3 },
  warningText: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 15, marginTop: 8, paddingHorizontal: 3 },
  setupCard: { backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 14, marginBottom: 10 },
  setupRow: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  setupIcon: { width: 40, height: 40, borderRadius: 14, backgroundColor: palette.softLime, alignItems: 'center', justifyContent: 'center' },
  setupMessage: { color: palette.muted, fontFamily: type.regular, fontSize: 9.5, lineHeight: 15, marginTop: 10 },
  setupButton: { height: 39, borderRadius: radius.sm, backgroundColor: '#EDF2E7', alignItems: 'center', justifyContent: 'center', marginTop: 11 },
  setupButtonText: { color: palette.forest, fontFamily: type.demi, fontSize: 11 },
});
