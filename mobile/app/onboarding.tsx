import { type Href, useRouter } from 'expo-router';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import Animated, {
  interpolateColor,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';

import { BrandMark } from '@/src/components/brand-mark';
import { Glyph, type GlyphName } from '@/src/components/glyph';
import {
  Bar,
  Card,
  CountUp,
  GlassFooter,
  MacroChip,
  Metric,
  PrimaryButton,
  Reveal,
  Screen,
  SectionTitle,
  Tap,
  Well,
} from '@/src/components/ui';
import { healthSetupCopy, syncNativeHealth } from '@/src/lib/health';
import { calculatePersonalTargets, goalLabel } from '@/src/lib/personalization';
import { ensureSpeechPermission, speechAvailable } from '@/src/lib/speech';
import { useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { macroColor, motion, palette, radius, shadow, space, tabular, text } from '@/src/theme';
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
const THUMB = 9;

type Choice<T extends string> = {
  value: T;
  title: string;
  body: string;
  icon: GlyphName;
};

const goalChoices: Choice<PrimaryGoal>[] = [
  { value: 'lose-fat', title: 'Lose fat, steadily', body: 'Use a measured deficit and adjust from my trend—not a crash plan.', icon: 'chart' },
  { value: 'build-muscle', title: 'Build muscle and strength', body: 'Prioritize training, protein and a controlled energy surplus.', icon: 'dumbbell' },
  { value: 'maintain', title: 'Maintain my weight', body: 'Keep energy stable while improving food quality and routine.', icon: 'heart' },
  { value: 'improve-fitness', title: 'Improve fitness and energy', body: 'Build cardio, strength, sleep and daily movement together.', icon: 'steps' },
  { value: 'build-habits', title: 'Become consistent first', body: 'Start with targets I can repeat on an ordinary week.', icon: 'spark' },
];

const sexChoices: Choice<EquationSex>[] = [
  { value: 'female', title: 'Female equation', body: 'Uses the female Mifflin–St Jeor constant.', icon: 'user' },
  { value: 'male', title: 'Male equation', body: 'Uses the male Mifflin–St Jeor constant.', icon: 'user' },
  { value: 'neutral', title: 'Use a midpoint', body: 'Less precise, but does not require choosing either equation.', icon: 'scale' },
];

const paceChoices: Choice<GoalPace>[] = [
  { value: 'gentle', title: 'Gentle', body: 'Smaller calorie change; easiest to sustain and assess.', icon: 'steps' },
  { value: 'steady', title: 'Steady', body: 'A practical middle ground for most routines.', icon: 'trend' },
  { value: 'ambitious', title: 'Ambitious', body: 'Larger change; monitor hunger, recovery and performance.', icon: 'flame' },
];

const activityChoices: Choice<ActivityLevel>[] = [
  { value: 'mostly-seated', title: 'Mostly seated', body: 'Desk-based day with little walking outside planned exercise.', icon: 'keyboard' },
  { value: 'lightly-active', title: 'Some daily movement', body: 'Regular errands or walking, but much of the day is seated.', icon: 'steps' },
  { value: 'active', title: 'Active most days', body: 'A mobile job, frequent walking or regular training.', icon: 'trend' },
  { value: 'very-active', title: 'Very active', body: 'Physical work, long training sessions or high daily movement.', icon: 'flame' },
];

const workoutChoices: Choice<WorkoutPreference>[] = [
  { value: 'gym', title: 'Gym training', body: 'Weights, machines or structured classes.', icon: 'dumbbell' },
  { value: 'walking', title: 'Walking and cardio', body: 'Steps and simple aerobic sessions fit best.', icon: 'steps' },
  { value: 'home', title: 'Home workouts', body: 'Bodyweight, yoga or dumbbells at home.', icon: 'home' },
  { value: 'mixed', title: 'A flexible mix', body: 'Use whichever option fits the day.', icon: 'spark' },
  { value: 'restarting', title: 'I am restarting', body: 'Begin below the standard target and build momentum.', icon: 'restart' },
];

const experienceChoices: Choice<ExperienceLevel>[] = [
  { value: 'new', title: 'New', body: 'I need clear, simple starting points.', icon: 'book' },
  { value: 'some', title: 'Some experience', body: 'I know the basics but want consistency.', icon: 'muscle' },
  { value: 'experienced', title: 'Experienced', body: 'I track performance and can handle more detail.', icon: 'trophy' },
];

const dietChoices: Choice<DietStyle>[] = [
  { value: 'home-indian', title: 'Mostly home-cooked Indian', body: 'Roti, rice, dal, sabzi, curd and recipe-based portions.', icon: 'home' },
  { value: 'vegetarian', title: 'Vegetarian', body: 'Dairy and/or eggs are fine; keep protein practical.', icon: 'bowl' },
  { value: 'vegan', title: 'Vegan', body: 'Use plant-only suggestions and protein sources.', icon: 'sun' },
  { value: 'mixed', title: 'Mixed diet', body: 'Home food, cafés, takeout, meat and vegetarian meals.', icon: 'spark' },
  { value: 'high-protein', title: 'Already protein-first', body: 'I regularly use eggs, meat, paneer, tofu, whey or soy.', icon: 'muscle' },
];

const challengeChoices: Choice<MainChallenge>[] = [
  { value: 'portions', title: 'Portion uncertainty', body: 'I struggle to estimate home food and serving sizes.', icon: 'scale' },
  { value: 'protein', title: 'Getting enough protein', body: 'I need realistic protein choices across the day.', icon: 'muscle' },
  { value: 'cravings', title: 'Cravings or snacking', body: 'Evenings or stressful days usually derail me.', icon: 'flame' },
  { value: 'time', title: 'Time and planning', body: 'Suggestions must work on busy days.', icon: 'timer' },
  { value: 'consistency', title: 'Staying consistent', body: 'I start well and then stop tracking.', icon: 'calendar' },
];

const toneChoices: Choice<CoachingTone>[] = [
  { value: 'gentle', title: 'Gentle', body: 'Encourage me without guilt or pressure.', icon: 'heart' },
  { value: 'direct', title: 'Direct', body: 'Give me the clearest next action.', icon: 'target' },
  { value: 'data-led', title: 'Data-led', body: 'Explain the numbers and tradeoffs briefly.', icon: 'chart' },
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
  const scroller = useRef<ScrollView>(null);

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

  useEffect(() => {
    scroller.current?.scrollTo({ y: 0, animated: false });
  }, [step]);

  async function checkVoice() {
    setVoiceBusy(true);
    setVoiceMessage('');
    try {
      if (!speechAvailable()) {
        setVoiceMessage('This device cannot transcribe speech. Use the keyboard to log.');
        return;
      }
      const permission = await ensureSpeechPermission();
      setVoiceMessage(
        permission.granted
          ? 'Microphone and on-device dictation are ready. Speech is transcribed on this phone — no account and no connection needed.'
          : permission.canAskAgain
            ? 'Vigorly needs microphone and speech access to log by voice.'
            : 'Microphone or speech access is off. Turn it on in Settings to log by voice.',
      );
    } catch (error) {
      setVoiceMessage(
        `${error instanceof Error ? error.message : 'Voice setup could not be checked.'} Typed logging always works.`,
      );
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
    return (
      <Screen edges={['top', 'bottom']}>
        <View style={styles.loading}>
          <ActivityIndicator color={palette.lime} />
        </View>
      </Screen>
    );
  }

  return (
    <Screen edges={['top', 'bottom']}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.fill}>
        <View style={styles.topBar}>
          {step > 0 ? (
            <Tap
              accessibilityLabel="Go back"
              hitSlop={10}
              onPress={() => setStep((current) => current - 1)}
              scaleTo={0.9}
              style={styles.back}>
              <Glyph color={palette.ink} name="chevronLeft" size={18} />
            </Tap>
          ) : <View style={styles.back} />}
          <Text style={styles.stepCount}>STEP {step + 1} OF {TOTAL_STEPS}</Text>
          <View style={styles.back} />
        </View>

        <StepProgress step={step} total={TOTAL_STEPS} />

        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          ref={scroller}
          showsVerticalScrollIndicator={false}
          style={styles.fill}>
          <View key={step}>
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
          </View>
        </ScrollView>

        <GlassFooter>
          {step < TOTAL_STEPS - 1 ? (
            <PrimaryButton
              disabled={!canContinue}
              icon={step === 0 ? 'spark' : 'chevron'}
              label={step === 0 ? 'Build my plan' : 'Continue'}
              onPress={() => setStep((current) => current + 1)}
            />
          ) : (
            <PrimaryButton icon="check" label="Save plan and continue" onPress={() => void finish()} />
          )}
          {step === 8 ? <Text style={styles.footerHint}>Bowl size is optional and can be added later.</Text> : null}
        </GlassFooter>
      </KeyboardAvoidingView>
    </Screen>
  );
}

/* ------------------------------------------------------------------ *
 * Progress
 * ------------------------------------------------------------------ */

/** Slim track under the header. The lime head springs forward on every step. */
function StepProgress({ step, total }: { step: number; total: number }) {
  const [width, setWidth] = useState(0);
  const ratio = (step + 1) / total;
  const position = useSharedValue(0);

  useEffect(() => {
    position.value = withSpring(ratio, motion.enter);
  }, [position, ratio]);

  const head = useAnimatedStyle(() => {
    const limit = Math.max(0, width - THUMB);
    return {
      transform: [{ translateX: Math.min(Math.max(position.value * width - THUMB / 2, 0), limit) }],
    };
  });

  return (
    <View
      accessibilityLabel={`Step ${step + 1} of ${total}`}
      accessibilityRole="progressbar"
      style={styles.trackWrap}>
      <View onLayout={(event) => setWidth(event.nativeEvent.layout.width)} style={styles.track}>
        <Bar delay={0} height={4} value={ratio} />
        {width > 0 ? <Animated.View style={[styles.trackHead, head]} /> : null}
      </View>
    </View>
  );
}

/* ------------------------------------------------------------------ *
 * Step scaffolding
 * ------------------------------------------------------------------ */

function StepIntro({ body, eyebrow, title }: { body: string; eyebrow: string; title: string }) {
  return (
    <View style={styles.intro}>
      <Text style={styles.eyebrow}>{eyebrow}</Text>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.body}>{body}</Text>
    </View>
  );
}

function Welcome() {
  return (
    <>
      <Reveal>
        <View style={styles.brandRow}>
          <BrandMark size={54} />
        </View>
        <Text style={styles.eyebrow}>CALORIE LENS</Text>
        <Text style={styles.title}>Your plan should know who it is for.</Text>
        <Text style={styles.body}>
          A few honest answers will set your calories, macros, movement targets and coaching priorities. Every answer stays editable.
        </Text>
      </Reveal>

      <Reveal index={1} style={styles.group}>
        <Card padded={false} style={styles.promiseCard}>
          <PromiseRow icon="chart" text="Targets calculated from your body, routine and goal" />
          <PromiseRow icon="dumbbell" text="Suggestions shaped around your time and training style" />
          <PromiseRow icon="bowl" last text="Food guidance that respects how you actually eat" />
        </Card>
      </Reveal>

      <Reveal index={2} style={styles.group}>
        <Text style={styles.fineprint}>
          For adults 18+. Estimates support general wellness and do not replace medical or dietetic care.
        </Text>
      </Reveal>
    </>
  );
}

function PromiseRow({ icon, last, text: copy }: { icon: GlyphName; last?: boolean; text: string }) {
  return (
    <View style={[styles.promiseRow, !last && styles.promiseBorder]}>
      <View style={styles.promiseIcon}>
        <Glyph color={palette.lime} name={icon} size={17} />
      </View>
      <Text style={styles.promiseText}>{copy}</Text>
    </View>
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
    <>
      <Reveal>
        <StepIntro body={body} eyebrow={eyebrow} title={title} />
      </Reveal>
      <ChoiceList choices={choices} onSelect={onSelect} selected={selected} />
    </>
  );
}

/* ------------------------------------------------------------------ *
 * Choice cards
 * ------------------------------------------------------------------ */

function ChoiceList<T extends string>({
  choices,
  onSelect,
  selected,
  startIndex = 1,
}: {
  choices: Choice<T>[];
  onSelect: (value: T) => void;
  selected: T | null;
  startIndex?: number;
}) {
  return (
    <View style={styles.choices}>
      {choices.map((choice, index) => (
        <Reveal index={startIndex + index} key={choice.value}>
          <ChoiceCard active={selected === choice.value} choice={choice} onSelect={onSelect} />
        </Reveal>
      ))}
    </View>
  );
}

function ChoiceCard<T extends string>({
  active,
  choice,
  onSelect,
}: {
  active: boolean;
  choice: Choice<T>;
  onSelect: (value: T) => void;
}) {
  return (
    <Tap
      accessibilityLabel={`${choice.title}. ${choice.body}`}
      onPress={() => onSelect(choice.value)}
      scaleTo={0.985}
      style={[styles.choice, active && styles.choiceOn]}>
      <View style={[styles.choiceIcon, active && styles.choiceIconOn]}>
        <Glyph color={active ? palette.onLime : palette.inkMid} name={choice.icon} size={18} />
      </View>
      <View style={styles.choiceCopy}>
        <Text style={styles.choiceTitle}>{choice.title}</Text>
        <Text style={styles.choiceBody}>{choice.body}</Text>
      </View>
      <SelectMark active={active} />
    </Tap>
  );
}

/** The selected-state check. Springs open so picking an answer feels physical. */
function SelectMark({ active }: { active: boolean }) {
  const on = useSharedValue(active ? 1 : 0);

  useEffect(() => {
    on.value = withSpring(active ? 1 : 0, motion.bouncy);
  }, [active, on]);

  const shell = useAnimatedStyle(() => ({
    backgroundColor: interpolateColor(on.value, [0, 1], [palette.surfaceLo, palette.lime]),
    borderColor: interpolateColor(on.value, [0, 1], [palette.lineHi, palette.lime]),
  }));

  const mark = useAnimatedStyle(() => ({
    opacity: on.value,
    transform: [{ scale: 0.4 + on.value * 0.6 }],
  }));

  return (
    <Animated.View style={[styles.mark, shell]}>
      <Animated.View style={mark}>
        <Glyph color={palette.onLime} name="check" size={12} strokeWidth={2.4} />
      </Animated.View>
    </Animated.View>
  );
}

/* ------------------------------------------------------------------ *
 * Steps
 * ------------------------------------------------------------------ */

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
    <>
      <Reveal>
        <StepIntro
          eyebrow="YOUR STARTING POINT"
          title="Calculate energy from your body—not a generic 2,200."
          body="Age, height and weight estimate resting energy. The equation option is about physiology used by the formula, not gender identity."
        />
      </Reveal>

      <Reveal index={1}>
        <Card>
          <View style={styles.grid}>
            <NumberField label="Age" onChange={setAge} placeholder="29" unit="years" value={age} />
            <NumberField label="Height" onChange={setHeight} placeholder="172" unit="cm" value={height} />
            <NumberField label="Current weight" onChange={setWeight} placeholder="74" unit="kg" value={weight} />
          </View>
        </Card>
      </Reveal>

      <Reveal index={2}>
        <SectionTitle title="Energy equation" />
      </Reveal>
      <ChoiceList choices={sexChoices} onSelect={setEquationSex} selected={equationSex} startIndex={3} />
    </>
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
    <>
      <Reveal>
        <StepIntro
          eyebrow="DIRECTION, NOT A DEADLINE"
          title="How quickly should the starting plan move?"
          body={goal ? goalNotes[goal] : 'The pace controls the starting calorie adjustment.'}
        />
      </Reveal>

      <Reveal index={1}>
        <Card>
          <View style={styles.grid}>
            <NumberField
              label="Target weight"
              onChange={setTargetWeight}
              placeholder="Optional"
              unit="kg"
              value={targetWeight}
            />
          </View>
          {targetError ? (
            <View style={styles.noteRow}>
              <Glyph color={palette.danger} name="alert" size={14} />
              <Text style={styles.errorText}>
                For this goal, choose a target in the intended direction or leave it blank.
              </Text>
            </View>
          ) : (
            <View style={styles.noteRow}>
              <Glyph color={palette.inkLow} name="info" size={14} />
              <Text style={styles.helperText}>
                A target weight helps the coach frame progress, but it does not create a deadline.
              </Text>
            </View>
          )}
        </Card>
      </Reveal>

      <Reveal index={2}>
        <SectionTitle title="Starting pace" />
      </Reveal>
      <ChoiceList choices={paceChoices} onSelect={setPace} selected={pace} startIndex={3} />
    </>
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
    <>
      <Reveal>
        <StepIntro
          eyebrow="A PLAN THAT FITS THE WEEK"
          title="What training can you realistically repeat?"
          body="These answers set weekly minutes, strength frequency and the kind of suggestions you receive."
        />
      </Reveal>
      <ChoiceList choices={workoutChoices} onSelect={setPreference} selected={preference} />

      <Reveal index={6}>
        <SectionTitle title="Your experience" />
      </Reveal>
      <ChoiceList choices={experienceChoices} onSelect={setExperience} selected={experience} startIndex={7} />

      <Reveal index={10} style={styles.group}>
        <Card>
          <View style={styles.grid}>
            <NumberField
              label="Days available"
              onChange={setTrainingDays}
              placeholder="3"
              unit="/ week"
              value={trainingDays}
            />
            <NumberField
              label="Time per session"
              onChange={setAvailableMinutes}
              placeholder="30"
              unit="minutes"
              value={availableMinutes}
            />
          </View>
        </Card>
      </Reveal>
    </>
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
    <>
      <Reveal>
        <StepIntro
          eyebrow="YOUR FOOD, NOT A TEMPLATE"
          title="What does eating normally look like?"
          body="The coach uses this to choose relevant protein sources, meal examples and portion advice."
        />
      </Reveal>
      <ChoiceList choices={dietChoices} onSelect={setDietStyle} selected={dietStyle} />

      <Reveal index={6} style={styles.group}>
        <Card>
          <View style={styles.grid}>
            <NumberField
              label="Meals most days"
              onChange={setMealsPerDay}
              placeholder="3"
              unit="/ day"
              value={mealsPerDay}
            />
          </View>
          <View style={styles.stacked}>
            <TextField
              label="Allergies or foods to avoid"
              onChange={setAllergies}
              placeholder="e.g. peanuts, shellfish"
              value={allergies}
            />
          </View>
          <View style={styles.noteRow}>
            <Glyph color={palette.inkLow} name="info" size={14} />
            <Text style={styles.helperText}>Separate multiple items with commas. Leave blank if none.</Text>
          </View>
        </Card>
      </Reveal>
    </>
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
    <>
      <Reveal>
        <StepIntro
          eyebrow="WHAT USUALLY GETS IN THE WAY"
          title="Make the advice useful on difficult days."
          body="Your main challenge determines which gap the app calls out first."
        />
      </Reveal>
      <ChoiceList choices={challengeChoices} onSelect={setChallenge} selected={challenge} />

      <Reveal index={6}>
        <SectionTitle title="How should the coach speak?" />
      </Reveal>
      <ChoiceList choices={toneChoices} onSelect={setTone} selected={tone} startIndex={7} />

      <Reveal index={10} style={styles.group}>
        <Card>
          <TextField
            label="Injuries or movement limits"
            onChange={setInjuries}
            placeholder="e.g. sensitive left knee"
            value={injuries}
          />
          <View style={styles.noteRow}>
            <Glyph color={palette.inkLow} name="info" size={14} />
            <Text style={styles.helperText}>
              The app avoids suggesting around a noted limitation, but it cannot diagnose or rehabilitate an injury.
            </Text>
          </View>
        </Card>
      </Reveal>
    </>
  );
}

function Calibration({ bowl, setBowl }: { bowl: string; setBowl: (value: string) => void }) {
  return (
    <>
      <Reveal>
        <StepIntro
          eyebrow="ONE USEFUL MEASUREMENT"
          title="Make home-food estimates less random."
          body="Your usual bowl size changes calculations for dal, rajma, rice and other foods logged by volume."
        />
      </Reveal>

      <Reveal index={1}>
        <Card>
          <View style={styles.grid}>
            <NumberField label="Your usual bowl" onChange={setBowl} placeholder="200" unit="ml" value={bowl} />
          </View>
          <Well style={styles.tip}>
            <Glyph color={palette.lime} name="water" size={17} />
            <Text style={styles.tipText}>
              Fill the bowl with water once and pour it into a measuring jug. You can skip this and add it later.
            </Text>
          </Well>
        </Card>
      </Reveal>

      <Reveal index={2} style={styles.group}>
        <Card>
          <Text style={styles.cardTitle}>What will stay approximate?</Text>
          <Text style={styles.cardBody}>
            Oil, recipes and restaurant portions still vary. The app shows a range and its assumptions before saving.
          </Text>
        </Card>
      </Reveal>
    </>
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
    <>
      <Reveal>
        <StepIntro
          eyebrow="YOUR STARTING PLAN"
          title={`${goalLabel(goal ?? undefined)}—with numbers that have a reason.`}
          body="These are starting estimates. Your 2–4 week trend, energy and training performance should guide later adjustments."
        />
      </Reveal>

      <Reveal index={1}>
        <Card glow raised>
          <Text style={styles.planLabel}>DAILY TARGETS</Text>
          <View style={styles.planValueRow}>
            <CountUp style={styles.planValue} value={goals.calories} />
            <Text style={styles.planUnit}>kcal</Text>
          </View>
          <Text style={styles.planSummary}>{plan.summary}</Text>

          <View style={styles.macros}>
            <MacroChip color={macroColor.protein} goal={goals.protein} label="Protein" value={goals.protein} />
            <MacroChip color={macroColor.carbs} goal={goals.carbs} label="Carbs" value={goals.carbs} />
            <MacroChip color={macroColor.fat} goal={goals.fat} label="Fat" value={goals.fat} />
          </View>

          <Well style={styles.method}>
            <Text style={styles.methodLabel}>METHOD</Text>
            <Text style={styles.methodText}>
              {plan.method} · maintenance ~{plan.maintenanceCalories} kcal
            </Text>
          </Well>
        </Card>
      </Reveal>

      <Reveal index={2} style={styles.metrics}>
        <Metric
          accent={palette.info}
          detail="Daily target"
          icon="water"
          label="Water"
          value={`${(goals.waterMl / 1000).toFixed(1)} L`}
        />
        <Metric
          accent={palette.lime}
          detail="Daily target"
          icon="steps"
          label="Steps"
          value={goals.steps.toLocaleString()}
        />
        <Metric
          accent={palette.fat}
          detail="Per week"
          icon="timer"
          label="Training"
          value={`${goals.weeklyWorkoutMinutes} min`}
        />
      </Reveal>

      {plan.warnings.length ? (
        <Reveal index={3} style={styles.group}>
          <Card>
            {plan.warnings.map((warning, index) => (
              <View key={warning} style={[styles.warningRow, index > 0 && styles.warningSpacing]}>
                <Glyph color={palette.warn} name="info" size={14} />
                <Text style={styles.warningText}>{warning}</Text>
              </View>
            ))}
          </Card>
        </Reveal>
      ) : null}

      <Reveal index={4}>
        <SectionTitle title="Optional connection checks" />
      </Reveal>
      <Reveal index={5}>
        <SetupCard
          body={setupDetail}
          busy={healthBusy}
          button="Connect"
          icon="heart"
          message={healthMessage}
          onPress={onConnectHealth}
          title={setupTitle}
        />
      </Reveal>
      <Reveal index={6} style={styles.group}>
        <SetupCard
          body="Checks microphone and speech permission. Dictation is transcribed on this device, so voice logging is free and needs no account."
          busy={voiceBusy}
          button="Test voice"
          icon="mic"
          message={voiceMessage}
          onPress={onTestVoice}
          title="Voice logging"
        />
      </Reveal>
    </>
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
    <Card>
      <View style={styles.setupHead}>
        <View style={styles.setupIcon}>
          <Glyph color={palette.lime} name={icon} size={18} />
        </View>
        <View style={styles.choiceCopy}>
          <Text style={styles.choiceTitle}>{title}</Text>
          <Text style={styles.choiceBody}>{body}</Text>
        </View>
      </View>
      {message ? (
        <Well style={styles.setupMessage}>
          <Text style={styles.setupMessageText}>{message}</Text>
        </Well>
      ) : null}
      <Tap
        accessibilityLabel={button}
        disabled={busy}
        onPress={onPress}
        scaleTo={0.975}
        style={styles.setupAction}>
        {busy
          ? <ActivityIndicator color={palette.lime} size="small" />
          : <Text style={styles.setupActionLabel}>{button}</Text>}
      </Tap>
    </Card>
  );
}

/* ------------------------------------------------------------------ *
 * Fields
 * ------------------------------------------------------------------ */

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
  const [focused, setFocused] = useState(false);
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label.toUpperCase()}</Text>
      <View style={[styles.shell, focused && styles.shellOn]}>
        <TextInput
          accessibilityLabel={label}
          keyboardType="decimal-pad"
          onBlur={() => setFocused(false)}
          onChangeText={(next) => onChange(next.replace(/[^\d.]/g, ''))}
          onFocus={() => setFocused(true)}
          placeholder={placeholder}
          placeholderTextColor={palette.inkLow}
          selectionColor={palette.lime}
          selectTextOnFocus
          style={styles.numberInput}
          value={value}
        />
        <Text numberOfLines={1} style={styles.fieldUnit}>{unit}</Text>
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
  const [focused, setFocused] = useState(false);
  return (
    <View style={styles.textField}>
      <Text style={styles.fieldLabel}>{label.toUpperCase()}</Text>
      <View style={[styles.shell, focused && styles.shellOn]}>
        <TextInput
          accessibilityLabel={label}
          autoCapitalize="sentences"
          onBlur={() => setFocused(false)}
          onChangeText={onChange}
          onFocus={() => setFocused(true)}
          placeholder={placeholder}
          placeholderTextColor={palette.inkLow}
          selectionColor={palette.lime}
          style={styles.textInput}
          value={value}
        />
      </View>
    </View>
  );
}

/* ------------------------------------------------------------------ *
 * Validation helpers — unchanged
 * ------------------------------------------------------------------ */

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
  fill: { flex: 1 },
  loading: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  /* header + progress */
  topBar: {
    height: 46,
    paddingHorizontal: space.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
  },
  back: {
    width: 36,
    height: 36,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepCount: { ...text.label, ...tabular, flex: 1, color: palette.inkLow, textAlign: 'center' },
  trackWrap: { paddingHorizontal: space.md, paddingBottom: space.md },
  track: { height: 4, justifyContent: 'center' },
  trackHead: {
    ...shadow.glowSoft,
    position: 'absolute',
    top: -2.5,
    left: 0,
    width: THUMB,
    height: THUMB,
    borderRadius: THUMB / 2,
    backgroundColor: palette.lime,
  },

  content: { paddingHorizontal: space.md, paddingBottom: space.tabClearance },

  /* step intro */
  intro: { marginBottom: space.lg },
  eyebrow: { ...text.label, color: palette.lime, marginBottom: 10 },
  title: { ...text.title, color: palette.ink },
  body: { ...text.body, color: palette.inkMid, marginTop: 10 },
  group: { marginTop: space.md },

  /* welcome */
  brandRow: { marginTop: space.sm, marginBottom: space.lg },
  promiseCard: { paddingHorizontal: 14 },
  promiseRow: { minHeight: 62, flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 12 },
  promiseBorder: { borderBottomWidth: 1, borderBottomColor: palette.line },
  promiseIcon: {
    width: 34,
    height: 34,
    borderRadius: 12,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  promiseText: { ...text.body, flex: 1, color: palette.ink },
  fineprint: { ...text.caption, fontSize: 10.5, color: palette.inkLow },

  /* choice cards */
  choices: { gap: 10 },
  choice: {
    minHeight: 78,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 14,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surface,
  },
  choiceOn: { borderColor: palette.lime, backgroundColor: palette.limeSoft },
  choiceIcon: {
    width: 40,
    height: 40,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surfaceLo,
    alignItems: 'center',
    justifyContent: 'center',
  },
  choiceIconOn: { borderColor: palette.lime, backgroundColor: palette.lime },
  choiceCopy: { flex: 1 },
  choiceTitle: { ...text.row, color: palette.ink },
  choiceBody: { ...text.caption, color: palette.inkMid, marginTop: 3 },
  mark: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* fields */
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  field: { flexBasis: '44%', flexGrow: 1, gap: 8 },
  textField: { gap: 8 },
  stacked: { marginTop: 14 },
  fieldLabel: { ...text.label, color: palette.inkLow },
  shell: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    minHeight: 58,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surfaceLo,
    paddingHorizontal: 14,
  },
  shellOn: { borderColor: palette.lime },
  numberInput: { ...text.headline, ...tabular, flex: 1, color: palette.ink, paddingVertical: 14 },
  fieldUnit: { ...text.caption, fontSize: 11, color: palette.inkLow },
  textInput: { ...text.body, flex: 1, color: palette.ink, paddingVertical: 14 },

  /* inline notes */
  noteRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, marginTop: 14 },
  warningRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  helperText: { ...text.caption, flex: 1, color: palette.inkLow },
  errorText: { ...text.caption, flex: 1, color: palette.danger },
  warningText: { ...text.caption, flex: 1, color: palette.inkMid },
  warningSpacing: { marginTop: 12 },

  /* calibration */
  tip: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginTop: 14 },
  tipText: { ...text.caption, flex: 1, color: palette.inkMid },
  cardTitle: { ...text.section, color: palette.ink },
  cardBody: { ...text.caption, color: palette.inkMid, marginTop: 6 },

  /* plan payoff */
  planLabel: { ...text.label, color: palette.lime },
  planValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 8, marginTop: 12 },
  planValue: { ...text.hero, ...tabular, color: palette.ink },
  planUnit: { ...text.caption, color: palette.inkMid },
  planSummary: { ...text.caption, color: palette.inkMid, marginTop: 8 },
  macros: { flexDirection: 'row', gap: 7, marginTop: 18 },
  method: { marginTop: 14 },
  methodLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1.2, color: palette.inkLow },
  methodText: { ...text.caption, fontSize: 11, color: palette.inkMid, marginTop: 4 },
  metrics: { flexDirection: 'row', gap: 8, marginTop: 10 },

  /* permission cards */
  setupHead: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  setupIcon: {
    width: 40,
    height: 40,
    borderRadius: 14,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  setupMessage: { marginTop: 12 },
  setupMessageText: { ...text.caption, color: palette.inkMid },
  setupAction: {
    minHeight: 46,
    marginTop: 14,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surfaceHi,
    alignItems: 'center',
    justifyContent: 'center',
  },
  setupActionLabel: { ...text.value, color: palette.lime },

  footerHint: { ...text.caption, fontSize: 11, color: palette.inkLow, textAlign: 'center' },
});
