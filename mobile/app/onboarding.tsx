import { type Href, useRouter } from 'expo-router';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Keyboard,
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
  GhostButton,
  GlassFooter,
  MacroChip,
  Metric,
  PrimaryButton,
  Reveal,
  Ring,
  Screen,
  SectionTitle,
  Tap,
  Well,
} from '@/src/components/ui';
import { healthSetupCopy, healthSnapshotHasSamples, syncNativeHealth } from '@/src/lib/health';
import { useReducedMotion } from '@/src/lib/accessibility';
import { calculatePersonalTargets, goalLabel } from '@/src/lib/personalization';
import { ensureSpeechPermission, speechAvailable } from '@/src/lib/speech';
import { targetWeightError } from '@/src/lib/weight';
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
  Goals,
  MainChallenge,
  PersonalProfile,
  PrimaryGoal,
  WorkoutPreference,
} from '@/src/types';

/**
 * First run is deliberately short: promise -> goal -> essentials -> plan.
 * Everything else in this file remains available as progressive profile
 * setup, but none of it is allowed to delay the first useful log.
 */
const TOTAL_STEPS = 8;
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

const workoutChoices: Choice<WorkoutPreference>[] = [
  { value: 'gym', title: 'Gym training', body: 'Weights, machines or structured classes.', icon: 'dumbbell' },
  { value: 'walking', title: 'Walking and cardio', body: 'Steps and simple aerobic sessions fit best.', icon: 'steps' },
  { value: 'home', title: 'Home workouts', body: 'Bodyweight, yoga or dumbbells at home.', icon: 'home' },
  { value: 'mixed', title: 'A flexible mix', body: 'Use whichever option fits the day.', icon: 'spark' },
  { value: 'restarting', title: 'I am restarting', body: 'Begin below the standard target and build momentum.', icon: 'restart' },
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

const paceChoices: Choice<GoalPace>[] = [
  { value: 'gentle', title: 'Gentle', body: 'Smaller changes that are easier to sustain.', icon: 'heart' },
  { value: 'steady', title: 'Steady', body: 'A balanced pace for most people.', icon: 'trend' },
  { value: 'ambitious', title: 'Ambitious', body: 'A faster pace within conservative safety limits.', icon: 'flame' },
];

const activityChoices: Choice<ActivityLevel>[] = [
  { value: 'mostly-seated', title: 'Mostly seated', body: 'Desk-based days with little planned movement.', icon: 'home' },
  { value: 'lightly-active', title: 'Lightly active', body: 'Some walking or one to two sessions each week.', icon: 'steps' },
  { value: 'active', title: 'Active', body: 'Regular movement or three to five weekly sessions.', icon: 'steps' },
  { value: 'very-active', title: 'Very active', body: 'Physical work or demanding training most days.', icon: 'dumbbell' },
];

const experienceChoices: Choice<ExperienceLevel>[] = [
  { value: 'new', title: 'New to training', body: 'Prioritize simple movements and gradual progression.', icon: 'spark' },
  { value: 'some', title: 'Some experience', body: 'Use familiar movements with clear progression.', icon: 'trend' },
  { value: 'experienced', title: 'Experienced', body: 'Expose advanced set controls and performance detail.', icon: 'trophy' },
];

const toneChoices: Choice<CoachingTone>[] = [
  { value: 'gentle', title: 'Gentle', body: 'Encouraging, practical and never guilt-based.', icon: 'heart' },
  { value: 'direct', title: 'Direct', body: 'Short, clear next actions without extra explanation.', icon: 'target' },
  { value: 'data-led', title: 'Data-led', body: 'Show the trend and explain why a suggestion changed.', icon: 'chart' },
];

export default function OnboardingScreen() {
  const router = useRouter();
  const {
    applyHealthSnapshot,
    data,
    hydrated,
    savePersonalization,
    updateCoachMemory,
    updateGoals,
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
  const [bowl] = useState(data.estimation.bowlMl ? String(data.estimation.bowlMl) : '');
  const [healthBusy, setHealthBusy] = useState(false);
  const [healthMessage, setHealthMessage] = useState('');
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [voiceMessage, setVoiceMessage] = useState('');
  const [voiceReady, setVoiceReady] = useState(false);
  const [customGoals, setCustomGoals] = useState<Goals | null>(null);
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
    trainingDays: numericAllowZero(trainingDays),
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
  const targetError = targetWeightError(primaryGoal, numeric(weight), numeric(targetWeight));

  const canContinue = step === 0
    || (step === 1 && Boolean(primaryGoal))
    || (step === 2
      && Boolean(equationSex)
      && inRange(age, 18, 90)
      && inRange(height, 125, 230)
      && inRange(weight, 35, 250)
      && !targetError)
    || (step === 3 && Boolean(goalPace) && Boolean(activityLevel))
    || (step === 4
      && Boolean(workoutPreference)
      && Boolean(experienceLevel)
      && inRangeAllowZero(trainingDays, 0, 7)
      && inRange(availableMinutes, 10, 240))
    || (step === 5 && Boolean(dietStyle) && inRange(mealsPerDay, 1, 8))
    || (step === 6 && Boolean(mainChallenge) && Boolean(coachingTone))
    || step >= 7;

  useEffect(() => {
    scroller.current?.scrollTo({ y: 0, animated: false });
  }, [step]);

  async function checkVoice() {
    setVoiceBusy(true);
    setVoiceMessage('');
    try {
      if (!speechAvailable()) {
        setVoiceReady(false);
        setVoiceMessage('This device cannot transcribe speech. Use the keyboard to log.');
        return;
      }
      const permission = await ensureSpeechPermission();
      setVoiceReady(permission.granted);
      setVoiceMessage(
        permission.granted
          ? 'Microphone and dictation are ready. Vigorly never receives your audio; iOS or Android handles speech recognition and may require a connection.'
          : permission.canAskAgain
            ? 'Vigorly needs microphone and speech access to log by voice.'
            : 'Microphone or speech access is off. Turn it on in Settings to log by voice.',
      );
    } catch (error) {
      setVoiceReady(false);
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
      const hasSamples = healthSnapshotHasSamples(snapshot);
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
    if (customGoals) updateGoals(customGoals);
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
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          onTouchMove={Keyboard.dismiss}
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
                  setTargetWeight={setTargetWeight}
                  setWeight={setWeight}
                  targetError={targetError}
                  targetWeight={targetWeight}
                  weight={weight}
                />
              ) : step === 3 ? (
                <MovementProfile
                  activityLevel={activityLevel}
                  goalPace={goalPace}
                  setActivityLevel={setActivityLevel}
                  setGoalPace={setGoalPace}
                />
              ) : step === 4 ? (
                <TrainingProfile
                  availableMinutes={availableMinutes}
                  experienceLevel={experienceLevel}
                  setAvailableMinutes={setAvailableMinutes}
                  setExperienceLevel={setExperienceLevel}
                  setTrainingDays={setTrainingDays}
                  setWorkoutPreference={setWorkoutPreference}
                  trainingDays={trainingDays}
                  workoutPreference={workoutPreference}
                />
              ) : step === 5 ? (
                <FoodProfile
                  allergies={allergies}
                  dietStyle={dietStyle}
                  mealsPerDay={mealsPerDay}
                  setAllergies={setAllergies}
                  setDietStyle={setDietStyle}
                  setMealsPerDay={setMealsPerDay}
                />
              ) : step === 6 ? (
                <CoachingProfile
                  coachingTone={coachingTone}
                  injuries={injuries}
                  mainChallenge={mainChallenge}
                  setCoachingTone={setCoachingTone}
                  setInjuries={setInjuries}
                  setMainChallenge={setMainChallenge}
                />
              ) : (
                <PlanPreview
                  customized={Boolean(customGoals)}
                  goal={primaryGoal}
                  healthBusy={healthBusy}
                  healthMessage={healthMessage}
                  onConnectHealth={() => void connectHealth()}
                  onCustomizeGoals={setCustomGoals}
                  onResetGoals={() => setCustomGoals(null)}
                  onTestVoice={() => void checkVoice()}
                  plan={preview.plan}
                  goals={customGoals ?? preview.goals}
                  recommendedGoals={preview.goals}
                  setupDetail={setup.detail}
                  setupTitle={setup.title}
                  currentWeightKg={profile.weightKg}
                  targetWeightKg={profile.targetWeightKg}
                  voiceBusy={voiceBusy}
                  voiceMessage={voiceMessage}
                  voiceReady={voiceReady}
                />
              )}
          </View>
        </ScrollView>

        <GlassFooter>
          {step < TOTAL_STEPS - 1 ? (
            <>
              <PrimaryButton
                disabled={!canContinue}
                icon={step === 0 ? 'spark' : 'chevron'}
                label={step === 0 ? 'Build my plan' : 'Continue'}
                onPress={() => {
                  Keyboard.dismiss();
                  setStep((current) => current + 1);
                }}
              />
              {step === 0 ? (
                <GhostButton compact label="Log now, set up later" onPress={() => void finish()} />
              ) : null}
            </>
          ) : (
            <PrimaryButton icon="check" label="Start with this plan" onPress={() => void finish()} />
          )}
          {step === TOTAL_STEPS - 1 ? (
            <Text style={styles.footerHint}>Next, create an account to sync this plan—or choose guest mode for this device.</Text>
          ) : null}
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
  const reducedMotion = useReducedMotion();
  const [width, setWidth] = useState(0);
  const ratio = (step + 1) / total;
  const position = useSharedValue(0);

  useEffect(() => {
    position.value = reducedMotion ? ratio : withSpring(ratio, motion.enter);
  }, [position, ratio, reducedMotion]);

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
      <Text numberOfLines={2} style={styles.body}>{body}</Text>
    </View>
  );
}

function Welcome() {
  return (
    <>
      <Reveal>
        <View style={styles.welcomeVisual}>
          <View style={styles.welcomeHalo} />
          <Ring size={156} thickness={8} value={0.76}>
            <View style={styles.brandCore}>
              <BrandMark size={58} />
            </View>
          </Ring>
          <View style={[styles.floatingGlyph, styles.floatingFood]}>
            <Glyph color={palette.protein} name="bowl" size={17} />
          </View>
          <View style={[styles.floatingGlyph, styles.floatingTrain]}>
            <Glyph color={palette.info} name="dumbbell" size={17} />
          </View>
          <View style={[styles.floatingGlyph, styles.floatingProgress]}>
            <Glyph color={palette.fat} name="trend" size={17} />
          </View>
        </View>
        <Text style={styles.eyebrow}>VIGORLY</Text>
        <Text style={styles.title}>A plan built around you.</Text>
        <Text style={styles.body}>Four quick steps. Every target stays editable.</Text>
      </Reveal>

      <Reveal index={1} style={styles.promiseGrid}>
        <PromiseRow icon="target" text="Your targets" />
        <PromiseRow icon="dumbbell" text="Your training" />
        <PromiseRow icon="bowl" text="Your food" />
      </Reveal>

      <Reveal index={2} style={styles.group}>
        <Text style={styles.fineprint}>
          For adults 18+. Estimates support general wellness and do not replace medical or dietetic care.
        </Text>
      </Reveal>
    </>
  );
}

function PromiseRow({ icon, text: copy }: { icon: GlyphName; text: string }) {
  return (
    <View style={styles.promiseRow}>
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
        <Reveal index={startIndex + index} key={choice.value} style={styles.choiceItem}>
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
        <Glyph color={active ? palette.onLime : palette.inkMid} name={choice.icon} size={22} />
      </View>
      <View style={styles.choiceCopy}>
        <Text style={styles.choiceTitle}>{choice.title}</Text>
      </View>
      <SelectMark active={active} />
    </Tap>
  );
}

/** The selected-state check. Springs open so picking an answer feels physical. */
function SelectMark({ active }: { active: boolean }) {
  const reducedMotion = useReducedMotion();
  const on = useSharedValue(active ? 1 : 0);

  useEffect(() => {
    on.value = reducedMotion
      ? (active ? 1 : 0)
      : withSpring(active ? 1 : 0, motion.bouncy);
  }, [active, on, reducedMotion]);

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
  setTargetWeight,
  setWeight,
  targetError,
  targetWeight,
  weight,
}: {
  age: string;
  equationSex: EquationSex | null;
  height: string;
  setAge: (value: string) => void;
  setEquationSex: (value: EquationSex) => void;
  setHeight: (value: string) => void;
  setTargetWeight: (value: string) => void;
  setWeight: (value: string) => void;
  targetError: string | null;
  targetWeight: string;
  weight: string;
}) {
  return (
    <>
      <Reveal>
        <StepIntro
          eyebrow="YOUR STARTING POINT"
          title="Calculate energy from your body—not a generic 2,200."
          body="Current weight estimates today’s energy needs. Target weight gives the plan a direction and stays editable as your needs change."
        />
      </Reveal>

      <Reveal index={1}>
        <Card>
          <View style={styles.grid}>
            <NumberField label="Age" onChange={setAge} placeholder="29" unit="years" value={age} />
            <NumberField label="Height" onChange={setHeight} placeholder="172" unit="cm" value={height} />
            <NumberField label="Current weight" onChange={setWeight} placeholder="74" unit="kg" value={weight} />
            <NumberField label="Target weight" onChange={setTargetWeight} placeholder="68" unit="kg" value={targetWeight} />
          </View>
          <View style={styles.noteRow}>
            <Glyph color={targetError ? palette.danger : palette.inkLow} name={targetError ? 'alert' : 'target'} size={14} />
            <Text style={targetError ? styles.errorText : styles.helperText}>
              {targetError ?? 'Your calorie target uses your current body; the adjustment and progress guidance use the target direction.'}
            </Text>
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

function MovementProfile({
  activityLevel,
  goalPace,
  setActivityLevel,
  setGoalPace,
}: {
  activityLevel: ActivityLevel | null;
  goalPace: GoalPace | null;
  setActivityLevel: (value: ActivityLevel) => void;
  setGoalPace: (value: GoalPace) => void;
}) {
  return (
    <>
      <Reveal>
        <StepIntro
          eyebrow="YOUR WEEK"
          title="Match the plan to real life."
          body="Your usual movement sets maintenance energy. Your preferred pace controls how quickly targets change."
        />
      </Reveal>
      <SectionTitle title="Usual activity" />
      <ChoiceList choices={activityChoices} onSelect={setActivityLevel} selected={activityLevel} />
      <SectionTitle title="Preferred pace" />
      <ChoiceList choices={paceChoices} onSelect={setGoalPace} selected={goalPace} />
    </>
  );
}

function TrainingProfile({
  availableMinutes,
  experienceLevel,
  setAvailableMinutes,
  setExperienceLevel,
  setTrainingDays,
  setWorkoutPreference,
  trainingDays,
  workoutPreference,
}: {
  availableMinutes: string;
  experienceLevel: ExperienceLevel | null;
  setAvailableMinutes: (value: string) => void;
  setExperienceLevel: (value: ExperienceLevel) => void;
  setTrainingDays: (value: string) => void;
  setWorkoutPreference: (value: WorkoutPreference) => void;
  trainingDays: string;
  workoutPreference: WorkoutPreference | null;
}) {
  return (
    <>
      <Reveal>
        <StepIntro
          eyebrow="TRAINING FIT"
          title="Build a week you can finish."
          body="Availability controls weekly minutes and strength frequency. Zero training days is valid during recovery."
        />
      </Reveal>
      <SectionTitle title="Best fit" />
      <ChoiceList choices={workoutChoices} onSelect={setWorkoutPreference} selected={workoutPreference} />
      <SectionTitle title="Experience" />
      <ChoiceList choices={experienceChoices} onSelect={setExperienceLevel} selected={experienceLevel} />
      <Reveal index={2}>
        <Card>
          <View style={styles.grid}>
            <NumberField label="Training days" onChange={setTrainingDays} placeholder="3" unit="/ week" value={trainingDays} />
            <NumberField label="Time available" onChange={setAvailableMinutes} placeholder="30" unit="min / session" value={availableMinutes} />
          </View>
        </Card>
      </Reveal>
    </>
  );
}

function FoodProfile({
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
          eyebrow="FOOD FIT"
          title="Make nutrition recognizably yours."
          body="This changes meal examples and safety warnings. It never guesses that an allergy is only a preference."
        />
      </Reveal>
      <ChoiceList choices={dietChoices} onSelect={setDietStyle} selected={dietStyle} />
      <Reveal index={2}>
        <Card>
          <View style={styles.grid}>
            <NumberField label="Meals per day" onChange={setMealsPerDay} placeholder="3" unit="meals" value={mealsPerDay} />
          </View>
          <ProfileTextField
            label="Allergies or foods to avoid"
            onChange={setAllergies}
            placeholder="e.g. peanuts, dairy"
            value={allergies}
          />
        </Card>
      </Reveal>
    </>
  );
}

function CoachingProfile({
  coachingTone,
  injuries,
  mainChallenge,
  setCoachingTone,
  setInjuries,
  setMainChallenge,
}: {
  coachingTone: CoachingTone | null;
  injuries: string;
  mainChallenge: MainChallenge | null;
  setCoachingTone: (value: CoachingTone) => void;
  setInjuries: (value: string) => void;
  setMainChallenge: (value: MainChallenge) => void;
}) {
  return (
    <>
      <Reveal>
        <StepIntro
          eyebrow="COACHING FIT"
          title="Choose what help should feel like."
          body="Your challenge sets the first priority. Injuries are remembered so training suggestions can stay conservative."
        />
      </Reveal>
      <SectionTitle title="Main challenge" />
      <ChoiceList choices={challengeChoices} onSelect={setMainChallenge} selected={mainChallenge} />
      <SectionTitle title="Coaching style" />
      <ChoiceList choices={toneChoices} onSelect={setCoachingTone} selected={coachingTone} />
      <Reveal index={2}>
        <Card>
          <ProfileTextField
            label="Injuries or movement limits"
            onChange={setInjuries}
            placeholder="e.g. recovering right knee"
            value={injuries}
          />
          <Text style={styles.helperText}>Vigorly provides general wellness guidance, not injury diagnosis or rehabilitation advice.</Text>
        </Card>
      </Reveal>
    </>
  );
}

function PlanPreview({
  currentWeightKg,
  customized,
  goal,
  goals,
  healthBusy,
  healthMessage,
  onConnectHealth,
  onCustomizeGoals,
  onResetGoals,
  onTestVoice,
  plan,
  recommendedGoals,
  setupDetail,
  setupTitle,
  targetWeightKg,
  voiceBusy,
  voiceMessage,
  voiceReady,
}: {
  currentWeightKg?: number;
  customized: boolean;
  goal: PrimaryGoal | null;
  goals: Goals;
  healthBusy: boolean;
  healthMessage: string;
  onConnectHealth: () => void;
  onCustomizeGoals: (goals: Goals) => void;
  onResetGoals: () => void;
  onTestVoice: () => void;
  plan: ReturnType<typeof calculatePersonalTargets>['plan'];
  recommendedGoals: Goals;
  setupDetail: string;
  setupTitle: string;
  targetWeightKg?: number;
  voiceBusy: boolean;
  voiceMessage: string;
  voiceReady: boolean;
}) {
  const [editingTargets, setEditingTargets] = useState(false);

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
          <View style={styles.planHead}>
            <Text style={styles.planLabel}>{customized ? 'CUSTOM DAILY TARGETS' : 'RECOMMENDED DAILY TARGETS'}</Text>
            <Tap
              accessibilityLabel="Customize recommended targets"
              onPress={() => setEditingTargets((current) => !current)}
              scaleTo={0.95}>
              <Text style={styles.planEdit}>{editingTargets ? 'Close' : 'Customize'}</Text>
            </Tap>
          </View>
          <View style={styles.planValueRow}>
            <CountUp style={styles.planValue} value={goals.calories} />
            <Text style={styles.planUnit}>kcal</Text>
          </View>
          <Text style={styles.planSummary}>
            {customized ? 'Adjusted to your preferences. You can reset to the calculated recommendation at any time.' : plan.summary}
          </Text>

          {currentWeightKg && targetWeightKg ? (
            <Well style={styles.targetJourney}>
              <View>
                <Text style={styles.methodLabel}>CURRENT</Text>
                <Text style={styles.targetValue}>{currentWeightKg.toFixed(1)} kg</Text>
              </View>
              <Glyph color={palette.inkLow} name="chevron" size={15} />
              <View style={styles.targetEnd}>
                <Text style={styles.methodLabel}>TARGET</Text>
                <Text style={styles.targetValue}>{targetWeightKg.toFixed(1)} kg</Text>
              </View>
            </Well>
          ) : null}

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

      {editingTargets ? (
        <Reveal index={2} style={styles.group}>
          <TargetEditor
            goals={goals}
            onApply={(next) => {
              onCustomizeGoals(next);
              setEditingTargets(false);
            }}
            onReset={() => {
              onResetGoals();
              setEditingTargets(false);
            }}
            recommended={recommendedGoals}
          />
        </Reveal>
      ) : null}

      <Reveal index={editingTargets ? 3 : 2} style={styles.metrics}>
        <Metric
          accent={palette.info}
          detail={customized ? 'Your target' : 'Recommended'}
          icon="water"
          label="Water"
          progress={1}
          value={`${(goals.waterMl / 1000).toFixed(1)} L`}
        />
        <Metric
          accent={palette.lime}
          detail={customized ? 'Your target' : 'Recommended'}
          icon="steps"
          label="Steps"
          progress={1}
          value={goals.steps.toLocaleString()}
        />
        <Metric
          accent={palette.fat}
          detail={customized ? 'Your weekly target' : 'Recommended / week'}
          icon="timer"
          label="Training"
          progress={1}
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
          body="Your device handles dictation; Vigorly never receives the audio."
          busy={voiceBusy}
          button={voiceReady ? 'Voice configured' : 'Turn on voice'}
          configured={voiceReady}
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
  configured = false,
  icon,
  message,
  onPress,
  title,
}: {
  body: string;
  busy: boolean;
  button: string;
  configured?: boolean;
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
          <Text numberOfLines={2} style={styles.setupBody}>{body}</Text>
        </View>
      </View>
      {message ? (
        <Well style={styles.setupMessage}>
          <Text style={styles.setupMessageText}>{message}</Text>
        </Well>
      ) : null}
      <Tap
        accessibilityLabel={button}
        disabled={busy || configured}
        onPress={onPress}
        scaleTo={0.975}
        style={[styles.setupAction, configured && styles.setupActionConfigured]}>
        {busy
          ? <ActivityIndicator color={palette.lime} size="small" />
          : (
            <View style={styles.setupActionContent}>
              {configured ? <Glyph color={palette.lime} name="check" size={15} /> : null}
              <Text style={styles.setupActionLabel}>{button}</Text>
            </View>
          )}
      </Tap>
    </Card>
  );
}

function TargetEditor({
  goals,
  onApply,
  onReset,
  recommended,
}: {
  goals: Goals;
  onApply: (goals: Goals) => void;
  onReset: () => void;
  recommended: Goals;
}) {
  const [draft, setDraft] = useState<Record<keyof Goals, string>>(() => ({
    calories: String(goals.calories),
    protein: String(goals.protein),
    carbs: String(goals.carbs),
    fat: String(goals.fat),
    waterMl: String(goals.waterMl),
    steps: String(goals.steps),
    weeklyWorkoutMinutes: String(goals.weeklyWorkoutMinutes),
    strengthDays: String(goals.strengthDays),
  }));

  function field(key: keyof Goals) {
    return (value: string) => setDraft((current) => ({ ...current, [key]: value }));
  }

  function apply() {
    const parsedStrengthDays = Number(draft.strengthDays);
    onApply({
      calories: Math.max(500, Number(draft.calories) || recommended.calories),
      protein: Math.max(10, Number(draft.protein) || recommended.protein),
      carbs: Math.max(20, Number(draft.carbs) || recommended.carbs),
      fat: Math.max(20, Number(draft.fat) || recommended.fat),
      waterMl: Math.max(250, Number(draft.waterMl) || recommended.waterMl),
      steps: Math.max(500, Number(draft.steps) || recommended.steps),
      weeklyWorkoutMinutes: Math.max(
        0,
        draft.weeklyWorkoutMinutes.trim() !== '' && Number.isFinite(Number(draft.weeklyWorkoutMinutes))
          ? Number(draft.weeklyWorkoutMinutes)
          : recommended.weeklyWorkoutMinutes,
      ),
      strengthDays: Math.min(7, Math.max(
        0,
        draft.strengthDays.trim() !== '' && Number.isFinite(parsedStrengthDays)
          ? parsedStrengthDays
          : recommended.strengthDays,
      )),
    });
  }

  return (
    <Card>
      <Text style={styles.cardTitle}>Customize the recommendation</Text>
      <Text style={styles.cardBody}>These values become your active targets. You can change them again from Profile & goals.</Text>
      <View style={[styles.grid, styles.targetGrid]}>
        <NumberField label="Calories" onChange={field('calories')} placeholder={String(recommended.calories)} unit="kcal" value={draft.calories} />
        <NumberField label="Protein" onChange={field('protein')} placeholder={String(recommended.protein)} unit="g" value={draft.protein} />
        <NumberField label="Carbs" onChange={field('carbs')} placeholder={String(recommended.carbs)} unit="g" value={draft.carbs} />
        <NumberField label="Fat" onChange={field('fat')} placeholder={String(recommended.fat)} unit="g" value={draft.fat} />
        <NumberField label="Water" onChange={field('waterMl')} placeholder={String(recommended.waterMl)} unit="ml" value={draft.waterMl} />
        <NumberField label="Steps" onChange={field('steps')} placeholder={String(recommended.steps)} unit="steps" value={draft.steps} />
        <NumberField label="Weekly training" onChange={field('weeklyWorkoutMinutes')} placeholder={String(recommended.weeklyWorkoutMinutes)} unit="min" value={draft.weeklyWorkoutMinutes} />
        <NumberField label="Strength days" onChange={field('strengthDays')} placeholder={String(recommended.strengthDays)} unit="/ week" value={draft.strengthDays} />
      </View>
      <View style={styles.editorActions}>
        <GhostButton compact label="Reset recommendation" onPress={onReset} />
        <PrimaryButton icon="check" label="Use these targets" onPress={apply} />
      </View>
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

function ProfileTextField({
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
    <View style={[styles.textField, styles.stacked]}>
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
          returnKeyType="done"
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

function numericAllowZero(value: string) {
  const parsed = Number(value);
  return value.trim() !== '' && Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

function inRange(value: string, min: number, max: number) {
  const parsed = numeric(value);
  return parsed != null && parsed >= min && parsed <= max;
}

function inRangeAllowZero(value: string, min: number, max: number) {
  const parsed = numericAllowZero(value);
  return parsed != null && parsed >= min && parsed <= max;
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
  welcomeVisual: {
    height: 214,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    marginTop: -8,
    marginBottom: space.sm,
  },
  welcomeHalo: {
    position: 'absolute',
    width: 188,
    height: 188,
    borderRadius: 94,
    backgroundColor: 'rgba(198,255,60,0.06)',
  },
  brandCore: {
    width: 112,
    height: 112,
    borderRadius: 38,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceHi,
    borderWidth: 1,
    borderColor: palette.lineHi,
  },
  floatingGlyph: {
    position: 'absolute',
    width: 42,
    height: 42,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.lineHi,
    ...shadow.card,
  },
  floatingFood: { left: 26, top: 44 },
  floatingTrain: { right: 30, top: 72 },
  floatingProgress: { left: 54, bottom: 22 },
  promiseGrid: { flexDirection: 'row', gap: space.sm, marginTop: space.lg },
  promiseRow: {
    flex: 1,
    minHeight: 104,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 9,
    padding: 10,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: palette.line,
    backgroundColor: palette.surface,
  },
  promiseIcon: {
    width: 40,
    height: 40,
    borderRadius: 14,
    backgroundColor: palette.limeSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  promiseText: { ...text.micro, fontFamily: text.value.fontFamily, color: palette.ink, textAlign: 'center' },
  fineprint: { ...text.caption, fontSize: 10.5, color: palette.inkLow },

  /* choice cards */
  choices: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  choiceItem: { flexBasis: '47%', flexGrow: 1 },
  choice: {
    minHeight: 132,
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 10,
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
  choiceCopy: { minHeight: 40, justifyContent: 'flex-end' },
  choiceTitle: { ...text.row, color: palette.ink, maxWidth: 112 },
  mark: {
    position: 'absolute',
    top: 12,
    right: 12,
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
  planHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: space.sm },
  planLabel: { ...text.label, color: palette.lime },
  planEdit: { ...text.value, fontSize: 12, color: palette.lime, paddingVertical: 6 },
  planValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 8, marginTop: 12 },
  planValue: { ...text.hero, ...tabular, color: palette.ink },
  planUnit: { ...text.caption, color: palette.inkMid },
  planSummary: { ...text.caption, color: palette.inkMid, marginTop: 8 },
  targetJourney: { flexDirection: 'row', alignItems: 'center', gap: space.sm, marginTop: 14 },
  targetEnd: { flex: 1, alignItems: 'flex-end' },
  targetValue: { ...text.value, ...tabular, color: palette.ink, marginTop: 3 },
  macros: { flexDirection: 'row', gap: 7, marginTop: 18 },
  method: { marginTop: 14 },
  methodLabel: { ...text.label, fontSize: 8.5, letterSpacing: 1.2, color: palette.inkLow },
  methodText: { ...text.caption, fontSize: 11, color: palette.inkMid, marginTop: 4 },
  metrics: { flexDirection: 'row', gap: 8, marginTop: 10 },
  targetGrid: { marginTop: 16 },
  editorActions: { gap: 8, marginTop: 16 },

  /* permission cards */
  setupHead: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  setupBody: { ...text.caption, color: palette.inkMid, marginTop: 3 },
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
  setupActionConfigured: { borderColor: `${palette.lime}55`, backgroundColor: palette.limeSoft },
  setupActionContent: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  setupActionLabel: { ...text.value, color: palette.lime },

  footerHint: { ...text.caption, fontSize: 11, color: palette.inkLow, textAlign: 'center' },
});
