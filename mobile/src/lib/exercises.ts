import { EXTENDED_EXERCISES } from '@/src/lib/exercise-catalog-extra';

/**
 * Built-in exercise library.
 *
 * Modeled on the structure of dedicated strength loggers (Hevy and similar):
 * every exercise carries its target muscles, equipment, logging kind,
 * step-by-step instructions, coaching tips, an animated demo reference, and a
 * MET estimate from the 2024 Adult Compendium of Physical Activities used for
 * transparent session-energy estimates.
 */

export type MuscleGroup =
  | 'chest'
  | 'back'
  | 'shoulders'
  | 'traps'
  | 'biceps'
  | 'triceps'
  | 'forearms'
  | 'quads'
  | 'hamstrings'
  | 'glutes'
  | 'calves'
  | 'core'
  | 'full body'
  | 'cardio';

export type Equipment =
  | 'barbell'
  | 'dumbbell'
  | 'machine'
  | 'cable'
  | 'bodyweight'
  | 'kettlebell'
  | 'band'
  | 'other';

export type ExerciseKind = 'weight-reps' | 'reps-only' | 'duration';

export type FigureTemplate =
  | 'squat'
  | 'front-squat'
  | 'goblet-squat'
  | 'hinge'
  | 'good-morning'
  | 'lunge'
  | 'step-up'
  | 'bench-press'
  | 'incline-press'
  | 'fly'
  | 'pushup'
  | 'dip'
  | 'overhead-press'
  | 'lateral-raise'
  | 'front-raise'
  | 'rear-fly'
  | 'face-pull'
  | 'upright-row'
  | 'pullup'
  | 'pulldown'
  | 'straight-arm-pulldown'
  | 'bent-row'
  | 'seated-row'
  | 'single-arm-row'
  | 'shrug'
  | 'curl'
  | 'preacher-curl'
  | 'pushdown'
  | 'overhead-triceps'
  | 'skullcrusher'
  | 'wrist-curl'
  | 'leg-press'
  | 'leg-extension'
  | 'leg-curl-lying'
  | 'leg-curl-seated'
  | 'calf-raise'
  | 'calf-seated'
  | 'hip-thrust'
  | 'glute-bridge'
  | 'kickback'
  | 'side-leg-raise'
  | 'swing'
  | 'carry'
  | 'hang'
  | 'plank'
  | 'side-plank'
  | 'crunch'
  | 'leg-raise-lying'
  | 'hanging-knee-raise'
  | 'russian-twist'
  | 'ab-wheel'
  | 'mountain-climber'
  | 'dead-bug'
  | 'back-extension'
  | 'nordic-curl'
  | 'burpee'
  | 'thruster'
  | 'run'
  | 'walk'
  | 'cycle'
  | 'rowing'
  | 'jump-rope'
  | 'stair';

export type FigureGear =
  | 'none'
  | 'barbell'
  | 'barbell-front'
  | 'barbell-back'
  | 'dumbbells'
  | 'kettlebell'
  | 'cable-high'
  | 'cable-low'
  | 'machine'
  | 'bar-overhead'
  | 'bench';

export type Exercise = {
  id: string;
  name: string;
  aliases: string[];
  primaryMuscle: MuscleGroup;
  secondaryMuscles: MuscleGroup[];
  equipment: Equipment;
  kind: ExerciseKind;
  template: FigureTemplate;
  gear: FigureGear;
  /** MET estimate for energy integration (2024 Adult Compendium family codes). */
  met: number;
  instructions: string[];
  tips: string[];
};

function ex(
  id: string,
  name: string,
  primaryMuscle: MuscleGroup,
  secondaryMuscles: MuscleGroup[],
  equipment: Equipment,
  kind: ExerciseKind,
  template: FigureTemplate,
  gear: FigureGear,
  met: number,
  instructions: string[],
  tips: string[],
  aliases: string[] = [],
): Exercise {
  return {
    id,
    name,
    aliases,
    primaryMuscle,
    secondaryMuscles,
    equipment,
    kind,
    template,
    gear,
    met,
    instructions,
    tips,
  };
}

const CORE_EXERCISES: Exercise[] = [
  // ------------------------------------------------------------------ chest
  ex('bench-press', 'Bench Press (Barbell)', 'chest', ['triceps', 'shoulders'], 'barbell', 'weight-reps', 'bench-press', 'barbell', 5.0, [
    'Lie on the bench with your eyes under the bar, feet flat on the floor.',
    'Grip slightly wider than shoulder width, squeeze your shoulder blades together, and unrack.',
    'Lower the bar with control to your mid-chest, elbows about 45–70° from your torso.',
    'Touch lightly, then press back up until your arms are straight over your shoulders.',
  ], [
    'Keep your wrists stacked over your elbows.',
    'Feet stay planted; drive them into the floor as you press.',
    'Use safety pins or a spotter near failure.',
  ], ['flat bench', 'barbell bench']),
  ex('incline-bench-press', 'Incline Bench Press (Barbell)', 'chest', ['shoulders', 'triceps'], 'barbell', 'weight-reps', 'incline-press', 'barbell', 5.0, [
    'Set the bench to a 30–45° incline and lie back with feet planted.',
    'Grip slightly wider than shoulders and unrack over your upper chest.',
    'Lower the bar to just below your collarbones under control.',
    'Press up and slightly back until your arms lock out.',
  ], [
    'Do not let the bar drift over your face at lockout.',
    'A lower incline hits more chest; steeper shifts to shoulders.',
  ], ['incline barbell press']),
  ex('dumbbell-bench-press', 'Dumbbell Bench Press', 'chest', ['triceps', 'shoulders'], 'dumbbell', 'weight-reps', 'bench-press', 'dumbbells', 5.0, [
    'Sit with dumbbells on your thighs, then lie back and bring them to your chest.',
    'Press the dumbbells up until your arms are straight over your shoulders.',
    'Lower with control until you feel a stretch across your chest.',
    'Press back up without letting the weights collide at the top.',
  ], [
    'Dumbbells allow a deeper, freer path than a bar — control the bottom.',
    'Kick the weights up with your knees to get into position safely.',
  ]),
  ex('incline-dumbbell-press', 'Incline Dumbbell Press', 'chest', ['shoulders', 'triceps'], 'dumbbell', 'weight-reps', 'incline-press', 'dumbbells', 5.0, [
    'Set the bench to 30–45° and sit back with a dumbbell in each hand at shoulder height.',
    'Press both dumbbells up and slightly together until your arms are straight.',
    'Lower under control to the top of your chest.',
  ], [
    'Keep your lower back lightly on the bench — no big arch.',
  ]),
  ex('machine-chest-press', 'Chest Press (Machine)', 'chest', ['triceps', 'shoulders'], 'machine', 'weight-reps', 'bench-press', 'machine', 3.5, [
    'Adjust the seat so the handles line up with your mid-chest.',
    'Grip the handles, keep your back on the pad, and press forward to full extension.',
    'Return with control until your hands are near your chest.',
  ], [
    'Do not bounce at the stretched position; stay smooth.',
  ]),
  ex('pushup', 'Push-Up', 'chest', ['triceps', 'shoulders', 'core'], 'bodyweight', 'reps-only', 'pushup', 'none', 3.8, [
    'Start in a straight line from head to heels, hands slightly wider than shoulders.',
    'Lower your chest to just above the floor, elbows about 45° from your body.',
    'Press back up to a full lockout while keeping your hips level.',
  ], [
    'Squeeze your glutes to keep the plank position.',
    'Elevate your hands to make it easier; elevate your feet to make it harder.',
  ], ['press up', 'pushups']),
  ex('cable-chest-fly', 'Cable Fly', 'chest', ['shoulders'], 'cable', 'weight-reps', 'fly', 'cable-high', 3.5, [
    'Set both pulleys to chest height and stand one step forward with a soft bend in the elbows.',
    'Bring your hands together in a wide arc in front of your chest.',
    'Return until you feel a stretch in your chest, keeping the same elbow bend.',
  ], [
    'Think of hugging a barrel — the elbows stay almost fixed.',
  ], ['cable crossover']),
  ex('dumbbell-fly', 'Dumbbell Fly', 'chest', ['shoulders'], 'dumbbell', 'weight-reps', 'fly', 'dumbbells', 3.5, [
    'Lie on a flat bench holding dumbbells over your chest, palms facing each other.',
    'With slightly bent elbows, open your arms in a wide arc until you feel a chest stretch.',
    'Bring the dumbbells back together over your chest along the same arc.',
  ], [
    'Go lighter than you think; the stretched position is the hard part.',
  ]),
  ex('pec-deck', 'Pec Deck (Machine Fly)', 'chest', ['shoulders'], 'machine', 'weight-reps', 'fly', 'machine', 3.5, [
    'Sit with your back on the pad and forearms or hands on the pads at chest height.',
    'Squeeze the pads together in front of your chest.',
    'Open back up with control to a comfortable stretch.',
  ], [
    'Keep your shoulders down and back against the pad.',
  ], ['butterfly machine']),
  ex('chest-dip', 'Dip (Chest)', 'chest', ['triceps', 'shoulders'], 'bodyweight', 'reps-only', 'dip', 'none', 5.0, [
    'Support yourself on parallel bars with straight arms.',
    'Lean your torso forward slightly and lower until your upper arms are about parallel.',
    'Press back up to straight arms without locking out harshly.',
  ], [
    'The forward lean is what shifts the work to your chest.',
    'Add weight with a belt once bodyweight sets feel controlled.',
  ], ['dips']),

  // ------------------------------------------------------------------- back
  ex('deadlift', 'Deadlift (Barbell)', 'back', ['glutes', 'hamstrings', 'traps', 'forearms', 'core'], 'barbell', 'weight-reps', 'hinge', 'barbell', 6.0, [
    'Stand with the bar over your mid-foot, feet hip-width apart.',
    'Hinge down and grip just outside your legs; shins touch the bar.',
    'Brace, flatten your back, and push the floor away to stand up tall.',
    'Keep the bar against your body the whole way up and down.',
  ], [
    'The bar travels in a straight vertical line over mid-foot.',
    'Reset your brace before every rep; do not bounce the plates.',
  ], ['conventional deadlift']),
  ex('romanian-deadlift', 'Romanian Deadlift', 'hamstrings', ['glutes', 'back'], 'barbell', 'weight-reps', 'hinge', 'barbell', 5.0, [
    'Hold the bar at hip height with a shoulder-width grip.',
    'Push your hips back and lower the bar down your thighs with soft knees.',
    'Stop when you feel a strong hamstring stretch (usually mid-shin).',
    'Drive your hips forward to stand back up.',
  ], [
    'This is a hip hinge, not a squat — knees barely change angle.',
    'Keep the bar dragging against your legs.',
  ], ['rdl']),
  ex('pullup', 'Pull-Up', 'back', ['biceps', 'forearms', 'core'], 'bodyweight', 'reps-only', 'pullup', 'bar-overhead', 5.0, [
    'Hang from the bar with an overhand grip slightly wider than shoulders.',
    'Pull your chest toward the bar by driving your elbows down.',
    'Get your chin over the bar, then lower all the way with control.',
  ], [
    'Start each rep from a dead hang for honest reps.',
    'Use a band or assisted machine to build up volume.',
  ], ['pull ups', 'pullups']),
  ex('chinup', 'Chin-Up', 'back', ['biceps', 'forearms'], 'bodyweight', 'reps-only', 'pullup', 'bar-overhead', 5.0, [
    'Hang from the bar with an underhand, shoulder-width grip.',
    'Pull your chin over the bar, leading with your chest.',
    'Lower under control to straight arms.',
  ], [
    'The underhand grip puts more load on your biceps than a pull-up.',
  ], ['chin ups']),
  ex('lat-pulldown', 'Lat Pulldown', 'back', ['biceps', 'forearms'], 'cable', 'weight-reps', 'pulldown', 'machine', 3.5, [
    'Sit with your thighs under the pads and grip the bar wider than shoulders.',
    'Pull the bar down to your upper chest while keeping your torso tall.',
    'Squeeze your lats, then let the bar rise until your arms are fully stretched.',
  ], [
    'Lead with your elbows, not your hands.',
    'Avoid swinging your torso to grind out reps.',
  ], ['pulldown']),
  ex('barbell-row', 'Bent-Over Row (Barbell)', 'back', ['biceps', 'traps', 'core'], 'barbell', 'weight-reps', 'bent-row', 'barbell', 5.0, [
    'Hinge to about 45° with the bar hanging from straight arms.',
    'Pull the bar to your lower ribs, elbows tracking close to your body.',
    'Lower under control without letting your torso rise.',
  ], [
    'Brace hard; your torso angle should not change during the set.',
  ], ['bent over row']),
  ex('dumbbell-row', 'One-Arm Dumbbell Row', 'back', ['biceps', 'traps'], 'dumbbell', 'weight-reps', 'single-arm-row', 'dumbbells', 5.0, [
    'Place one knee and hand on a bench, back flat, dumbbell hanging in the other hand.',
    'Row the dumbbell to your hip, elbow driving toward the ceiling.',
    'Lower to a full stretch without rotating your torso.',
  ], [
    'Pull toward your hip, not your armpit, to feel your lat.',
  ], ['single arm row']),
  ex('seated-cable-row', 'Seated Cable Row', 'back', ['biceps', 'traps'], 'cable', 'weight-reps', 'seated-row', 'machine', 3.5, [
    'Sit tall with knees slightly bent and grab the handle.',
    'Pull the handle to your stomach, squeezing your shoulder blades together.',
    'Let your arms straighten fully forward with control.',
  ], [
    'Keep your torso nearly still — a small stretch forward is fine, no heaving.',
  ], ['cable row']),
  ex('tbar-row', 'T-Bar Row', 'back', ['biceps', 'traps'], 'barbell', 'weight-reps', 'bent-row', 'barbell', 5.0, [
    'Straddle the bar, hinge to about 45°, and grip the handles.',
    'Row the weight to your chest, keeping your back flat.',
    'Lower until your arms are straight.',
  ], [
    'Keep your chest up against the momentum of heavier plates.',
  ]),
  ex('machine-row', 'Seated Row (Machine)', 'back', ['biceps', 'traps'], 'machine', 'weight-reps', 'seated-row', 'machine', 3.5, [
    'Adjust the chest pad so your arms fully extend at the start.',
    'Row the handles to your ribs and squeeze your shoulder blades.',
    'Return to a full stretch with control.',
  ], [
    'The chest pad keeps you honest — no torso swing.',
  ]),
  ex('straight-arm-pulldown', 'Straight-Arm Pulldown', 'back', ['triceps', 'core'], 'cable', 'weight-reps', 'straight-arm-pulldown', 'cable-high', 3.5, [
    'Face a high pulley, grip the bar at shoulder width, and hinge slightly forward.',
    'With nearly straight arms, sweep the bar down to your thighs.',
    'Let it rise back overhead until you feel a lat stretch.',
  ], [
    'Think of your hands as hooks; your lats move the weight.',
  ], ['lat pullover cable']),
  ex('back-extension', 'Back Extension', 'back', ['glutes', 'hamstrings'], 'bodyweight', 'reps-only', 'back-extension', 'none', 3.5, [
    'Set your hips on the pad with your ankles anchored.',
    'Lower your torso toward the floor with a straight back.',
    'Raise your torso until it lines up with your legs — no hyperextension.',
  ], [
    'Squeeze your glutes at the top; hold a plate for load.',
  ], ['hyperextension']),
  ex('rack-pull', 'Rack Pull', 'back', ['glutes', 'traps', 'forearms'], 'barbell', 'weight-reps', 'hinge', 'barbell', 5.0, [
    'Set the bar on pins at or just below knee height.',
    'Hinge, grip, brace, and stand tall with the bar against your thighs.',
    'Lower to the pins under control and reset.',
  ], [
    'Heavier than a deadlift is fine, but keep the back flat.',
  ]),

  // ------------------------------------------------------------- shoulders
  ex('overhead-press', 'Overhead Press (Barbell)', 'shoulders', ['triceps', 'core', 'traps'], 'barbell', 'weight-reps', 'overhead-press', 'barbell', 5.0, [
    'Stand with the bar at your collarbones, hands just outside shoulders.',
    'Brace your core and press the bar straight up, moving your head back slightly.',
    'Lock out overhead with the bar over your mid-foot, then lower to your collarbones.',
  ], [
    'Squeeze your glutes so your lower back does not arch.',
  ], ['ohp', 'military press', 'shoulder press barbell']),
  ex('dumbbell-shoulder-press', 'Shoulder Press (Dumbbell)', 'shoulders', ['triceps'], 'dumbbell', 'weight-reps', 'overhead-press', 'dumbbells', 5.0, [
    'Sit or stand with dumbbells at shoulder height, palms forward.',
    'Press both dumbbells overhead until your arms are straight.',
    'Lower to ear level or slightly below with control.',
  ], [
    'Do not flare the dumbbells far out; press in a slight arc inward.',
  ], ['seated dumbbell press']),
  ex('machine-shoulder-press', 'Shoulder Press (Machine)', 'shoulders', ['triceps'], 'machine', 'weight-reps', 'overhead-press', 'machine', 3.5, [
    'Adjust the seat so the handles start at shoulder height.',
    'Press to a full lockout overhead.',
    'Lower with control back to the start.',
  ], [
    'Keep your back against the pad the whole set.',
  ]),
  ex('lateral-raise', 'Lateral Raise (Dumbbell)', 'shoulders', [], 'dumbbell', 'weight-reps', 'lateral-raise', 'dumbbells', 3.5, [
    'Stand tall with dumbbells at your sides, slight bend in the elbows.',
    'Raise your arms out to the sides until they reach shoulder height.',
    'Lower slowly back to your sides.',
  ], [
    'Lead with your elbows and pour slightly forward — no shrugging.',
    'Light weight, strict form beats heavy swinging.',
  ], ['side raise', 'lat raise']),
  ex('cable-lateral-raise', 'Cable Lateral Raise', 'shoulders', [], 'cable', 'weight-reps', 'lateral-raise', 'cable-low', 3.5, [
    'Stand side-on to a low pulley with the handle in your far hand.',
    'Raise your arm out to shoulder height, elbow slightly bent.',
    'Lower with control against the cable.',
  ], [
    'The cable keeps tension at the bottom where dumbbells go slack.',
  ]),
  ex('front-raise', 'Front Raise (Dumbbell)', 'shoulders', [], 'dumbbell', 'weight-reps', 'front-raise', 'dumbbells', 3.5, [
    'Hold dumbbells in front of your thighs, palms facing you.',
    'Raise one or both arms straight in front to shoulder height.',
    'Lower slowly without swinging.',
  ], [
    'Brace your core so your lower back does not arch.',
  ]),
  ex('rear-delt-fly', 'Rear Delt Fly (Dumbbell)', 'shoulders', ['back', 'traps'], 'dumbbell', 'weight-reps', 'rear-fly', 'dumbbells', 3.5, [
    'Hinge forward to about 45–90° with dumbbells hanging below your chest.',
    'Raise both arms out to the sides, squeezing your rear shoulders.',
    'Lower with control, keeping a slight elbow bend.',
  ], [
    'Keep the movement small and strict; momentum steals the work.',
  ], ['reverse fly', 'bent over fly']),
  ex('reverse-pec-deck', 'Reverse Pec Deck', 'shoulders', ['back', 'traps'], 'machine', 'weight-reps', 'rear-fly', 'machine', 3.5, [
    'Sit facing the pad with the handles in front at shoulder height.',
    'Sweep your arms back in a wide arc as far as your rear delts allow.',
    'Return with control.',
  ], [
    'Keep your elbows slightly bent and at shoulder height.',
  ]),
  ex('face-pull', 'Face Pull (Cable)', 'shoulders', ['traps', 'back'], 'cable', 'weight-reps', 'face-pull', 'cable-high', 3.5, [
    'Set a rope at upper-chest to face height and grab it with thumbs toward you.',
    'Pull the rope toward your face, elbows high and wide.',
    'Finish with your hands beside your ears, then return slowly.',
  ], [
    'Think “show your biceps” at the end position.',
    'Great as a high-rep shoulder-health staple.',
  ]),
  ex('upright-row', 'Upright Row (Barbell)', 'shoulders', ['traps', 'biceps'], 'barbell', 'weight-reps', 'upright-row', 'barbell', 3.5, [
    'Hold the bar in front of your thighs with a grip just inside shoulder width.',
    'Pull the bar up your body to about chest height, elbows leading.',
    'Lower with control.',
  ], [
    'Stop at chest height; pulling higher can pinch the shoulders.',
  ]),

  // ------------------------------------------------------------------ traps
  ex('barbell-shrug', 'Shrug (Barbell)', 'traps', ['forearms'], 'barbell', 'weight-reps', 'shrug', 'barbell', 3.5, [
    'Hold the bar in front of your thighs with straight arms.',
    'Shrug your shoulders straight up toward your ears.',
    'Pause briefly, then lower fully.',
  ], [
    'No rolling — straight up, straight down.',
  ]),
  ex('dumbbell-shrug', 'Shrug (Dumbbell)', 'traps', ['forearms'], 'dumbbell', 'weight-reps', 'shrug', 'dumbbells', 3.5, [
    'Stand with dumbbells at your sides.',
    'Shrug straight up, pause, and lower with control.',
  ], [
    'A one-second pause at the top doubles the value of the set.',
  ]),

  // ----------------------------------------------------------------- biceps
  ex('barbell-curl', 'Bicep Curl (Barbell)', 'biceps', ['forearms'], 'barbell', 'weight-reps', 'curl', 'barbell', 3.5, [
    'Stand holding the bar with an underhand, shoulder-width grip.',
    'Curl the bar to shoulder height while keeping your elbows at your sides.',
    'Lower all the way to straight arms.',
  ], [
    'If your elbows drift forward or your back swings, the weight is too heavy.',
  ]),
  ex('dumbbell-curl', 'Bicep Curl (Dumbbell)', 'biceps', ['forearms'], 'dumbbell', 'weight-reps', 'curl', 'dumbbells', 3.5, [
    'Stand with dumbbells at your sides, palms forward.',
    'Curl one or both dumbbells to shoulder height.',
    'Lower under control to a full stretch.',
  ], [
    'Rotating from neutral to palms-up as you curl adds a supination squeeze.',
  ]),
  ex('hammer-curl', 'Hammer Curl', 'biceps', ['forearms'], 'dumbbell', 'weight-reps', 'curl', 'dumbbells', 3.5, [
    'Hold dumbbells at your sides with palms facing each other.',
    'Curl to shoulder height keeping the neutral grip.',
    'Lower slowly.',
  ], [
    'Hits the brachialis and forearms harder than a standard curl.',
  ]),
  ex('incline-dumbbell-curl', 'Incline Curl (Dumbbell)', 'biceps', ['forearms'], 'dumbbell', 'weight-reps', 'curl', 'dumbbells', 3.5, [
    'Lie back on an incline bench with dumbbells hanging at full stretch.',
    'Curl both dumbbells up without letting your elbows drift forward.',
    'Lower to the stretched position.',
  ], [
    'The stretch at the bottom is the point — do not cut it short.',
  ]),
  ex('preacher-curl', 'Preacher Curl', 'biceps', ['forearms'], 'machine', 'weight-reps', 'preacher-curl', 'machine', 3.5, [
    'Rest your upper arms on the preacher pad and grip the bar.',
    'Curl to the top without lifting your arms off the pad.',
    'Lower until your arms are almost straight.',
  ], [
    'Stop just short of a harsh lockout at the bottom.',
  ]),
  ex('cable-curl', 'Cable Curl', 'biceps', ['forearms'], 'cable', 'weight-reps', 'curl', 'cable-low', 3.5, [
    'Stand facing a low pulley with an underhand grip on the bar.',
    'Curl to shoulder height with elbows pinned at your sides.',
    'Lower with control against the cable.',
  ], [
    'Constant cable tension makes lighter weights feel harder — that is fine.',
  ]),
  ex('concentration-curl', 'Concentration Curl', 'biceps', [], 'dumbbell', 'weight-reps', 'preacher-curl', 'dumbbells', 3.5, [
    'Sit with your elbow braced against your inner thigh, dumbbell hanging.',
    'Curl to your shoulder with zero body swing.',
    'Lower slowly to a full stretch.',
  ], [
    'A strict mind-muscle finisher; go light.',
  ]),

  // ---------------------------------------------------------------- triceps
  ex('triceps-pushdown', 'Triceps Pushdown (Cable)', 'triceps', [], 'cable', 'weight-reps', 'pushdown', 'cable-high', 3.5, [
    'Face a high pulley and grip the bar or rope with elbows at your sides.',
    'Push down until your arms are fully straight.',
    'Let your hands rise to chest height while your elbows stay pinned.',
  ], [
    'Only your forearms should move.',
  ], ['tricep pushdown', 'rope pushdown']),
  ex('overhead-cable-extension', 'Overhead Triceps Extension (Cable)', 'triceps', [], 'cable', 'weight-reps', 'overhead-triceps', 'cable-low', 3.5, [
    'Face away from a low pulley holding a rope behind your head.',
    'Extend your arms overhead until straight.',
    'Bend your elbows to lower the rope behind your head to a deep stretch.',
  ], [
    'The long head of the triceps loves the stretched position.',
  ]),
  ex('skullcrusher', 'Skullcrusher (EZ-Bar)', 'triceps', [], 'barbell', 'weight-reps', 'skullcrusher', 'barbell', 3.5, [
    'Lie on a bench holding the bar over your shoulders.',
    'Bend only your elbows to lower the bar toward your forehead or just behind it.',
    'Extend back to straight arms.',
  ], [
    'Keep your upper arms angled slightly back for constant tension.',
  ], ['lying triceps extension']),
  ex('close-grip-bench', 'Close-Grip Bench Press', 'triceps', ['chest', 'shoulders'], 'barbell', 'weight-reps', 'bench-press', 'barbell', 5.0, [
    'Lie on the bench and grip the bar at about shoulder width.',
    'Lower to your lower chest with elbows tucked close.',
    'Press back up to lockout.',
  ], [
    'Tucked elbows shift the work from chest to triceps.',
  ]),
  ex('bench-dip', 'Bench Dip', 'triceps', ['chest', 'shoulders'], 'bodyweight', 'reps-only', 'dip', 'bench', 3.8, [
    'Place your hands on a bench behind you, legs extended forward.',
    'Lower your hips by bending your elbows to about 90°.',
    'Press back up to straight arms.',
  ], [
    'Keep your hips close to the bench to protect your shoulders.',
  ]),
  ex('overhead-dumbbell-extension', 'Overhead Triceps Extension (Dumbbell)', 'triceps', [], 'dumbbell', 'weight-reps', 'overhead-triceps', 'dumbbells', 3.5, [
    'Hold one dumbbell with both hands overhead.',
    'Lower it behind your head by bending your elbows.',
    'Extend back to straight arms overhead.',
  ], [
    'Keep your elbows pointing forward, not flaring wide.',
  ]),
  ex('diamond-pushup', 'Diamond Push-Up', 'triceps', ['chest', 'shoulders', 'core'], 'bodyweight', 'reps-only', 'pushup', 'none', 3.8, [
    'Set up a push-up with your hands close, thumbs and index fingers near each other.',
    'Lower your chest to your hands with elbows tracking back.',
    'Press to a full lockout.',
  ], [
    'Harder than standard push-ups — expect fewer reps.',
  ]),

  // --------------------------------------------------------------- forearms
  ex('wrist-curl', 'Wrist Curl', 'forearms', [], 'dumbbell', 'weight-reps', 'wrist-curl', 'dumbbells', 3.5, [
    'Sit with your forearms on your thighs, palms up, wrists past your knees.',
    'Let the weight roll toward your fingers, then curl your wrists up.',
    'Lower slowly.',
  ], [
    'High reps (15–25) work well here.',
  ]),
  ex('reverse-wrist-curl', 'Reverse Wrist Curl', 'forearms', [], 'dumbbell', 'weight-reps', 'wrist-curl', 'dumbbells', 3.5, [
    'Sit with forearms on your thighs, palms down.',
    'Raise the backs of your hands upward as far as they go.',
    'Lower with control.',
  ], [
    'Use much less weight than palm-up wrist curls.',
  ]),
  ex('farmers-carry', "Farmer's Carry", 'forearms', ['traps', 'core', 'full body'], 'dumbbell', 'duration', 'carry', 'dumbbells', 5.0, [
    'Pick up a heavy dumbbell or kettlebell in each hand.',
    'Walk tall with short quick steps and locked-in posture.',
    'Set the weights down with a flat back when the time or distance is done.',
  ], [
    'Grip, traps, and core all work — heavier is better if posture holds.',
  ], ['farmers walk']),
  ex('dead-hang', 'Dead Hang', 'forearms', ['back', 'shoulders'], 'bodyweight', 'duration', 'hang', 'bar-overhead', 2.8, [
    'Hang from a bar with straight arms and relaxed but engaged shoulders.',
    'Breathe and hold for time.',
  ], [
    'Builds grip and decompresses the spine after pulling work.',
  ]),

  // ------------------------------------------------------------------ quads
  ex('squat', 'Squat (Barbell)', 'quads', ['glutes', 'hamstrings', 'core'], 'barbell', 'weight-reps', 'squat', 'barbell-back', 5.0, [
    'Rest the bar on your upper back and stand with feet shoulder-width, toes slightly out.',
    'Brace, then sit down and back until your thighs reach at least parallel.',
    'Keep your knees tracking over your toes and your whole foot planted.',
    'Drive back up to standing without letting your chest collapse.',
  ], [
    'Depth you can control beats depth you fall into.',
    'Big breath at the top; exhale through the sticking point.',
  ], ['back squat', 'barbell squat']),
  ex('front-squat', 'Front Squat (Barbell)', 'quads', ['glutes', 'core'], 'barbell', 'weight-reps', 'front-squat', 'barbell-front', 5.0, [
    'Rack the bar on your front shoulders with elbows high.',
    'Squat down keeping your torso as upright as possible.',
    'Drive up while keeping your elbows lifted.',
  ], [
    'If your wrists are tight, use a cross-arm grip or straps.',
  ]),
  ex('goblet-squat', 'Goblet Squat', 'quads', ['glutes', 'core'], 'dumbbell', 'weight-reps', 'goblet-squat', 'kettlebell', 5.0, [
    'Hold a dumbbell or kettlebell vertically against your chest.',
    'Squat between your knees to full comfortable depth.',
    'Stand back up, keeping the weight tight to your body.',
  ], [
    'The best squat teacher — the front load keeps your torso honest.',
  ]),
  ex('leg-press', 'Leg Press (Machine)', 'quads', ['glutes', 'hamstrings'], 'machine', 'weight-reps', 'leg-press', 'machine', 5.0, [
    'Sit in the machine with feet shoulder-width on the platform.',
    'Lower the platform until your knees near your chest without your lower back rounding off the pad.',
    'Press back up, stopping just short of locked knees.',
  ], [
    'Never let your hips curl off the seat at the bottom.',
  ]),
  ex('leg-extension', 'Leg Extension (Machine)', 'quads', [], 'machine', 'weight-reps', 'leg-extension', 'machine', 3.5, [
    'Sit with the pad on your shins and knees lined up with the machine pivot.',
    'Extend your legs to straight and squeeze your quads.',
    'Lower with control through the full range.',
  ], [
    'A one-second squeeze at the top beats extra plates.',
  ]),
  ex('bulgarian-split-squat', 'Bulgarian Split Squat', 'quads', ['glutes', 'hamstrings', 'core'], 'dumbbell', 'weight-reps', 'lunge', 'dumbbells', 5.0, [
    'Stand a stride ahead of a bench and place your rear foot on it.',
    'Lower straight down until your front thigh is about parallel.',
    'Drive through your front foot to stand.',
  ], [
    'Expect a balance challenge at first; hold dumbbells once stable.',
  ], ['rear foot elevated split squat']),
  ex('walking-lunge', 'Walking Lunge', 'quads', ['glutes', 'hamstrings'], 'dumbbell', 'weight-reps', 'lunge', 'dumbbells', 5.0, [
    'Step forward and lower until both knees are near 90°.',
    'Drive through your front foot and step straight into the next lunge.',
    'Alternate legs with a tall torso.',
  ], [
    'Shorter steps hit quads; longer steps hit glutes.',
  ]),
  ex('hack-squat', 'Hack Squat (Machine)', 'quads', ['glutes'], 'machine', 'weight-reps', 'leg-press', 'machine', 5.0, [
    'Set your shoulders under the pads with feet shoulder-width on the platform.',
    'Squat down to at least parallel, knees tracking over toes.',
    'Press back up without locking your knees harshly.',
  ], [
    'Lower foot placement targets quads more.',
  ]),
  ex('step-up', 'Step-Up', 'quads', ['glutes'], 'dumbbell', 'weight-reps', 'step-up', 'dumbbells', 5.0, [
    'Stand facing a knee-high box with dumbbells at your sides.',
    'Step up and drive through the top foot until you stand tall on the box.',
    'Step down with control and repeat.',
  ], [
    'Push through the top leg — do not bounce off the bottom foot.',
  ]),
  ex('bodyweight-squat', 'Squat (Bodyweight)', 'quads', ['glutes'], 'bodyweight', 'reps-only', 'squat', 'none', 3.0, [
    'Stand with feet shoulder-width, arms out for balance.',
    'Squat to full comfortable depth with heels down.',
    'Stand back up tall.',
  ], [
    'Perfect warm-up and high-rep home option.',
  ], ['air squat']),

  // ------------------------------------------------------------- hamstrings
  ex('lying-leg-curl', 'Lying Leg Curl (Machine)', 'hamstrings', [], 'machine', 'weight-reps', 'leg-curl-lying', 'machine', 3.5, [
    'Lie face down with the pad on your lower calves.',
    'Curl your heels toward your glutes.',
    'Lower slowly to a full stretch.',
  ], [
    'Keep your hips pressed into the bench — no rocking.',
  ]),
  ex('seated-leg-curl', 'Seated Leg Curl (Machine)', 'hamstrings', [], 'machine', 'weight-reps', 'leg-curl-seated', 'machine', 3.5, [
    'Sit with the pad on your lower calves and the lap pad snug.',
    'Curl your heels down and under the seat.',
    'Return slowly to a full stretch.',
  ], [
    'The seated angle stretches the hamstrings harder than lying curls.',
  ]),
  ex('stiff-leg-deadlift', 'Stiff-Leg Deadlift (Dumbbell)', 'hamstrings', ['glutes', 'back'], 'dumbbell', 'weight-reps', 'hinge', 'dumbbells', 5.0, [
    'Hold dumbbells in front of your thighs.',
    'Hinge at the hips with nearly straight knees until you feel a deep stretch.',
    'Squeeze your glutes to stand tall.',
  ], [
    'Range of motion comes from your hips, never your lower back rounding.',
  ]),
  ex('good-morning', 'Good Morning (Barbell)', 'hamstrings', ['glutes', 'back'], 'barbell', 'weight-reps', 'good-morning', 'barbell-back', 3.5, [
    'Rest a light bar on your upper back.',
    'With soft knees, hinge your torso toward horizontal.',
    'Drive your hips forward to return to standing.',
  ], [
    'Start very light; this rewards patience and punishes ego.',
  ]),
  ex('nordic-curl', 'Nordic Hamstring Curl', 'hamstrings', ['glutes', 'core'], 'bodyweight', 'reps-only', 'nordic-curl', 'none', 3.8, [
    'Kneel with your ankles anchored and body upright.',
    'Lower your torso forward as slowly as possible using your hamstrings.',
    'Catch yourself with your hands and push back to the start.',
  ], [
    'Even 3–5 slow negatives is a strong hamstring stimulus.',
  ]),

  // ----------------------------------------------------------------- glutes
  ex('hip-thrust', 'Hip Thrust (Barbell)', 'glutes', ['hamstrings', 'quads'], 'barbell', 'weight-reps', 'hip-thrust', 'barbell', 5.0, [
    'Sit with your upper back on a bench, bar over your hips (use a pad).',
    'Plant your feet so your shins are vertical at the top.',
    'Drive your hips up until your body is a flat line, squeezing your glutes.',
    'Lower your hips with control and repeat.',
  ], [
    'Chin tucked, ribs down — finish with glutes, not lower back.',
  ]),
  ex('glute-bridge', 'Glute Bridge', 'glutes', ['hamstrings'], 'bodyweight', 'reps-only', 'glute-bridge', 'none', 3.0, [
    'Lie on your back with knees bent and feet flat near your hips.',
    'Drive your hips up and squeeze your glutes hard at the top.',
    'Lower with control.',
  ], [
    'Pause a full second at the top of every rep.',
  ]),
  ex('cable-kickback', 'Glute Kickback (Cable)', 'glutes', ['hamstrings'], 'cable', 'weight-reps', 'kickback', 'cable-low', 3.5, [
    'Attach an ankle cuff to a low pulley and face the machine.',
    'Kick your leg straight back and slightly up, squeezing your glute.',
    'Return with control without arching your lower back.',
  ], [
    'Small strict range beats a big swinging one.',
  ]),
  ex('sumo-deadlift', 'Sumo Deadlift', 'glutes', ['quads', 'hamstrings', 'back', 'traps'], 'barbell', 'weight-reps', 'hinge', 'barbell', 6.0, [
    'Take a wide stance with toes out, hands gripping inside your knees.',
    'Drop your hips, chest up, and push the floor apart to stand.',
    'Lock out with glutes, then lower under control.',
  ], [
    'More upright torso than conventional — knees out hard.',
  ]),
  ex('kettlebell-swing', 'Kettlebell Swing', 'glutes', ['hamstrings', 'back', 'core', 'cardio'], 'kettlebell', 'weight-reps', 'swing', 'kettlebell', 7.5, [
    'Stand over the kettlebell, hinge, and hike it back between your legs.',
    'Snap your hips forward so the bell floats to chest height.',
    'Let it swing back into the next hinge — arms stay relaxed.',
  ], [
    'It is a hip snap, not an arm lift or a squat.',
  ]),
  ex('hip-abduction', 'Hip Abduction (Machine)', 'glutes', [], 'machine', 'weight-reps', 'side-leg-raise', 'machine', 3.5, [
    'Sit in the machine with the pads outside your knees.',
    'Press your knees apart as far as they go.',
    'Return slowly.',
  ], [
    'Lean slightly forward to bias the upper glutes.',
  ]),

  // ----------------------------------------------------------------- calves
  ex('standing-calf-raise', 'Standing Calf Raise', 'calves', [], 'machine', 'weight-reps', 'calf-raise', 'machine', 3.5, [
    'Stand with the balls of your feet on the platform, heels hanging.',
    'Lower your heels to a deep stretch.',
    'Rise as high as possible onto your toes and pause.',
  ], [
    'Full stretch, full squeeze — no bouncing.',
  ]),
  ex('seated-calf-raise', 'Seated Calf Raise', 'calves', [], 'machine', 'weight-reps', 'calf-seated', 'machine', 3.5, [
    'Sit with the pads on your knees and the balls of your feet on the platform.',
    'Lower to a full stretch, then press up onto your toes.',
  ], [
    'The bent knee shifts work to the soleus — both raises matter.',
  ]),
  ex('single-leg-calf-raise', 'Single-Leg Calf Raise', 'calves', [], 'bodyweight', 'reps-only', 'calf-raise', 'none', 3.0, [
    'Stand on one foot on a step, heel hanging off.',
    'Lower to a deep stretch, then rise all the way up.',
  ], [
    'Hold a dumbbell in one hand when 15 clean reps feel easy.',
  ]),

  // ------------------------------------------------------------------- core
  ex('plank', 'Plank', 'core', ['shoulders', 'glutes'], 'bodyweight', 'duration', 'plank', 'none', 2.8, [
    'Set your forearms under your shoulders and step your feet back.',
    'Form one straight line from head to heels.',
    'Squeeze your glutes, brace your abs, and breathe while holding.',
  ], [
    'Quality beats duration — stop when your hips sag.',
  ]),
  ex('side-plank', 'Side Plank', 'core', ['shoulders', 'glutes'], 'bodyweight', 'duration', 'side-plank', 'none', 2.8, [
    'Lie on your side with your elbow under your shoulder.',
    'Lift your hips so your body is a straight line.',
    'Hold, then switch sides.',
  ], [
    'Push the floor away with your forearm; do not hang on the shoulder.',
  ]),
  ex('crunch', 'Crunch', 'core', [], 'bodyweight', 'reps-only', 'crunch', 'none', 2.8, [
    'Lie on your back with knees bent, hands lightly behind your head.',
    'Curl your ribs toward your hips, lifting your shoulder blades off the floor.',
    'Lower with control.',
  ], [
    'Exhale as you curl up; no pulling on your neck.',
  ]),
  ex('cable-crunch', 'Cable Crunch', 'core', [], 'cable', 'weight-reps', 'crunch', 'cable-high', 3.5, [
    'Kneel below a high pulley holding a rope beside your head.',
    'Crunch your ribs toward your hips against the cable.',
    'Return until your abs reach a full stretch.',
  ], [
    'Hips stay still — the spine flexes, the arms just hold.',
  ]),
  ex('hanging-knee-raise', 'Hanging Knee Raise', 'core', ['forearms'], 'bodyweight', 'reps-only', 'hanging-knee-raise', 'bar-overhead', 3.8, [
    'Hang from a bar with straight arms.',
    'Raise your knees toward your chest, curling your pelvis up at the top.',
    'Lower slowly without swinging.',
  ], [
    'The pelvic curl at the top is what makes it an ab exercise.',
  ]),
  ex('hanging-leg-raise', 'Hanging Leg Raise', 'core', ['forearms'], 'bodyweight', 'reps-only', 'hanging-knee-raise', 'bar-overhead', 3.8, [
    'Hang from a bar and keep your legs straight.',
    'Raise your legs to at least hip height, curling your pelvis.',
    'Lower with total control.',
  ], [
    'Bend your knees slightly if your hamstrings are tight.',
  ]),
  ex('russian-twist', 'Russian Twist', 'core', [], 'bodyweight', 'reps-only', 'russian-twist', 'none', 2.8, [
    'Sit with knees bent, heels lightly down, torso leaned back slightly.',
    'Rotate your ribcage side to side, touching the floor beside your hips.',
  ], [
    'Rotate from the trunk; arms just follow. Hold a weight to progress.',
  ]),
  ex('bicycle-crunch', 'Bicycle Crunch', 'core', [], 'bodyweight', 'reps-only', 'crunch', 'none', 2.8, [
    'Lie on your back with hands behind your head and legs lifted.',
    'Bring opposite elbow and knee together while extending the other leg.',
    'Alternate sides in a smooth rhythm.',
  ], [
    'Slow and controlled beats fast and sloppy.',
  ]),
  ex('ab-wheel-rollout', 'Ab Wheel Rollout', 'core', ['shoulders', 'back'], 'other', 'reps-only', 'ab-wheel', 'none', 3.8, [
    'Kneel holding the wheel under your shoulders.',
    'Roll forward as far as you can keep your lower back flat.',
    'Pull back with your abs and lats to the start.',
  ], [
    'Range grows over weeks — never let the hips sag through.',
  ]),
  ex('mountain-climber', 'Mountain Climber', 'core', ['shoulders', 'cardio'], 'bodyweight', 'duration', 'mountain-climber', 'none', 8.0, [
    'Start in a straight-arm plank.',
    'Drive one knee toward your chest, then switch legs quickly.',
    'Keep your hips level and steady throughout.',
  ], [
    'A cardio-core hybrid — treat it as intervals.',
  ]),
  ex('dead-bug', 'Dead Bug', 'core', [], 'bodyweight', 'reps-only', 'dead-bug', 'none', 2.8, [
    'Lie on your back, arms up, knees bent at 90° over your hips.',
    'Lower one arm and the opposite leg toward the floor while your back stays flat.',
    'Return and switch sides.',
  ], [
    'If your lower back arches off the floor, shorten the range.',
  ]),
  ex('lying-leg-raise', 'Lying Leg Raise', 'core', [], 'bodyweight', 'reps-only', 'leg-raise-lying', 'none', 2.8, [
    'Lie flat with legs straight and hands beside your hips.',
    'Raise your legs to vertical, then lower them slowly without arching your back.',
  ], [
    'Press your lower back gently into the floor the whole time.',
  ]),

  // ----------------------------------------------------------- conditioning
  ex('treadmill-run', 'Treadmill Run', 'cardio', ['quads', 'calves'], 'machine', 'duration', 'run', 'machine', 8.5, [
    'Warm up with 3–5 minutes of easy walking or jogging.',
    'Run at a pace you could hold while speaking short sentences.',
    'Cool down with easy walking.',
  ], [
    'Log the honest average effort, not the sprint finish.',
  ]),
  ex('outdoor-run', 'Running (Outdoor)', 'cardio', ['quads', 'calves'], 'bodyweight', 'duration', 'run', 'none', 8.5, [
    'Start with a few minutes of brisk walking or easy jogging.',
    'Settle into a steady rhythm with relaxed shoulders.',
    'Finish with an easy cooldown.',
  ], [
    'Most runs should feel comfortable; save hard efforts for 1–2 days a week.',
  ], ['jogging', 'run']),
  ex('walking', 'Walking', 'cardio', ['calves'], 'bodyweight', 'duration', 'walk', 'none', 3.8, [
    'Walk tall at a purposeful pace.',
    'Swing your arms naturally and breathe easy.',
  ], [
    'The most underrated recovery and fat-loss tool in the app.',
  ], ['brisk walk']),
  ex('cycling', 'Cycling', 'cardio', ['quads', 'calves'], 'machine', 'duration', 'cycle', 'machine', 7.0, [
    'Set the saddle so your knee has a slight bend at the bottom.',
    'Pedal at a steady cadence you can sustain.',
  ], [
    'Around 80–95 RPM is comfortable for most riders.',
  ], ['bike', 'stationary bike', 'spin']),
  ex('rowing-machine', 'Rowing Machine', 'cardio', ['back', 'quads', 'core'], 'machine', 'duration', 'rowing', 'machine', 7.0, [
    'Catch: shins vertical, arms long, chest up.',
    'Drive with your legs, then swing your torso, then pull the handle to your ribs.',
    'Return arms–torso–legs in that order.',
  ], [
    'Legs do about 60% of the work — it is not an arm pull.',
  ], ['rower', 'erg']),
  ex('jump-rope', 'Jump Rope', 'cardio', ['calves', 'shoulders'], 'other', 'duration', 'jump-rope', 'none', 11.0, [
    'Spin the rope from your wrists with elbows close.',
    'Jump just high enough to clear the rope, landing softly.',
  ], [
    'Short frequent sessions build the skill fastest.',
  ], ['skipping']),
  ex('burpee', 'Burpee', 'full body', ['chest', 'quads', 'core', 'cardio'], 'bodyweight', 'reps-only', 'burpee', 'none', 8.0, [
    'Squat down, place your hands, and kick back to a plank.',
    'Do a push-up, jump your feet back in, and leap up with arms overhead.',
  ], [
    'Pace them — smooth continuous reps beat fast crashes.',
  ]),
  ex('elliptical', 'Elliptical Trainer', 'cardio', ['quads', 'glutes'], 'machine', 'duration', 'stair', 'machine', 5.0, [
    'Stand tall and drive through the pedals in smooth ovals.',
    'Use the handles lightly; legs do the work.',
  ], [
    'A joint-friendly option for easy aerobic days.',
  ]),
  ex('stair-climber', 'Stair Climber', 'cardio', ['quads', 'glutes', 'calves'], 'machine', 'duration', 'stair', 'machine', 9.0, [
    'Step at a rhythm you can hold without leaning on the rails.',
    'Place your whole foot on each step.',
  ], [
    'Hands off the rails as much as possible for honest effort.',
  ]),

  // ---------------------------------------------------------------- hybrids
  ex('thruster', 'Thruster (Dumbbell)', 'full body', ['quads', 'shoulders', 'glutes', 'core'], 'dumbbell', 'weight-reps', 'thruster', 'dumbbells', 8.0, [
    'Hold dumbbells at your shoulders and squat to parallel.',
    'Drive up and use the momentum to press the weights overhead.',
    'Lower the weights to your shoulders as you descend into the next rep.',
  ], [
    'One flowing movement — squat drive powers the press.',
  ]),
  ex('clean-and-press', 'Clean and Press (Barbell)', 'full body', ['shoulders', 'glutes', 'traps', 'core'], 'barbell', 'weight-reps', 'thruster', 'barbell', 8.0, [
    'Deadlift the bar explosively and catch it at your shoulders.',
    'Press it overhead to a full lockout.',
    'Lower to the shoulders, then the floor, and reset.',
  ], [
    'Learn the catch with an empty bar before adding plates.',
  ]),
];

/** Core movements plus the extended catalog used by phone, voice, and Watch. */
export const EXERCISES: Exercise[] = [...CORE_EXERCISES, ...EXTENDED_EXERCISES];

export const TEMPLATE_ROUTINE_SEEDS: {
  name: string;
  folder: string;
  items: { exerciseId: string; sets: number; repsMin?: number; repsMax?: number; durationSec?: number }[];
}[] = [
  {
    name: 'Full Body A',
    folder: 'Starter program',
    items: [
      { exerciseId: 'squat', sets: 3, repsMin: 5, repsMax: 8 },
      { exerciseId: 'bench-press', sets: 3, repsMin: 6, repsMax: 10 },
      { exerciseId: 'seated-cable-row', sets: 3, repsMin: 8, repsMax: 12 },
      { exerciseId: 'lateral-raise', sets: 2, repsMin: 12, repsMax: 15 },
      { exerciseId: 'plank', sets: 3, durationSec: 40 },
    ],
  },
  {
    name: 'Full Body B',
    folder: 'Starter program',
    items: [
      { exerciseId: 'romanian-deadlift', sets: 3, repsMin: 6, repsMax: 10 },
      { exerciseId: 'overhead-press', sets: 3, repsMin: 6, repsMax: 10 },
      { exerciseId: 'lat-pulldown', sets: 3, repsMin: 8, repsMax: 12 },
      { exerciseId: 'walking-lunge', sets: 2, repsMin: 10, repsMax: 12 },
      { exerciseId: 'crunch', sets: 3, repsMin: 12, repsMax: 20 },
    ],
  },
  {
    name: 'Push Day',
    folder: 'Push · Pull · Legs',
    items: [
      { exerciseId: 'bench-press', sets: 4, repsMin: 5, repsMax: 8 },
      { exerciseId: 'dumbbell-shoulder-press', sets: 3, repsMin: 8, repsMax: 12 },
      { exerciseId: 'incline-dumbbell-press', sets: 3, repsMin: 8, repsMax: 12 },
      { exerciseId: 'lateral-raise', sets: 3, repsMin: 12, repsMax: 15 },
      { exerciseId: 'triceps-pushdown', sets: 3, repsMin: 10, repsMax: 15 },
    ],
  },
  {
    name: 'Pull Day',
    folder: 'Push · Pull · Legs',
    items: [
      { exerciseId: 'deadlift', sets: 3, repsMin: 3, repsMax: 6 },
      { exerciseId: 'pullup', sets: 3, repsMin: 5, repsMax: 10 },
      { exerciseId: 'seated-cable-row', sets: 3, repsMin: 8, repsMax: 12 },
      { exerciseId: 'face-pull', sets: 3, repsMin: 12, repsMax: 20 },
      { exerciseId: 'barbell-curl', sets: 3, repsMin: 8, repsMax: 12 },
    ],
  },
  {
    name: 'Leg Day',
    folder: 'Push · Pull · Legs',
    items: [
      { exerciseId: 'squat', sets: 4, repsMin: 5, repsMax: 8 },
      { exerciseId: 'romanian-deadlift', sets: 3, repsMin: 8, repsMax: 12 },
      { exerciseId: 'leg-press', sets: 3, repsMin: 10, repsMax: 15 },
      { exerciseId: 'seated-leg-curl', sets: 3, repsMin: 10, repsMax: 15 },
      { exerciseId: 'standing-calf-raise', sets: 4, repsMin: 10, repsMax: 15 },
    ],
  },
  {
    name: 'Home Bodyweight',
    folder: 'No equipment',
    items: [
      { exerciseId: 'bodyweight-squat', sets: 3, repsMin: 12, repsMax: 20 },
      { exerciseId: 'pushup', sets: 3, repsMin: 8, repsMax: 15 },
      { exerciseId: 'glute-bridge', sets: 3, repsMin: 12, repsMax: 20 },
      { exerciseId: 'plank', sets: 3, durationSec: 40 },
      { exerciseId: 'mountain-climber', sets: 3, durationSec: 30 },
    ],
  },
];

const byId = new Map(EXERCISES.map((exercise) => [exercise.id, exercise]));

export function findExercise(id: string) {
  return byId.get(id);
}

export const MUSCLE_GROUPS: MuscleGroup[] = [
  'chest', 'back', 'shoulders', 'traps', 'biceps', 'triceps', 'forearms',
  'quads', 'hamstrings', 'glutes', 'calves', 'core', 'full body', 'cardio',
];

export const EQUIPMENT_TYPES: Equipment[] = [
  'barbell', 'dumbbell', 'machine', 'cable', 'bodyweight', 'kettlebell', 'band', 'other',
];

export function searchExercises(options: {
  query?: string;
  muscle?: MuscleGroup | null;
  equipment?: Equipment | null;
}) {
  const normalize = (value: string) => value
    .toLocaleLowerCase()
    .replace(/[’']/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
  const query = normalize(options.query ?? '');
  return EXERCISES.filter((exercise) => {
    if (options.muscle && exercise.primaryMuscle !== options.muscle
      && !exercise.secondaryMuscles.includes(options.muscle)) return false;
    if (options.equipment && exercise.equipment !== options.equipment) return false;
    if (!query) return true;
    const searchable = normalize([
      exercise.name,
      ...exercise.aliases,
      exercise.primaryMuscle,
      ...exercise.secondaryMuscles,
      exercise.equipment,
      exercise.kind === 'duration' ? 'timed timer' : exercise.kind === 'reps-only' ? 'bodyweight reps' : 'weight reps kg',
    ].join(' '));
    const tokens = query.split(/\s+/).filter(Boolean);
    return tokens.every((token) => searchable.includes(token));
  });
}
