/**
 * Exercise how-to guides. GENERATED from researched, trainer-reviewed content —
 * regenerate rather than hand-editing.
 *
 * Provenance: every guide was written from reputable coaching sources
 * (StrengthLog, NASM, ACE, Squat University, Renaissance Periodization,
 * BarBend, and others), then graded by an adversarial reviewer against a
 * strict rubric (equipment-specific setup, ROM standards, breathing/tempo,
 * real mistakes, safety) and revised until every guide scored 10/10. Every
 * video URL was verified against YouTube's oEmbed metadata (exact title and
 * channel) at build time.
 */

export type ExerciseGuide = {
  setup: string[];
  execution: string[];
  breathing: string;
  tempo?: string;
  mistakes: string[];
  safety?: string;
  video?: { title: string; channel: string; url: string };
};

export const EXERCISE_GUIDES: Record<string, ExerciseGuide> = {
  "bench-press": {
    "setup": [
      "Set the rack hooks so the bar sits at wrist height when you lie with your eyes directly under it.",
      "Grip the bar about 1.5x shoulder width, thumbs wrapped, bar set low in your palms over the wrist bones.",
      "Squeeze your shoulder blades together and down, set a slight arch, and plant your feet flat under your knees."
    ],
    "execution": [
      "Unrack the bar and hold it locked out directly over your shoulders.",
      "Lower for about two seconds to your mid-to-lower sternum, elbows tucked 45-70 degrees from your torso.",
      "Touch your chest lightly around nipple line with forearms vertical - no bounce.",
      "Drive the bar up and slightly back until your elbows lock out over your shoulders.",
      "After the last rep, guide the bar back until it hits the uprights, then lower into the hooks."
    ],
    "breathing": "Inhale and brace hard before each descent, then exhale forcefully as you press to lockout.",
    "tempo": "2s down · brief touch, no bounce · drive up hard",
    "mistakes": [
      "Bouncing the bar off your chest — momentum hides weakness and risks your ribs / touch softly, then press.",
      "Flaring elbows to 90 degrees — overloads the shoulder joint / keep upper arms 45-70 degrees from your torso.",
      "Lifting hips off the bench — shortens range and strains the low back / keep glutes down, drive feet into the floor.",
      "Losing shoulder-blade retraction — destabilizes the shoulder / keep blades pinched down and back for the whole set.",
      "Thumbless grip — the bar can roll off onto your face / always wrap your thumbs."
    ],
    "safety": "Bench inside a rack with safety pins, or use a spotter, whenever you train near failure.",
    "video": {
      "title": "How To Get A Huge Bench Press with Perfect Technique",
      "channel": "Jeff Nippard",
      "url": "https://www.youtube.com/watch?v=vcBig73ojpE"
    }
  },
  "incline-bench-press": {
    "setup": [
      "Set the rack hooks so the bar sits at wrist height when you lie with eyes under it.",
      "Set the bench to 30 degrees - no steeper than 45 - so the upper chest, not the delts, leads.",
      "Lie with your eyes under the bar; grip slightly wider than shoulder width, thumbs wrapped.",
      "Pull your shoulder blades back and down, keep a slight arch, and plant your feet flat."
    ],
    "execution": [
      "Unrack and hold the bar locked out directly above your shoulders.",
      "Lower for two seconds to your upper chest, just below the collarbones, elbows 45-60 degrees from your torso.",
      "Touch lightly without bouncing; from the side, your forearms should be vertical.",
      "Press up and slightly back until your elbows lock out over your shoulders.",
      "Rack by touching the uprights first, then lowering the bar into the hooks."
    ],
    "breathing": "Inhale and brace at the top, hold while lowering, and exhale as you press to lockout.",
    "tempo": "2s down · brief touch · drive up",
    "mistakes": [
      "Bench set above 45 degrees — front delts take over from the upper chest / keep it around 30 degrees.",
      "Bouncing the bar off your chest — robs tension and risks injury / touch under control, then press.",
      "Elbows flared straight out — shoulder strain / keep them 45-60 degrees from your torso.",
      "Hips sliding down the pad — turns it into a sloppy flat press / sit tall with glutes against the seat."
    ],
    "safety": "Use a rack with safeties or a spotter when training near failure.",
    "video": {
      "title": "How To: Incline Barbell Bench Press | 3 GOLDEN RULES! (MADE BETTER!)",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=SrqOu55lrYU"
    }
  },
  "dumbbell-bench-press": {
    "setup": [
      "Sit on a flat bench with a dumbbell standing upright on each thigh, palms facing in.",
      "Lie back while kicking the dumbbells up one at a time to shoulder level.",
      "Turn your palms forward, pinch your shoulder blades down and back, and plant your feet flat."
    ],
    "execution": [
      "Press both dumbbells up to lockout directly over your mid-chest, stopping an inch apart.",
      "Lower for two to three seconds until the dumbbells reach the sides of your chest.",
      "At the bottom, keep forearms vertical and elbows about 45 degrees from your torso, feeling a pec stretch.",
      "Press up and slightly inward along the same path back to lockout.",
      "To finish, lower the dumbbells to your thighs and sit up - never drop them behind you."
    ],
    "breathing": "Inhale as you lower the dumbbells, exhale as you press them back to lockout.",
    "tempo": "2-3s down · brief stretch · press up",
    "mistakes": [
      "Clanging the dumbbells at the top — bleeds tension and tips your balance / stop an inch apart at lockout.",
      "Dropping too deep too fast — strains the shoulder capsule / control the descent to a comfortable stretch.",
      "Elbows drifting straight out — impingement risk / keep them about 45 degrees from your torso.",
      "Twisting to dump the weights behind you — rotator cuff risk / bring them to your thighs and sit up."
    ],
    "safety": "Use the thigh kick-up to start and the sit-up dismount to finish, especially with heavy dumbbells.",
    "video": {
      "title": "Flat Dumbbell Bench Press",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=YQ2s_Y7g5Qk"
    }
  },
  "incline-dumbbell-press": {
    "setup": [
      "Set the bench to 30-45 degrees; start at 30 for the most upper-chest emphasis.",
      "Sit with the dumbbells upright on your thighs, then lie back, kicking them up one at a time.",
      "Hold them at shoulder level, palms forward, shoulder blades pinched down and back, feet flat."
    ],
    "execution": [
      "Press the dumbbells up to lockout directly above your upper chest.",
      "Lower for two to three seconds to the sides of your upper chest until you feel a stretch.",
      "At the bottom, keep forearms vertical and elbows about 45 degrees from your torso.",
      "Press up and slightly together until your arms lock, stopping just short of clanging."
    ],
    "breathing": "Inhale on the way down, exhale as you drive the dumbbells up.",
    "tempo": "2-3s down · 1s stretch · press up",
    "mistakes": [
      "Bench set too steep — delts steal the work / keep it between 30 and 45 degrees.",
      "Arching off the seat — turns it into a flat press / keep hips and upper back on the pad.",
      "Dumbbells wandering on uneven arcs — reps die early / keep forearms vertical and mirror both sides.",
      "Clanging at the top — loses tension / finish with the dumbbells an inch apart."
    ],
    "safety": "Lower the dumbbells to your thighs and sit up to finish; don't drop them from the top.",
    "video": {
      "title": "How To: Dumbbell Incline Press | 3 GOLDEN RULES (MADE BETTER!)",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=hChjZQhX1Ls"
    }
  },
  "machine-chest-press": {
    "setup": [
      "Adjust the seat so the handles line up with your mid-chest at nipple height.",
      "Sit with head, upper back, and hips against the pad, feet flat on the floor.",
      "Pinch your shoulder blades back into the pad and lift your chest before gripping the handles.",
      "Set the pin, then do one light rep to confirm the handles track evenly."
    ],
    "execution": [
      "Press the handles forward until your arms are fully straight, shoulders staying on the pad.",
      "Pause for one second at full extension.",
      "Return with control for two to three seconds until your hands are just outside your chest.",
      "Stop just before the weight stack touches down, keeping tension, then press again."
    ],
    "breathing": "Exhale as you press the handles out, inhale as you return them with control.",
    "tempo": "press out · 1s pause · 2-3s back",
    "mistakes": [
      "Seat too high or low — handles track at your shoulders or belly, straining joints / align them with mid-chest.",
      "Shoulders rolling forward at lockout — shifts load off the chest / keep your blades pinned to the pad.",
      "Letting the stack slam down — loses tension and control / stop just above the stack every rep.",
      "Half pressing — skips the lockout range / reach full elbow extension on every rep."
    ],
    "safety": "Machines differ - check the handle path with a light warm-up set before loading.",
    "video": {
      "title": "Machine Chest Press",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=NwzUje3z0qY"
    }
  },
  "pushup": {
    "setup": [
      "Place your hands slightly wider than shoulder width, directly under your shoulders, fingers pointing forward.",
      "Set feet hip width, squeeze glutes, and brace your abs so ears, hips, and ankles form one line.",
      "Start at the top with elbows locked, neck neutral, eyes on the floor."
    ],
    "execution": [
      "Lower your whole body as one rigid plank for about two seconds.",
      "Keep elbows tracking back about 45 degrees from your torso, not flared to 90.",
      "Stop when your chest is a fist's height from the floor, nose, chest, and hips level.",
      "Press evenly through both palms back to full elbow lockout without the hips sagging or piking."
    ],
    "breathing": "Inhale on the way down, exhale as you push back up.",
    "tempo": "2s down · no pause · push up in 1s",
    "mistakes": [
      "Sagging hips — hyperextends the low back / squeeze glutes and brace abs before every rep.",
      "Elbows flared to 90 degrees — shoulder strain / keep upper arms about 45 degrees from your ribs.",
      "Head pecking at the floor — fakes depth / lead with your chest and keep ears over shoulders.",
      "Cutting depth — half the range, half the result / lower until your chest is a fist’s height from the floor."
    ],
    "safety": "If a full rep breaks your body line, elevate your hands on a bench and progress down.",
    "video": {
      "title": "The Perfect Push Up | Do it right!",
      "channel": "Calisthenicmovement",
      "url": "https://www.youtube.com/watch?v=IODxDxX7oi4"
    }
  },
  "cable-chest-fly": {
    "setup": [
      "Set both pulleys to shoulder height and attach single-grip handles.",
      "Grab the handles, step forward past the pulley line, and split your stance, one foot ahead.",
      "Lean slightly forward from the hips; start with arms wide, a slight elbow bend, hands behind shoulder line."
    ],
    "execution": [
      "Sweep both hands forward in a wide arc until they meet in front of your mid-chest.",
      "Keep the same slight elbow bend the entire rep - motion happens only at the shoulders.",
      "Squeeze your pecs for one second where your hands meet.",
      "Return along the same arc for two to three seconds until you feel a chest stretch.",
      "Stop with hands just behind your shoulder plane; don't let the stacks yank your arms open."
    ],
    "breathing": "Inhale as your arms open into the stretch, exhale as you sweep the handles together.",
    "tempo": "sweep together · 1s squeeze · 2-3s open",
    "mistakes": [
      "Bending and straightening the elbows — turns the fly into a weak press / lock in a slight bend and keep it.",
      "Square, upright stance — the stacks pull you backward / stagger your feet and lean slightly forward.",
      "Overstretching behind the shoulders — strains pec and shoulder / end the opening just behind your shoulder line.",
      "Loading too heavy — shrugging and momentum take over / drop weight until every arc is smooth."
    ],
    "safety": "Stop the stretch short of any front-shoulder pinch.",
    "video": {
      "title": "Cable Flye",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=4mfLHnFL0Uw"
    }
  },
  "dumbbell-fly": {
    "setup": [
      "Sit with dumbbells on your thighs; kick them up one at a time as you lie back.",
      "Lie flat with feet planted and shoulder blades pinched, dumbbells locked out over your mid-chest.",
      "Turn palms to face each other and unlock your elbows into a slight, fixed bend."
    ],
    "execution": [
      "Open your arms in a wide arc, lowering for two to three seconds.",
      "Keep the elbow bend frozen; move only at the shoulders.",
      "Stop when your upper arms are about parallel to the floor and you feel a clear pec stretch.",
      "Reverse the same arc, as if hugging a barrel, until the dumbbells are an inch apart.",
      "Squeeze your chest for one second at the top without turning it into a press."
    ],
    "breathing": "Inhale as your arms open into the stretch, exhale as you bring the dumbbells together.",
    "tempo": "2-3s open · 1s stretch · squeeze together",
    "mistakes": [
      "Elbows bending under load — triceps take over and it becomes a press / go lighter, freeze the elbow angle.",
      "Dropping elbows far below the bench — overloads the pec at its weakest / stop at a strong, pain-free stretch.",
      "Jerking out of the bottom — invites a pec strain / reverse the arc smoothly, no bounce.",
      "Going too heavy — range shrinks and shoulders shrug / pick a weight that allows a full, slow arc."
    ],
    "safety": "Progress in small weight jumps and never bounce out of the stretched position.",
    "video": {
      "title": "How to Perform Dumbbell Flys | Chest Exercise Tutorial",
      "channel": "Buff Dudes Workouts",
      "url": "https://www.youtube.com/watch?v=LzFvciCdoW0"
    }
  },
  "pec-deck": {
    "setup": [
      "Set the seat so your hands and elbows sit level with your mid-chest, upper arms horizontal.",
      "Sit with back and head flat against the pad, feet planted on the floor.",
      "Grip the handles with a slight elbow bend; on arm-pad machines, set forearms flat on the pads."
    ],
    "execution": [
      "Bring both handles together in one smooth arc until they meet in front of your chest.",
      "Keep the slight elbow bend fixed; move only at the shoulder joint.",
      "Squeeze your pecs for one second at the point of contact.",
      "Open back up for two to three seconds until your elbows reach just behind your shoulder plane.",
      "Stop before the stack touches down to keep tension for the next rep."
    ],
    "breathing": "Exhale as you squeeze the handles together, inhale as you open back to the stretch.",
    "tempo": "close · 1s squeeze · 2-3s open",
    "mistakes": [
      "Seat too low or high — upper arms angle up or down and strain the shoulders / keep them horizontal.",
      "Back peeling off the pad — momentum replaces pec work / stay glued to the backrest.",
      "Overstretching backward — front-shoulder strain / end the opening just behind your shoulder plane.",
      "Shrugging into the squeeze — traps steal the work / keep shoulders down and chest tall."
    ],
    "safety": "If the machine has a range-limiter pin, set the stretch to stop within a pain-free range.",
    "video": {
      "title": "Pec Deck Flye",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=O-OBCfyh9Fw"
    }
  },
  "chest-dip": {
    "setup": [
      "Use parallel bars about shoulder width apart; grip with thumbs wrapped, bars low in your palms.",
      "Jump to support: elbows locked, shoulders pressed down away from your ears, ankles crossed behind you.",
      "Tip your torso forward about 30 degrees and keep your gaze slightly down to hold the lean."
    ],
    "execution": [
      "Bend your elbows and lower for two to three seconds, keeping the forward lean.",
      "Let your elbows travel back and slightly out, about 45 degrees from your ribs.",
      "Descend until your shoulders sit just below your elbows and you feel a pec stretch.",
      "Press back up to full lockout, holding the lean so your chest keeps working.",
      "Squeeze your chest for one second at the top before the next rep."
    ],
    "breathing": "Inhale at the top, hold your breath through the descent, and exhale as you lock out.",
    "tempo": "2-3s down · brief stretch · drive up",
    "mistakes": [
      "Staying bolt upright — shifts the load to your triceps / keep the 30-degree forward lean throughout.",
      "Half reps — the deep stretch is the growth zone / lower until shoulders drop just below the elbows.",
      "Shoulders shrugging or rolling forward — impingement risk / press them down before and during every rep.",
      "Kipping or swinging the legs — momentum steals tension / cross your ankles and keep your body quiet."
    ],
    "safety": "Earn clean, full-range bodyweight sets before loading a dip belt, and stop if the front shoulder pinches.",
    "video": {
      "title": "Are You Doing Dips Properly? (AVOID MISTAKES!)",
      "channel": "ATHLEAN-X",
      "url": "https://www.youtube.com/watch?v=vi1-BOcj3cQ"
    }
  },
  "triceps-pushdown": {
    "setup": [
      "Attach a straight or EZ bar to the highest pulley setting.",
      "Grip overhand at shoulder width or slightly narrower, wrists straight.",
      "Stand a half step back, feet shoulder width, hinging slightly forward from the hips.",
      "Pin your elbows to your sides with the bar starting around chest height."
    ],
    "execution": [
      "Push the bar straight down until your elbows lock out completely, bar near your thighs.",
      "Keep elbows glued to your ribs; only the forearms move.",
      "Squeeze the triceps hard for one second at lockout.",
      "Let the bar rise under control for two seconds back to chest height, elbows never drifting."
    ],
    "breathing": "Exhale as you press the bar down, inhale as you let it rise under control.",
    "tempo": "press down · 1s squeeze · 2s up",
    "mistakes": [
      "Elbows drifting forward and up — lats and shoulders take over / keep them pinned to your ribs all set.",
      "Leaning your bodyweight onto the bar — momentum replaces triceps / hold a slight, fixed hip hinge.",
      "Stopping short of lockout — full extension is the triceps' main job / finish every rep with straight elbows.",
      "Wrists bending backward — strains the joint and leaks force / keep knuckles down and wrists neutral."
    ],
    "video": {
      "title": "Cable Triceps Pushdown",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=6Fzep104f0s"
    }
  },
  "overhead-cable-extension": {
    "setup": [
      "Attach a rope to a low pulley, grab both ends, and face away from the stack.",
      "Press the rope overhead, stagger your stance, and lean slightly forward with your core braced.",
      "Set upper arms vertical beside your ears, elbows pointing forward, knuckles to the ceiling."
    ],
    "execution": [
      "Bend only your elbows, lowering your hands behind your head for two to three seconds.",
      "Stop when your forearms touch your biceps - a full triceps stretch.",
      "Keep your upper arms frozen beside your ears the whole time.",
      "Extend back to complete lockout overhead, splitting the rope ends slightly at the top.",
      "Squeeze the triceps for one second before the next rep."
    ],
    "breathing": "Inhale as your hands drop behind your head, exhale as you extend to lockout.",
    "tempo": "2-3s down · full stretch · extend and squeeze",
    "mistakes": [
      "Elbows flaring wide — shoulders and chest join in / keep them narrow, pointing forward beside your ears.",
      "Upper arms see-sawing each rep — turns it into a press / freeze them vertical, moving only the forearms.",
      "Overarching the lower back — the cable pulls you into extension / brace your abs and stagger your stance.",
      "Cutting the stretch short — the lengthened position drives growth / let forearms close fully onto the biceps."
    ],
    "video": {
      "title": "Cable Overhead Triceps Extension",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=1u18yJELsh0"
    }
  },
  "skullcrusher": {
    "setup": [
      "Lie on a flat bench, feet planted, holding an EZ bar by the inner grips, palms overhand.",
      "Press the bar to lockout, then tip your arms a few degrees back toward your head.",
      "Pinch your shoulder blades and keep your wrists straight."
    ],
    "execution": [
      "Bend only your elbows, lowering the bar for two to three seconds.",
      "Keep elbows shoulder width apart and pointing at the ceiling - no flare.",
      "Stop the bar about an inch above your forehead, or just past your head for more stretch.",
      "Extend your elbows back to the tipped-back lockout without letting the upper arms swing.",
      "Squeeze for one second, keeping the bar behind your eye line between reps."
    ],
    "breathing": "Inhale as the bar lowers toward your forehead, exhale as you extend to lockout.",
    "tempo": "2-3s down · no pause · extend up",
    "mistakes": [
      "Flaring the elbows outward — turns it into a close-grip press / keep them shoulder width, aimed up.",
      "Upper arms swinging back and forth — cheats the triceps / lock them still and move only at the elbow.",
      "Rushing the descent — the bar is over your face / lower on a strict two-to-three-second count.",
      "Grinding to failure alone — obvious hazard / stop a rep short or keep a spotter close."
    ],
    "safety": "Use a spotter or switch to dumbbells near failure - the bar travels over your face.",
    "video": {
      "title": "Skull Crushers | Triceps | How-To Exercise Tutorial",
      "channel": "Buff Dudes Workouts",
      "url": "https://www.youtube.com/watch?v=QXzhjRnYRT0"
    }
  },
  "close-grip-bench": {
    "setup": [
      "Set the rack hooks so the bar sits at wrist height when you lie with eyes under it.",
      "Lie with your eyes under the bar; grip at shoulder width so your hands stack directly above your shoulders.",
      "Wrap your thumbs, set wrists straight, and pinch shoulder blades back and down with a slight arch.",
      "Plant your feet flat under your knees."
    ],
    "execution": [
      "Unrack and hold the bar locked out over your shoulders.",
      "Lower for two seconds with elbows tucked, upper arms brushing your sides.",
      "Touch your lower chest, where the ribs end, without bouncing.",
      "Press up and slightly back to full lockout, elbows finishing straight.",
      "Rack by touching the uprights first, then lowering into the hooks."
    ],
    "breathing": "Inhale and brace at the top, hold while lowering, and exhale as you press to lockout.",
    "tempo": "2s down · brief touch · drive up",
    "mistakes": [
      "Gripping so narrow your hands nearly touch — wrecks wrists and elbows for zero gain / stay at shoulder width.",
      "Elbows flaring wide — turns it back into a regular bench / keep upper arms brushing your torso.",
      "Bouncing off the chest — momentum steals the triceps work / touch softly, then press.",
      "Wrists rolling backward — joints ache and force leaks / keep the bar stacked over your wrist bones."
    ],
    "safety": "Use safety pins or a spotter near failure.",
    "video": {
      "title": "CLOSE GRIP PRESS | Triceps | How-To Exercise Tutorial",
      "channel": "Buff Dudes Workouts",
      "url": "https://www.youtube.com/watch?v=cXbSJHtjrQQ"
    }
  },
  "bench-dip": {
    "setup": [
      "Sit on the edge of a bench and grip it beside your hips, fingers forward, hands shoulder width.",
      "Slide your hips just off the bench with legs extended and heels on the floor.",
      "Bend your knees to make it easier, or elevate your feet on a second bench to make it harder."
    ],
    "execution": [
      "Lower for two seconds by bending your elbows straight back, torso staying upright.",
      "Keep your hips brushing the bench and shoulders pressed down, away from your ears.",
      "Stop when your elbows reach about 90 degrees.",
      "Press through your palms back to straight arms without shrugging."
    ],
    "breathing": "Inhale on the way down, exhale as you press back up.",
    "tempo": "2s down · no pause · press up",
    "mistakes": [
      "Hips drifting forward of the bench — loads the front shoulder capsule / keep glutes skimming the bench edge.",
      "Sinking far below 90 degrees — shoulder strain outweighs any triceps gain / stop at a right angle.",
      "Shrugging shoulders toward the ears — impingement risk / keep them pressed down the entire set.",
      "Elbows flaring sideways — shifts stress to the shoulders / point them straight back over the bench."
    ],
    "safety": "Skip this movement if you have front-shoulder pain - pushdowns or parallel-bar dips are friendlier.",
    "video": {
      "title": "How to Do Triceps Bench Dips",
      "channel": "LIVESTRONG",
      "url": "https://www.youtube.com/watch?v=0326dy_-CzM"
    }
  },
  "overhead-dumbbell-extension": {
    "setup": [
      "Stand with feet shoulder width, or sit on a low-back bench, holding one dumbbell.",
      "Cup the inside of the top plate with both palms, thumbs around the handle, and press it overhead.",
      "Set both upper arms vertical beside your ears, elbows pointing forward, ribs pulled down."
    ],
    "execution": [
      "Bend only your elbows, lowering the dumbbell behind your head for two to three seconds.",
      "Stop when your forearms touch your biceps and you feel a full triceps stretch.",
      "Keep both upper arms frozen and vertical.",
      "Extend back to lockout directly overhead and squeeze for one second."
    ],
    "breathing": "Inhale as the dumbbell lowers behind your head, exhale as you press to lockout.",
    "tempo": "2-3s down · full stretch · extend up",
    "mistakes": [
      "Elbows flaring wide — shoulders take over and the stretch disappears / squeeze them toward your ears.",
      "Arching the lower back — overhead load pulls you into extension / brace abs, tuck ribs, squeeze glutes.",
      "Cutting the descent short — misses the stretch that drives growth / lower until forearms meet biceps.",
      "Dumbbell too heavy to control — it drifts toward your neck / size down until every rep is smooth."
    ],
    "safety": "Lift the dumbbell into position with both hands and keep your head neutral - never jerk it overhead.",
    "video": {
      "title": "How To: Dumbbell Seated Overhead Tricep Extension",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=YbX7Wd8jQ-Q"
    }
  },
  "diamond-pushup": {
    "setup": [
      "Place your hands together under the center of your chest, thumbs and index fingers touching in a diamond.",
      "Step back to a plank: feet together or hip width, glutes and abs braced, body one straight line."
    ],
    "execution": [
      "Lower for about two seconds, elbows tracking back along your ribs.",
      "Keep ears, hips, and ankles in one line the whole way down.",
      "Stop when your chest touches or nearly touches your hands.",
      "Press back up to full elbow lockout without your hips sagging or hiking."
    ],
    "breathing": "Inhale on the descent, exhale as you press back to lockout.",
    "tempo": "2s down · brief touch · press up",
    "mistakes": [
      "Hands placed ahead of your chest — overloads shoulders and wrists / keep the diamond stacked under your chest.",
      "Elbows winging outward — loses the triceps bias / drag them back along your ribs.",
      "Hips piking or sagging — unloads the triceps and chest / brace glutes and abs on every rep.",
      "Cutting depth — half the range, half the result / touch your chest to your hands."
    ],
    "safety": "If your wrists complain, separate your hands an inch or two - keeping elbows tucked preserves the triceps bias.",
    "video": {
      "title": "How to Perform Diamond Push Ups | Bodyweight Exercise Tutorial",
      "channel": "Buff Dudes Workouts",
      "url": "https://www.youtube.com/watch?v=ZR5U3sb-KeE"
    }
  },
  "deadlift": {
    "setup": [
      "Stand with feet hip-width apart, toes forward or slightly out, bar over mid-foot, about an inch from your shins.",
      "Hinge at the hips and grip the bar double overhand, hands just outside your legs, arms vertical.",
      "Bend your knees until your shins touch the bar without pushing it away from mid-foot.",
      "Set shoulders slightly in front of the bar, chest up, spine neutral from neck to tailbone.",
      "Pull the slack out of the bar and squeeze your lats, as if bending the bar around your shins."
    ],
    "execution": [
      "Take a deep breath into your belly and brace your core hard, like preparing for a punch.",
      "Drive the floor away with your legs; hips and shoulders rise together, back angle constant.",
      "Drag the bar up your shins and thighs in a straight vertical line over mid-foot.",
      "Past the knees, drive your hips forward to lockout: knees and hips straight, glutes squeezed.",
      "Stand tall at the top; do not lean back or shrug.",
      "Push your hips back first, bend your knees once the bar passes them, and lower under control.",
      "Let the bar settle, then re-set your spine and brace before the next rep — no bouncing."
    ],
    "breathing": "Inhale and brace at the bottom before every pull; hold your breath through the rep, then exhale at lockout or after lowering. Re-brace each rep.",
    "tempo": "Smooth, fast 1–2 s pull, 1 s at lockout, 2 s controlled lowering, full reset on the floor.",
    "mistakes": [
      "Rounding the lower back — shifts load from muscle to spinal discs / brace, lift your chest, and lighten until neutral holds.",
      "Bar drifting away from the shins — every inch multiplies lower-back leverage / squeeze your lats and drag the bar up your legs.",
      "Hips shooting up first — turns the lift into a stiff-leg pull / push the floor away with chest and hips rising together.",
      "Jerking the slack — yanking rounds your spine at the worst moment / tension the bar until it clicks, then drive.",
      "Hyperextending at lockout — leaning back compresses the lumbar spine / finish tall, glutes squeezed, hips under shoulders."
    ],
    "safety": "Never keep pulling once your lower back rounds — end the set. Add weight gradually, and use hook grip or straps before grip failure degrades your setup.",
    "video": {
      "title": "How To Deadlift: 5 Step Deadlift | 2022",
      "channel": "Alan Thrall (Untamed Strength)",
      "url": "https://www.youtube.com/watch?v=MBbyAqvTNkU"
    }
  },
  "pullup": {
    "setup": [
      "Grip the bar with palms facing away, hands slightly wider than shoulder-width, thumbs wrapped.",
      "Hang with elbows fully straight and feet off the floor, legs together or ankles crossed.",
      "Pull your shoulder blades down and back so your shoulders sit away from your ears.",
      "Brace your core and squeeze your glutes so your body hangs rigid, not swinging."
    ],
    "execution": [
      "Start every rep from a dead hang with straight elbows.",
      "Drive your elbows down toward your ribs, pulling your chest up to the bar.",
      "Pull until your chin clears the bar with a neutral neck — collarbone near bar height is the standard.",
      "Pause one second at the top, chest tall, shoulder blades squeezed.",
      "Lower in two to three seconds back to a full dead hang; that is one rep."
    ],
    "breathing": "Exhale as you pull up; inhale during the controlled lowering, or take a breath in the dead hang between reps.",
    "tempo": "1 s pull, 1 s hold at the top, 2–3 s lower, brief dead-hang reset between reps.",
    "mistakes": [
      "Kipping or swinging — momentum robs the lats and whips the shoulders / brace your trunk; do strict or band-assisted reps.",
      "Stopping short at the bottom — skipping full elbow extension cuts the lat stretch / lower to a complete dead hang every rep.",
      "Craning the chin over the bar — neck extension fakes the range / pull higher with your back until your collarbone approaches the bar.",
      "Shrugging to your ears — traps take over and shoulders pinch / lead by driving the shoulder blades down, then pull."
    ],
    "safety": "Lower to a controlled dead hang before letting go instead of dropping mid-rep; if strict reps aren't there yet, use band or machine assistance rather than kipping.",
    "video": {
      "title": "The Perfect Pull Up - Do it right!",
      "channel": "Calisthenicmovement",
      "url": "https://www.youtube.com/watch?v=eGo4IYlbE5g"
    }
  },
  "chinup": {
    "setup": [
      "Grip the bar with palms facing you, hands shoulder-width or slightly narrower.",
      "Hang with elbows fully straight, feet off the floor, ankles crossed if it stops swinging.",
      "Pull your shoulder blades down and back; keep ribs down and core braced."
    ],
    "execution": [
      "Pull your elbows down and slightly back, leading with your chest.",
      "Keep elbows close to your torso so biceps and lats share the load.",
      "Pull until your chin clears the bar with a neutral neck — no craning.",
      "Squeeze for one second at the top with your chest near the bar.",
      "Lower under control for two to three seconds to a full dead hang."
    ],
    "breathing": "Exhale on the way up, inhale as you lower; reset your breath in the dead hang if needed.",
    "tempo": "1 s up, 1 s squeeze at the top, 2–3 s down, full hang between reps.",
    "mistakes": [
      "Stopping above full extension — partial reps shortchange lats and biceps / straighten your elbows completely at the bottom of every rep.",
      "Swinging the legs — momentum does the lifting / brace abs and glutes and slow the lowering to kill the swing.",
      "Leading with the chin — neck extension fakes the rep standard / keep the neck neutral and pull your collarbone toward the bar.",
      "Flaring the elbows wide — throws work to a strained shoulder position / keep forearms vertical and elbows tracking in front of you."
    ],
    "safety": "If wrists or elbows ache with a narrow underhand grip, widen your hands slightly or switch to a neutral grip.",
    "video": {
      "title": "PERFECT CHIN-UPS | The Only Chin-up Tutorial You'll Ever Need (Full Guide)",
      "channel": "Simonster Strength",
      "url": "https://www.youtube.com/watch?v=e1YSApl-QcM"
    }
  },
  "lat-pulldown": {
    "setup": [
      "Adjust the thigh pad so your knees lock snugly under it, feet flat on the floor.",
      "Stand and grip the bar overhand, hands about one and a half times shoulder-width, thumbs wrapped.",
      "Sit down with arms fully extended, letting the stack stretch your lats upward.",
      "Lean back 10–20 degrees from vertical, lift your chest, and hold that angle all set."
    ],
    "execution": [
      "Begin each pull by drawing your shoulder blades down, then drive your elbows toward your hips.",
      "Pull the bar to your upper chest at collarbone level, elbows pointing down and slightly back.",
      "Squeeze your lats for one second at the bottom; wrists stay straight, hands just hooks.",
      "Let the bar rise under control until your elbows are straight and shoulders stretch up.",
      "Keep the torso angle frozen — no rocking backward to move the weight."
    ],
    "breathing": "Exhale as you pull the bar down; inhale as you return it under control.",
    "tempo": "1 s pull down, 1 s squeeze, 2–3 s controlled return to a full stretch.",
    "mistakes": [
      "Pulling behind the neck — forces a risky shoulder position with no extra lat work / always pull to the upper chest.",
      "Rocking the torso — momentum turns the pulldown into a sloppy row / set the 10–20 degree lean and freeze it.",
      "Death-gripping and curling the wrists — forearms and biceps steal the work / wrists straight, pull through the elbows.",
      "Cutting the top short — no stretch, less growth / let your arms straighten fully every rep."
    ],
    "safety": "Pull to your upper chest, never behind your neck — behind-the-neck pulldowns put the shoulder in a vulnerable position for no added benefit.",
    "video": {
      "title": "Normal Grip Pulldown",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=EUIri47Epcg"
    }
  },
  "barbell-row": {
    "setup": [
      "Stand with the bar over your mid-foot, feet hip- to shoulder-width apart.",
      "Push your hips back until your torso is 15–30 degrees above horizontal, knees slightly bent.",
      "Grip slightly wider than shoulder-width, overhand, arms hanging straight under your shoulders.",
      "Set a neutral spine from head to hips and brace your core before the first pull."
    ],
    "execution": [
      "Row the bar to your lower chest or upper abdomen, elbows driving up and back at about 45 degrees.",
      "Squeeze your shoulder blades together for one second; the bar touches or nearly touches your torso.",
      "Lower under control until your elbows are fully straight, keeping the torso angle unchanged.",
      "Hold the same hip hinge every rep — if your chest rises more than 10–15 degrees, that's cheating."
    ],
    "breathing": "Breathe in and brace before the pull, hold through the row, and exhale as the bar lowers. Re-breathe every rep.",
    "tempo": "Explosive 1 s pull, 1 s squeeze at the torso, 2 s lowering, brief pause at straight arms.",
    "mistakes": [
      "Standing too upright — turns rows into shrugs and cuts lat work / re-set the hinge to 15–30 degrees above horizontal.",
      "Heaving with the hips — momentum robs the upper back and jerks the lumbar spine / drop 10–20% of the load and freeze your torso.",
      "Rounding the lower back — flexion under load risks disc injury / brace harder, lighten, and keep the head-to-hip line neutral.",
      "Pulling high with flared elbows — strains shoulders and misses the lats / touch the lower chest or upper abs, elbows about 45 degrees.",
      "Cutting the bottom — partial reps skip the lat stretch / straighten your elbows fully before every pull."
    ],
    "safety": "Your lower back works isometrically the whole set — stop the moment the fixed torso angle or neutral spine starts breaking down.",
    "video": {
      "title": "\"How To\" Barbell Row",
      "channel": "Alan Thrall (Untamed Strength)",
      "url": "https://www.youtube.com/watch?v=G8l_8chR5BE"
    }
  },
  "dumbbell-row": {
    "setup": [
      "Place your left knee and left hand on a bench, right foot planted on the floor beside it.",
      "Set your torso nearly parallel to the floor — no more than 15 degrees above horizontal — back flat.",
      "Hold the dumbbell in your right hand, palm facing the bench, arm hanging straight under your shoulder.",
      "Square your shoulders and hips to the floor and brace your core."
    ],
    "execution": [
      "Pull the dumbbell up and slightly back toward your hip, elbow brushing your ribs.",
      "Drive the shoulder blade toward your spine as your elbow passes your torso.",
      "Lift until the elbow passes your torso line; the dumbbell finishes near your lower ribs.",
      "Pause one second with shoulders level — do not rotate your chest open.",
      "Lower in two seconds to a full stretch, letting the shoulder blade glide forward at the bottom.",
      "Complete all reps, then switch sides and match the count."
    ],
    "breathing": "Exhale as you row the dumbbell up; inhale during the two-second lowering.",
    "tempo": "1 s pull, 1 s pause at the ribs, 2 s lower into a full stretch.",
    "mistakes": [
      "Twisting the torso to lift higher — rotation fakes range the lats never see / keep your chest square to the floor and go lighter.",
      "Yanking with the arm — biceps take over the pull / start each rep by drawing the shoulder blade back, then the elbow.",
      "Rounding the back — a slack spine under load invites strain / set a flat line from neck to tailbone before rowing.",
      "Stopping high at the bottom — skips half the lat's range / lower until the arm hangs fully and the shoulder blade slides forward."
    ],
    "safety": "Keep your spine neutral when picking the dumbbell up and setting it down — most tweaks happen there, not mid-set.",
    "video": {
      "title": "How To: Dumbbell Bent-Over Row (Single-Arm)",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=pYcpY20QaE8"
    }
  },
  "seated-cable-row": {
    "setup": [
      "Attach a V-handle, sit down, and plant your feet on the platform with knees slightly bent — never locked.",
      "Grab the handle, then push back with your legs until your arms are straight and torso upright.",
      "Sit tall: chest up, spine neutral, torso vertical at 90 degrees to your legs."
    ],
    "execution": [
      "Pull the handle to your upper abdomen, driving your elbows straight back along your sides.",
      "Squeeze your shoulder blades together for one second, chest staying lifted.",
      "Return under control, letting the shoulder blades glide forward as your arms straighten and lats stretch.",
      "Keep torso sway within about 10 degrees of vertical in both directions; legs stay still."
    ],
    "breathing": "Exhale as you pull the handle in; inhale as you return to the stretched position.",
    "tempo": "1 s pull, 1 s squeeze, 2–3 s return; keep tension — no slack between reps.",
    "mistakes": [
      "Leaning far back to finish — turns the row into a lower-back swing / stay within 10 degrees of vertical and lower the stack.",
      "Collapsing forward under load at the stretch — a flexed spine plus pull equals strain / reach through the shoulder blades, spine neutral.",
      "Shrugging as you pull — upper traps override the mid-back / pull the shoulder blades down and together, elbows staying low.",
      "Locking the knees — transfers the pull onto your lumbar spine and hamstrings / keep a soft bend the entire set."
    ],
    "safety": "Bend your knees and keep a neutral spine when grabbing and releasing the handle, and return the stack under control rather than dropping it.",
    "video": {
      "title": "How To: Seated Cable Low-Row || PERFECT FORM",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=7o2oolbmzeI"
    }
  },
  "tbar-row": {
    "setup": [
      "Load one end of a landmine bar or T-bar machine and straddle it, feet shoulder-width.",
      "Hook the close-grip handle under the bar just behind the plates.",
      "Push your hips back until your torso is about 45 degrees above horizontal, knees slightly bent.",
      "Brace your core and lift the bar to hanging arms' length while holding that hinge."
    ],
    "execution": [
      "Row the handle toward your lower chest, elbows driving up and back.",
      "Bring the plates to your chest — or within an inch — and squeeze your shoulder blades for one second.",
      "Lower under control to straight arms without dropping your chest or bouncing the plates.",
      "Hold the 45-degree torso angle every rep; if you stand up as you pull, it's too heavy."
    ],
    "breathing": "Take a breath and brace before each pull; exhale through the lowering, then re-brace.",
    "tempo": "1 s pull, 1 s squeeze at the chest, 2 s lower to full arm extension.",
    "mistakes": [
      "Standing up as you row — hip extension fakes the pull / lock the hinge at about 45 degrees and lighten the bar.",
      "Rounding the lower back — spinal flexion in a loaded hinge risks injury / brace hard and keep your hips pushed back.",
      "Bouncing the plates off your chest — the rebound erases muscle tension / pause at the top and take two seconds down.",
      "Flaring the elbows wide — shifts stress to rear delts and shoulder joints / drive elbows up at roughly 45 degrees from your torso."
    ],
    "safety": "Your spine holds a loaded hinge all set — end the set when neutral breaks. Use 10–25 lb plates so the plates don't cut your range short.",
    "video": {
      "title": "How To T-Bar Row The Right Way! (BACK BUILDER!)",
      "channel": "Mind Pump TV",
      "url": "https://www.youtube.com/watch?v=5foJiIVhs8Q"
    }
  },
  "machine-row": {
    "setup": [
      "Set the seat so the handles align with your mid-chest and the pad supports your sternum.",
      "Plant your feet flat, press your chest into the pad, and take a neutral or overhand grip.",
      "Start with arms fully extended and shoulder blades stretched forward."
    ],
    "execution": [
      "Pull the handles toward your torso, driving elbows back and squeezing the shoulder blades together.",
      "Keep your chest glued to the pad for the entire rep — no pushing away from it.",
      "Pause one second at full contraction with elbows just past your torso.",
      "Return slowly until your arms are straight and shoulder blades glide forward; don't let the stack touch down."
    ],
    "breathing": "Exhale while pulling, inhale on the slow return; keep breathing steady since the pad supports your spine.",
    "tempo": "1 s pull, 1 s squeeze, 2–3 s return; constant tension with no rest at the stack.",
    "mistakes": [
      "Pushing off the chest pad — arching away adds momentum / keep your sternum in contact; the pad is your form judge.",
      "Pinning the shoulder blades at the stretch — half range limits growth / let them protract fully as the arms straighten.",
      "Curling with the wrists — wastes force before it reaches your back / wrists straight; treat your hands as hooks.",
      "Overloading the stack — short, jerky reps replace controlled pulls / choose a weight that allows a one-second pause every rep."
    ],
    "video": {
      "title": "Improve Your Machine Rows For A Bigger Back",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=5XfV6hvcxCE"
    }
  },
  "straight-arm-pulldown": {
    "setup": [
      "Set a straight bar or rope at the top of a cable stack.",
      "Grip overhand about shoulder-width and step back one to two feet until the cable is taut.",
      "Hinge your torso forward 20–30 degrees, hips back, knees soft, arms overhead with a 10–15 degree elbow bend.",
      "Brace your core and pre-tighten your lats before the first rep."
    ],
    "execution": [
      "Sweep the bar down in a wide arc to your thighs with arms staying straight.",
      "Move only at the shoulders; the elbow bend stays locked throughout.",
      "Squeeze your lats for one second at your thighs without hunching your shoulders forward.",
      "Let the bar rise along the same arc until your arms are overhead and lats stretch.",
      "Keep torso angle and elbow bend identical on every rep."
    ],
    "breathing": "Exhale as you sweep the bar to your thighs; inhale as your arms return overhead.",
    "tempo": "1–2 s pull down, 1 s squeeze, 2–3 s return into the overhead stretch.",
    "mistakes": [
      "Bending and straightening the elbows — turns the movement into a triceps pushdown / lock the slight bend; rotate only at the shoulders.",
      "Standing bolt upright — shortens the lat stretch and range / hinge 20–30 degrees and hold it.",
      "Dropping bodyweight onto the bar — leaning hides weak lats / lighten the stack and keep the torso frozen.",
      "Shoulders rolling forward at the bottom — shifts load off the lats / keep the chest proud and shoulder blades down."
    ],
    "video": {
      "title": "Build your LATS with Straight Arm Pulldowns | Exercise Tutorial",
      "channel": "Buff Dudes Workouts",
      "url": "https://www.youtube.com/watch?v=soX7zhZ7yfQ"
    }
  },
  "back-extension": {
    "setup": [
      "Use a 45-degree hyperextension bench; set the pad so your hip crease sits just above its top edge.",
      "Hook your ankles under the rollers, feet flat against the plate.",
      "Cross your arms over your chest, or hug a plate there for added load.",
      "Set your body in one straight line from head to heels, glutes and core lightly braced."
    ],
    "execution": [
      "Hinge at the hips and lower your torso with a flat back until your hamstrings stretch.",
      "Stop before your back starts to round — typically 60–90 degrees of hinge.",
      "Raise your torso by squeezing glutes and hamstrings until your body is one straight line again.",
      "Stop level with your legs — never hyperextend past straight.",
      "Control both directions; no swinging and no bouncing at the bottom."
    ],
    "breathing": "Inhale on the way down; exhale as you raise your torso back to the straight-line position.",
    "tempo": "2 s down, brief pause, 1–2 s up, 1 s glute squeeze at the top.",
    "mistakes": [
      "Hyperextending at the top — arching past straight pinches the lumbar spine / stop at one straight line and squeeze the glutes.",
      "Rounding to chase depth — spinal flexion under fatigue strains discs / lower only as far as a flat back allows.",
      "Swinging up on momentum — robs the glutes and hamstrings / take two seconds down and pause before rising.",
      "Pad set too high — blocks the hip hinge and forces the spine to bend / top of the pad at the hip crease exactly."
    ],
    "safety": "You should feel glutes, hamstrings, and working back muscles — never sharp spinal pain. With a lower-back condition, shorten the range and skip added load.",
    "video": {
      "title": "Back Extensions for Stronger Legs (THE RIGHT WAY)",
      "channel": "elitefts",
      "url": "https://www.youtube.com/watch?v=hK9Mf86KzTI"
    }
  },
  "rack-pull": {
    "setup": [
      "Set the rack pins so the bar rests just below your kneecaps, or at your chosen sticking-point height.",
      "Stand with the bar over your mid-foot, feet hip-width, shins nearly touching the bar.",
      "Hinge and grip shoulder-width, double overhand; add straps or hook grip as loads climb.",
      "Lift your chest, set a neutral spine, and place shoulders slightly ahead of the bar."
    ],
    "execution": [
      "Inhale, brace hard, and pull the slack out of the bar until your whole body is tense.",
      "Drive through your heels and push your hips forward, dragging the bar up your thighs.",
      "Lock out standing tall: hips and knees straight, glutes squeezed, shoulders back — no lean-back.",
      "Lower the bar under control to the pins and let it settle completely.",
      "Re-set your spine and brace before every rep; never bounce off the pins."
    ],
    "breathing": "Breathe and brace at the pins before every pull; hold through lockout and exhale as the bar returns.",
    "tempo": "1 s pull, 1 s hold at lockout, 2 s lower, dead-stop reset on the pins.",
    "mistakes": [
      "Bouncing off the pins — the rebound fakes strength and jars the spine / dead-stop, re-brace, then pull every rep.",
      "Hyperextending at lockout — leaning back compresses the lumbar spine / finish tall with hips under shoulders and glutes tight.",
      "Overloading because the range is short — bracing collapses and the back rounds / cap loads at what a neutral spine holds.",
      "Setting pins too high — a two-inch pull trains ego, not muscle / start just below the kneecap."
    ],
    "safety": "Rack pulls invite big load jumps — brace like a max deadlift on every rep, keep both pins at equal heights, and lower the bar rather than dropping it.",
    "video": {
      "title": "Gym Shorts (How To): The Rack Pull",
      "channel": "Barbell Logic",
      "url": "https://www.youtube.com/watch?v=0nJs6Cnfv3M"
    }
  },
  "barbell-shrug": {
    "setup": [
      "Set a barbell in a rack at mid-thigh height so you don't deadlift it every set.",
      "Grip double overhand just outside your thighs; add straps if grip fails before your traps.",
      "Stand tall with feet hip-width, knees soft, arms hanging straight, core braced."
    ],
    "execution": [
      "Shrug your shoulders straight up toward your ears as high as they go.",
      "Hold the top for one to two seconds, squeezing the upper traps hard.",
      "Keep arms straight the whole rep — no elbow bend to hoist the bar higher.",
      "Lower for two seconds until your shoulders are fully stretched down.",
      "Move strictly up and down; never roll the shoulders in circles."
    ],
    "breathing": "Exhale as you shrug up; inhale on the slow lowering or between reps at the bottom.",
    "tempo": "1 s up, 1–2 s hold at the top, 2 s down into a full stretch.",
    "mistakes": [
      "Rolling the shoulders — circular motion adds joint wear and zero extra trap work / shrug straight up, straight down.",
      "Bending the elbows — biceps lift the bar and the range shrinks / keep arms as passive hooks.",
      "Bouncing from the knees — leg drive steals the traps' work / stand still so only the shoulders move.",
      "Skipping the top hold — traps grow from the squeeze / own a full second at the top of every rep.",
      "Jutting the head forward — strains the neck under load / chin tucked, neck neutral, eyes forward."
    ],
    "video": {
      "title": "Barbell Shrug Technique For Growth | Targeting The Muscle",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=zfAHfyTB_Ao"
    }
  },
  "dumbbell-shrug": {
    "setup": [
      "Hold a dumbbell in each hand at your sides, palms facing your thighs.",
      "Stand tall with feet hip-width, knees soft, chest up, shoulders in a neutral start.",
      "Brace your core lightly so your torso doesn't sway."
    ],
    "execution": [
      "Shrug both shoulders straight up toward your ears as high as possible.",
      "Pause one to two seconds at the top and squeeze the traps.",
      "Lower slowly until your shoulders hang fully stretched; that is one rep.",
      "Keep arms straight and the dumbbells close to your body throughout."
    ],
    "breathing": "Exhale during the shrug; inhale as the dumbbells lower.",
    "tempo": "1 s up, 1–2 s top squeeze, 2 s down.",
    "mistakes": [
      "Rolling the shoulders forward or back — adds shoulder stress, not muscle / move vertically only.",
      "Rocking or heel-bouncing — leg drive creates the lift / stand planted; if you must bounce, the dumbbells are too heavy.",
      "Tiny range with huge dumbbells — partial shrugs stunt growth / pick a load that allows ear-to-stretch range with a pause.",
      "Pushing the head forward — loads the neck / keep the chin tucked and eyes forward."
    ],
    "video": {
      "title": "How To: Dumbbell Shrug",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=xDt6qbKgLkY"
    }
  },
  "wrist-curl": {
    "setup": [
      "Sit on a bench holding a light dumbbell in each hand, palms facing up.",
      "Rest your forearms on your thighs with wrists hanging just past your knees.",
      "Pin your forearms to your thighs — only the wrists move from here."
    ],
    "execution": [
      "Lower the dumbbells by extending your wrists down, letting them roll toward your fingertips for extra range.",
      "Curl your wrists up as high as they flex, squeezing your forearms.",
      "Pause one second at the top with forearms still pressed into your thighs.",
      "Lower over two seconds; no bouncing out of the bottom stretch."
    ],
    "breathing": "Exhale as you curl your wrists up; inhale on the slow lowering.",
    "tempo": "1 s up, 1 s squeeze, 2 s down through a full stretch.",
    "mistakes": [
      "Lifting the forearms off the thighs — elbows and momentum join in / keep forearms pinned for the entire set.",
      "Going too heavy — wrist flexors are small, so range shrinks to millimeters / use light loads through a full flex-to-stretch arc.",
      "Rushing the bottom turnaround — fast reversals strain wrist tendons / lower for two seconds and pause before curling.",
      "Training only flexion — imbalanced forearms invite wrist and elbow aches / pair with reverse wrist curls at similar volume."
    ],
    "safety": "Wrist tendons adapt more slowly than muscles — add load gradually and stop for sharp pain; a burning pump is normal, joint pain is not.",
    "video": {
      "title": "THIS is the perfect form to build your forearms and wrist curls by Dr. Jim Stoppani",
      "channel": "Jim Stoppani, PhD",
      "url": "https://www.youtube.com/watch?v=xeUGvGZzBG8"
    }
  },
  "reverse-wrist-curl": {
    "setup": [
      "Sit on a bench with a light dumbbell in each hand, palms facing down.",
      "Rest your forearms on your thighs with wrists hanging just past your knees.",
      "Load roughly half your wrist-curl weight — the extensors are much weaker muscles."
    ],
    "execution": [
      "Let the dumbbells lower until your wrists are fully flexed toward the floor.",
      "Raise the backs of your hands as far as your wrists extend.",
      "Pause one second at the top without the forearms leaving your thighs.",
      "Lower under control for two seconds back to the stretched position.",
      "Keep knuckles tracking straight up and down — no rolling side to side."
    ],
    "breathing": "Exhale as you raise your knuckles; inhale on the way down.",
    "tempo": "1 s up, 1 s hold, 2 s down.",
    "mistakes": [
      "Using wrist-curl weight — extensors handle far less and form collapses / start at about half your wrist-curl load.",
      "Bouncing at the bottom — jerks the extensor tendons near the elbow / use slow, dead-stop turnarounds.",
      "Elbows drifting upward — arms and shoulders fake the lift / keep forearms pressed into your thighs.",
      "Crushing the handle — an over-gripped hand limits wrist extension / hold firmly but let the wrist move freely."
    ],
    "safety": "These tendons share the outer-elbow attachment irritated in tennis elbow — sharp pain there means stop, lighten, and progress more slowly.",
    "video": {
      "title": "How to Do Dumbbell Reverse Wrist Curls",
      "channel": "LIVESTRONG",
      "url": "https://www.youtube.com/watch?v=krZ6pWGZ8xo"
    }
  },
  "farmers-carry": {
    "setup": [
      "Set two heavy dumbbells or farmer's handles on the floor, one beside each foot.",
      "Deadlift them up: hinge with a flat back, grip hard, and stand by driving through your legs.",
      "Stand tall — shoulders down and back, ribs stacked over hips, arms straight at your sides.",
      "Crush the handles and brace your core before taking the first step."
    ],
    "execution": [
      "Walk with short, quick, controlled steps, heel to toe, feet under your hips.",
      "Keep shoulders level and torso bolt upright; the weights must not swing or tilt you sideways.",
      "Look ahead with a neutral neck, not down at the floor.",
      "Carry for 50–100 feet (15–30 meters) or 30–45 seconds per set.",
      "Finish by hinging down with a flat back to set the weights on the floor."
    ],
    "breathing": "Never hold your breath — take steady, rhythmic breaths behind a braced core for the entire carry.",
    "tempo": "Brisk, even steps for 50–100 feet or 30–45 seconds; rest 1–2 minutes between carries.",
    "mistakes": [
      "Leaning or shrugging under the load — posture collapse turns a core drill into a back strain / go lighter and stay tall.",
      "Overstriding — long steps let the weights swing and yank your grip / short, quick steps with feet under your hips.",
      "Rounding your back lifting or lowering the weights — the riskiest moment of the exercise / deadlift them up and down.",
      "Holding your breath — you'll gas out mid-carry and lose bracing / breathe rhythmically behind the brace."
    ],
    "safety": "Pick a clear, flat walking path before you lift. If your grip is failing mid-carry, stop, hinge, and set the weights down rather than stumbling on.",
    "video": {
      "title": "How To Do The FARMER'S WALK Correctly",
      "channel": "SET FOR SET",
      "url": "https://www.youtube.com/watch?v=VBobkldqqvk"
    }
  },
  "dead-hang": {
    "setup": [
      "Use a sturdy overhead bar with a box or bench so you can reach it without jumping.",
      "Grip overhand, hands shoulder-width apart, thumbs wrapped around the bar.",
      "Step off the box and hang with your arms completely straight."
    ],
    "execution": [
      "Relax into the hang — feet off the floor, legs quiet, body still.",
      "Keep elbows locked straight; let the shoulder blades rise toward your ears for a passive stretch.",
      "Hold 10 seconds as a beginner; build gradually toward 45–60 seconds.",
      "Add sets before adding time — for example, three 30-second hangs.",
      "Finish by stepping back onto the box before releasing your grip — never drop."
    ],
    "breathing": "Breathe slowly and deeply for the entire hang — steady breathing relaxes you and extends the hold.",
    "tempo": "Static hold: 10 seconds for beginners, progressing to 45–60 seconds; 2–3 sets with full rest.",
    "mistakes": [
      "Jumping to catch the bar — a swinging start snaps load onto shoulders and grip / start from a box and settle first.",
      "Bending the elbows — turns the hang into a flexed-arm hold and kills the stretch / keep arms dead straight.",
      "Dropping off the bar — landing loaded risks ankles and knees / step back onto the box to exit.",
      "Hanging rigid with total-body tension — defeats the decompression purpose / breathe, relax your legs, and let the spine lengthen."
    ],
    "safety": "Step down from a box rather than dropping; stop if you feel shoulder pain rather than a stretch.",
    "video": {
      "title": "Hanging / Brachiation Exercise for Shoulder Health and Stronger Grip",
      "channel": "GMB Fitness (Praxis)",
      "url": "https://www.youtube.com/watch?v=f4OLLQXmRQs"
    }
  },
  "overhead-press": {
    "setup": [
      "Set the bar at upper-chest height in a rack and load your plates.",
      "Grip just outside shoulder width, palms forward, wrists stacked straight over your forearms.",
      "Unrack the bar onto your front delts, step back, and set your feet hip-width apart.",
      "Squeeze your glutes and brace your abs so your ribs stay down and your lower back stays neutral."
    ],
    "execution": [
      "Press the bar straight up while pulling your chin back out of its path.",
      "Once the bar passes your forehead, push your head forward through your arms.",
      "Lock out with the bar stacked over your shoulders and midfoot, biceps beside your ears.",
      "Lower the bar under control back to your front delts, then begin the next rep."
    ],
    "breathing": "Take a big breath and brace at the shoulders; exhale forcefully as the bar passes your head; re-breathe at the bottom of every rep.",
    "tempo": "Drive up in about 1 second, pause briefly at lockout, lower in 2 seconds.",
    "mistakes": [
      "Leaning back and arching the lower spine — shifts the load onto your lumbar back / squeeze your glutes and keep your ribs pulled down.",
      "Pressing the bar out around your face — the bar drifts forward and the press stalls / pull your chin back so the bar path stays vertical.",
      "Stopping short of lockout — you never reach the stable stacked finish and shortchange the delts / finish with biceps beside your ears.",
      "Elbows flared straight out to the sides — leaks pressing power and irritates the shoulders / keep forearms vertical, elbows slightly in front of the bar."
    ],
    "safety": "Press with a hard brace on every rep; a soft core under an overhead load is how lower backs get hurt. Keep the bar path clear of your chin and nose.",
    "video": {
      "title": "Build Bigger Shoulders With Perfect Training Technique (The Overhead Press)",
      "channel": "Jeff Nippard",
      "url": "https://www.youtube.com/watch?v=_RlRDWO2jfg"
    }
  },
  "dumbbell-shoulder-press": {
    "setup": [
      "Set the bench backrest just short of vertical, around 80-85 degrees.",
      "Sit down with a dumbbell standing upright on each thigh.",
      "Kick one knee up at a time to bring each dumbbell to shoulder height, palms facing forward.",
      "Plant your feet flat, press your upper back into the pad, dumbbells just outside your shoulders."
    ],
    "execution": [
      "Press both dumbbells up and slightly inward until your arms are straight, stopping short of clanging them together.",
      "Keep your forearms vertical under the dumbbells the whole way up.",
      "Lower with control until the dumbbells return to shoulder level and you feel a stretch in your delts."
    ],
    "breathing": "Inhale as you lower the dumbbells, exhale as you press to lockout.",
    "tempo": "1-2 seconds up, brief pause at the top, 2-3 seconds down.",
    "mistakes": [
      "Arching off the backrest — turns it into an incline press and strains the lower back / keep ribs down, glutes and upper back on the pad.",
      "Cutting the descent short — half reps skip the stretched range where delts grow best / lower until the dumbbells reach shoulder level.",
      "Letting the dumbbells drift forward or wide — strains the shoulder joint and wastes pressing power / keep them stacked over your elbows.",
      "Slamming the dumbbells together at the top — jars the shoulders and loses tension / stop just short of touching."
    ],
    "safety": "Kick the dumbbells up from your thighs one at a time to reach the start; never haul them from the floor straight overhead. Reverse the move to set them down.",
    "video": {
      "title": "How To: Dumbbell Shoulder Press",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=qEwKCR5JCog"
    }
  },
  "machine-shoulder-press": {
    "setup": [
      "Adjust the seat so the handles sit level with the tops of your shoulders.",
      "Select your weight and sit with hips, back, and head against the pad.",
      "Grip the handles with wrists straight and elbows pointing down, feet flat on the floor."
    ],
    "execution": [
      "Press the handles straight up until your arms are fully extended, without slamming into lockout.",
      "Pause for one second at the top.",
      "Lower with control to shoulder level, stopping before the weight stack touches down."
    ],
    "breathing": "Exhale as you press up, inhale as you lower back to shoulder level.",
    "tempo": "1-2 seconds up, 1-second pause, 2-3 seconds down.",
    "mistakes": [
      "Seat set too low — the handles start below your shoulders and jam the joint at the bottom / raise the seat until the handles match shoulder height.",
      "Arching your back off the pad to grind out reps — transfers stress to the lower back / lower the weight and keep your ribs down.",
      "Letting the stack rest between reps — tension drops to zero and the set becomes singles / stop just above the stack.",
      "Shrugging your shoulders toward your ears as you press — traps take over from delts / keep your shoulders pulled down."
    ],
    "safety": "If pressing with palms forward bothers your shoulders, use the neutral parallel-grip handles; the movement stays identical with less rotational stress.",
    "video": {
      "title": "How To Use The Shoulder Press Machine",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=TnhIyp4kmO8"
    }
  },
  "lateral-raise": {
    "setup": [
      "Stand tall with light dumbbells at your sides, palms facing your thighs.",
      "Set your feet hip-width apart and pull your shoulders down away from your ears.",
      "Bend your elbows slightly and keep that bend fixed for the whole set."
    ],
    "execution": [
      "Raise both arms straight out to the sides, leading with your elbows, not your hands.",
      "Stop exactly at shoulder height, arms parallel to the floor — never higher.",
      "Pause for one second at the top.",
      "Lower back to your sides over 2-3 seconds, resisting the whole way down."
    ],
    "breathing": "Exhale as you raise the dumbbells, inhale as you lower them.",
    "tempo": "1-2 seconds up, 1-second pause at shoulder height, 2-3 seconds down.",
    "mistakes": [
      "Swinging the torso to lift heavier dumbbells — momentum robs the side delts of tension / drop the weight and raise strictly.",
      "Raising above shoulder height — past parallel the upper traps take over / stop when your hands reach shoulder level.",
      "Tilting the dumbbells far thumb-down at the top — can pinch the shoulder / keep your hands level with your elbows, thumbs roughly neutral.",
      "Shrugging as you lift — traps steal the work / keep your shoulders pressed down throughout the raise."
    ],
    "safety": "If the top of the raise pinches, bring your arms about 20-30 degrees forward of straight sideways and stop just below shoulder height.",
    "video": {
      "title": "How To: Dumbbell Side Lateral Raise",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=3VcKaXpzqRo"
    }
  },
  "cable-lateral-raise": {
    "setup": [
      "Set the pulley to its lowest position and attach a single D-handle.",
      "Stand side-on to the machine and grab the handle with the hand farther from the stack.",
      "Start with that hand in front of the machine-side hip so the cable crosses your body; bend the elbow slightly.",
      "Hold the frame with your free hand and stand tall, shoulders down."
    ],
    "execution": [
      "Raise your arm out to the side in a wide arc, leading with your elbow.",
      "Stop at shoulder height, wrist level with your elbow and arm parallel to the floor.",
      "Lower over 2-3 seconds until your hand returns in front of the machine-side hip.",
      "Finish the set, turn around, and repeat with the other arm."
    ],
    "breathing": "Exhale as you raise the handle, inhale as you lower it.",
    "tempo": "1-2 seconds up, brief pause at shoulder height, 2-3 seconds down; the cable keeps tension even at the bottom.",
    "mistakes": [
      "Swinging your torso toward and away from the stack — momentum replaces delt tension / lock your torso and move only at the shoulder.",
      "Raising above shoulder height — the traps take over past parallel / stop when your wrist reaches shoulder level.",
      "Bending and straightening the elbow — turns the raise into a partial press / keep the slight elbow bend fixed.",
      "Shrugging as the arm rises — the traps steal the tension / keep the working shoulder pulled down."
    ],
    "video": {
      "title": "How To: Single Arm Standing Cable Lateral Raise",
      "channel": "Live Lean TV Daily Exercises",
      "url": "https://www.youtube.com/watch?v=nMCQuV6HE3A"
    }
  },
  "front-raise": {
    "setup": [
      "Stand tall, feet hip-width, a dumbbell in each hand resting on the front of your thighs, palms facing you.",
      "Soften your elbows slightly and brace your core so your torso cannot rock."
    ],
    "execution": [
      "Raise one dumbbell straight out in front of you, palm down, until your hand reaches shoulder height.",
      "Pause briefly at the top without shrugging.",
      "Lower it under control back to your thigh.",
      "Alternate arms until you finish equal reps per side."
    ],
    "breathing": "Exhale as each dumbbell rises, inhale as it lowers.",
    "tempo": "1-2 seconds up, brief pause at shoulder height, about 2 seconds down per arm.",
    "mistakes": [
      "Leaning back to launch the weight — your hips, not your front delts, do the lifting / lighten the load and keep ribs stacked over hips.",
      "Raising past shoulder height — adds trap work, not delt work / stop when your hand is level with your shoulder.",
      "Bending and straightening the elbow — turns the raise into a partial press / hold a fixed, slight elbow bend."
    ],
    "video": {
      "title": "How To: Dumbbell Front Raise",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=-t7fuZ0KhDA"
    }
  },
  "rear-delt-fly": {
    "setup": [
      "Hold light dumbbells and hinge at your hips until your torso is nearly parallel to the floor.",
      "Keep your back flat, knees soft, and let the dumbbells hang under your chest, palms facing each other.",
      "Bend your elbows slightly and fix that angle for the whole set."
    ],
    "execution": [
      "Raise both dumbbells out to the sides in a wide arc, leading with your elbows.",
      "Think wide, not back — do not row the weights toward your hips.",
      "Stop when your arms are level with your back, parallel to the floor, and squeeze the rear delts.",
      "Lower the dumbbells back beneath your chest with control."
    ],
    "breathing": "Exhale as you raise the dumbbells, inhale as you lower them, keeping your brace so your back stays flat.",
    "tempo": "1-2 seconds up, 1-second squeeze, 2-3 seconds down.",
    "mistakes": [
      "Standing too upright — the movement becomes a lateral raise and the side delts take over / hinge until your torso nears parallel.",
      "Bouncing the torso to swing the weights up — momentum replaces rear-delt work / use lighter dumbbells and hold the hinge dead still.",
      "Pinching the shoulder blades together — traps and rhomboids take over / keep the blades apart and move only at the shoulders.",
      "Bending the elbows more as you lift — turns the fly into a row / keep the slight fixed bend and sweep wide."
    ],
    "safety": "Keep your spine neutral throughout the hinge. If your lower back tires before your delts, sit on the end of a bench or lie chest-down on an incline bench instead.",
    "video": {
      "title": "How to do a Bent-Over Dumbbell Rear Fly",
      "channel": "National Academy of Sports Medicine (NASM)",
      "url": "https://www.youtube.com/watch?v=kLW7nbw4lcY"
    }
  },
  "reverse-pec-deck": {
    "setup": [
      "Set the handle arms so they start together at the front, and set the seat so the handles match shoulder height.",
      "Sit facing the machine with your chest against the pad.",
      "Grab the handles with palms facing each other and elbows slightly bent."
    ],
    "execution": [
      "Sweep your arms out and back in a wide arc, moving only at the shoulders.",
      "Stop when your hands are straight out to your sides, level with your shoulders.",
      "Squeeze your rear delts for one second.",
      "Return slowly until your hands are almost together, stopping before the stack touches down."
    ],
    "breathing": "Exhale as you sweep the handles back, inhale as you return them forward.",
    "tempo": "1-2 seconds back, 1-second squeeze, 2-3 seconds forward.",
    "mistakes": [
      "Squeezing the shoulder blades hard together — mid-traps and rhomboids take over / keep the blades quiet and sweep from the shoulder joint.",
      "Bending the elbows as you pull — turns the fly into a row / hold the slight elbow bend from start to finish.",
      "Going so heavy you jerk off the pad — momentum replaces rear-delt tension / choose a weight you can pause at the back.",
      "Letting the stack slam down between reps — tension resets to zero / stop just short of the stack every rep."
    ],
    "video": {
      "title": "How to Reverse Pec Dec For HUGE round rear delts - Exercise Set up and Form with Hypertrophy Coach",
      "channel": "Hypertrophy Coach",
      "url": "https://www.youtube.com/watch?v=LzsSQzHFuLU"
    }
  },
  "face-pull": {
    "setup": [
      "Set a rope attachment at face height or slightly above on a cable column.",
      "Grip the rope ends with palms facing each other, thumbs pointing back toward you.",
      "Step back until your arms are straight and the stack lifts; stagger your stance and brace."
    ],
    "execution": [
      "Pull the rope straight toward the bridge of your nose, elbows flaring high and wide.",
      "As it nears your face, pull the ends apart so your hands finish beside your ears.",
      "Finish with upper arms level with your shoulders and knuckles pointing at the ceiling.",
      "Hold the squeeze for one second, then let your arms extend slowly back to the start."
    ],
    "breathing": "Exhale as you pull to your face, inhale on the slow return.",
    "tempo": "1-2 seconds pull, 1-second hold at the face, 2-3 seconds return.",
    "mistakes": [
      "Pulling the rope to your chest or neck — it becomes a row and the external rotation disappears / aim at your nose, hands finishing by your ears.",
      "Elbows dropping toward your ribs — lats take over from the rear delts / keep your upper arms level with your shoulders.",
      "Loading too heavy — you lean, jerk, and cut the rotation short / use a weight you can pause at your face.",
      "Standing square and upright with a heavy stack — you get pulled forward onto your toes / stagger your stance and set your brace first."
    ],
    "safety": "Keep loads modest. The rotator-cuff muscles finishing this movement are small; every rep should reach the full hands-by-ears position without grinding.",
    "video": {
      "title": "Stop Doing Face Pulls Like This! (SAVE A FRIEND)",
      "channel": "ATHLEAN-X™",
      "url": "https://www.youtube.com/watch?v=eIq5CB9JfKE"
    }
  },
  "upright-row": {
    "setup": [
      "Grip the bar with an overhand grip at shoulder width or slightly wider.",
      "Stand tall with the bar resting on your thighs, arms straight, knees slightly bent.",
      "Brace your core and set your shoulders down and back."
    ],
    "execution": [
      "Lead with your elbows and drag the bar up close to your body.",
      "Stop when the bar reaches mid-chest height, elbows level with — never above — your shoulders.",
      "Pause for one second, elbows staying higher than your wrists.",
      "Lower the bar down the same path to your thighs over 2-3 seconds."
    ],
    "breathing": "Exhale as you pull the bar up, inhale as you lower it.",
    "tempo": "1-2 seconds up, 1-second pause at mid-chest, 2-3 seconds down.",
    "mistakes": [
      "Pulling to your chin — forces deep internal rotation under load and invites impingement / stop at mid-chest, elbows no higher than shoulders.",
      "Gripping very narrow — bends the wrists sharply and increases shoulder rotation / grip at shoulder width or a touch wider.",
      "Heaving with your hips and leaning back — momentum robs delts and traps and loads your spine / lighten the bar and pull strictly.",
      "Letting the bar swing away from your body — stresses the lower back and shoulders / keep it grazing your shirt the whole rep."
    ],
    "safety": "If your shoulders have a history of impingement, pull only to lower-chest height with a wide grip — or swap in lateral raises, which train the same delts with less rotation stress.",
    "video": {
      "title": "How To: Barbell Upright Row",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=amCU-ziHITM"
    }
  },
  "barbell-curl": {
    "setup": [
      "Stand with feet hip-width apart, knees soft, weight even across both feet.",
      "Grip the bar underhand at shoulder width.",
      "Stand tall with the bar at arm's length against your thighs, elbows pinned to your sides.",
      "Brace your core and set your shoulders down and back."
    ],
    "execution": [
      "Curl the bar in an arc toward your shoulders, moving only at the elbows.",
      "Squeeze your biceps at the top with your elbows still pinned at your sides.",
      "Lower the bar under control until your arms are completely straight."
    ],
    "breathing": "Exhale as you curl the bar up, inhale as you lower it.",
    "tempo": "1-2 seconds up, 1-second squeeze, 2-3 seconds down to straight arms.",
    "mistakes": [
      "Swinging the hips and leaning back — momentum robs the biceps and loads the spine / lighten the bar and keep your torso still.",
      "Elbows drifting forward at the top — front delts take over and biceps tension drops / keep your elbows pinned beside your ribs.",
      "Stopping the descent halfway — you skip the stretched range that drives growth / straighten your arms fully every rep.",
      "Wrists curling back toward you — strains the wrists and adds forearm cheat / keep your knuckles in line with your forearms."
    ],
    "safety": "If a straight bar aches your wrists, switch to an EZ-curl bar; the angled grips remove most wrist strain without changing the movement.",
    "video": {
      "title": "How To Do A Barbell Bicep Curl",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=N5x5M1x1Gd0"
    }
  },
  "dumbbell-curl": {
    "setup": [
      "Stand with feet hip-width apart, knees soft, weight even across both feet.",
      "Stand tall with a dumbbell in each hand at arm's length, palms facing forward.",
      "Pin your elbows to your sides and set your shoulders down and back."
    ],
    "execution": [
      "Curl both dumbbells toward your shoulders, keeping your upper arms completely still.",
      "Squeeze at the top, turning your pinkies slightly upward for a full contraction.",
      "Lower under control until your arms hang completely straight."
    ],
    "breathing": "Exhale as you curl up, inhale on the way down.",
    "tempo": "1-2 seconds up, 1-second squeeze, 2-3 seconds down.",
    "mistakes": [
      "Rocking the torso — momentum does the biceps' job / brace and keep your ribs stacked over your hips.",
      "Elbows sliding forward — the front delts lift the weight for you / keep your elbows pinned beside your ribs.",
      "Half-lowering between reps — you skip the stretch that builds the muscle / fully straighten your arms at the bottom.",
      "Shrugging the dumbbells up — traps creep into every rep / keep your shoulders down and move only your forearms."
    ],
    "video": {
      "title": "The ONLY Way You Should Be Doing Dumbbell Bicep Curls!",
      "channel": "Mind Pump TV",
      "url": "https://www.youtube.com/watch?v=in7PaeYlhrM"
    }
  },
  "hammer-curl": {
    "setup": [
      "Stand with feet hip-width apart, knees soft, weight even across both feet.",
      "Stand tall with dumbbells at your sides, palms facing your thighs in a neutral grip.",
      "Pin your elbows to your sides and brace lightly."
    ],
    "execution": [
      "Curl the dumbbells up thumbs-first, palms staying face-to-face the entire way.",
      "Stop when the top of each dumbbell nears your front shoulder and squeeze for a second.",
      "Lower under control until your arms are fully straight."
    ],
    "breathing": "Exhale as you curl, inhale as you lower.",
    "tempo": "1-2 seconds up, 1-second squeeze, 2-3 seconds down.",
    "mistakes": [
      "Swinging the weights up with your hips — momentum replaces the brachialis work / slow down and keep your torso still.",
      "Elbows drifting forward — shifts tension off the arm flexors / keep your upper arms vertical at your sides.",
      "Rotating the palms up at the top — turns it into a regular curl and loses the brachioradialis emphasis / hold the neutral grip throughout."
    ],
    "video": {
      "title": "How To: Dumbbell Hammer Curl",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=zC3nLlEvin4"
    }
  },
  "incline-dumbbell-curl": {
    "setup": [
      "Set an incline bench to about 45-60 degrees.",
      "Sit back with your head and upper back on the pad, a dumbbell hanging straight down in each hand.",
      "Turn your palms forward and let your arms hang slightly behind your torso — that stretch is the point."
    ],
    "execution": [
      "Curl both dumbbells toward your shoulders while your upper arms keep pointing straight at the floor.",
      "Squeeze at the top without letting your elbows swing forward or your shoulders roll off the pad.",
      "Lower slowly until your arms are completely straight and you feel a deep stretch in the biceps."
    ],
    "breathing": "Exhale as you curl, inhale as you lower into the stretch.",
    "tempo": "1-2 seconds up, 1-second squeeze, about 3 seconds down into the stretch.",
    "mistakes": [
      "Elbows drifting forward as you curl — kills the long-head stretch the incline exists for / keep your upper arms hanging straight down.",
      "Using your standing-curl weight — the stretched position is weaker and form collapses / start roughly 20-30 percent lighter.",
      "Lifting your head and shoulders off the pad — turns it into a partial seated curl / stay pressed back for the whole set.",
      "Cutting the bottom range — skips the stretch stimulus entirely / straighten your arms fully every rep."
    ],
    "safety": "This curl loads the biceps in a deep stretch. Add weight gradually and stop if you feel anything sharp in the front of your shoulder or elbow.",
    "video": {
      "title": "Dumbbell Incline Curl exercise tutorial",
      "channel": "Buff Dudes Workouts",
      "url": "https://www.youtube.com/shorts/S2cYwsDhpI4"
    }
  },
  "preacher-curl": {
    "setup": [
      "Adjust the seat so your armpits sit snugly over the top edge of the pad.",
      "Rest the backs of both upper arms flat on the pad and grip the handles.",
      "Sit tall with your chest against the pad, arms nearly straight."
    ],
    "execution": [
      "Curl the handles up until your forearms are just short of vertical.",
      "Squeeze your biceps for one second at the top.",
      "Lower slowly until your elbows are almost — but not completely — straight."
    ],
    "breathing": "Exhale as you curl, inhale as you lower.",
    "tempo": "1-2 seconds up, 1-second squeeze, about 3 seconds down.",
    "mistakes": [
      "Lifting the elbows off the pad at the top — the shoulders take over and the biceps unload / press your upper arms into the pad and stop short of vertical forearms.",
      "Snapping into a fully locked elbow at the bottom — jolts the stretched biceps tendon / stop a few degrees short, under control.",
      "Seat too low — you end up shrugging and pulling with your shoulders / raise the seat until your armpits anchor the pad's top edge.",
      "Dropping the weight on the way down — wastes the stretched-position tension preacher curls exist for / lower on a slow three count."
    ],
    "safety": "Never bounce out of the bottom. The pad fixes your biceps at long muscle length, where a fast reversal can strain the tendon — control every centimeter.",
    "video": {
      "title": "Machine Preacher Curl",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=Ja6ZlIDONac"
    }
  },
  "cable-curl": {
    "setup": [
      "Stand with feet hip-width or one foot slightly back, bracing against the stack’s forward pull.",
      "Attach a straight or EZ bar to the lowest pulley position.",
      "Grip it underhand at shoulder width and step back until the cable is taut with your arms straight.",
      "Stand tall with your elbows pinned to your sides."
    ],
    "execution": [
      "Curl the bar toward your shoulders, keeping your upper arms motionless.",
      "Squeeze your biceps for one second at the top.",
      "Lower under control to fully straight arms, stopping before the stack touches down."
    ],
    "breathing": "Exhale as you curl, inhale as you lower.",
    "tempo": "1-2 seconds up, 1-second squeeze, 2-3 seconds down.",
    "mistakes": [
      "Standing too close to the pulley — tension disappears at the bottom of every rep / step back until the cable is taut at full arm's length.",
      "Leaning back as you curl — turns the curl into a cable row / keep your torso vertical and your ribs down.",
      "Elbows drifting forward — front delts take over the top half / keep your elbows pinned beside your ribs.",
      "Letting the stack slam down between reps — the pause kills the cable's constant tension / stop just short of touchdown."
    ],
    "video": {
      "title": "How To Do A Cable Curl",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=GNlopToAZyg"
    }
  },
  "concentration-curl": {
    "setup": [
      "Sit on a bench with your feet flat and legs wide, one dumbbell in hand.",
      "Lean forward and brace the back of that upper arm against your inner thigh, just above the knee.",
      "Let the arm hang straight with your palm facing away from your leg."
    ],
    "execution": [
      "Curl the dumbbell toward your shoulder, keeping the upper arm glued to your thigh.",
      "At the top, rotate your pinky slightly upward and squeeze hard for a second.",
      "Lower slowly until your arm is completely straight.",
      "Finish all reps, then switch arms."
    ],
    "breathing": "Exhale as you curl, inhale as you lower.",
    "tempo": "1-2 seconds up, 1-second squeeze, 2-3 seconds down.",
    "mistakes": [
      "Letting the elbow slide off the thigh — the brace is the whole point of the exercise / re-set your arm against your thigh every rep.",
      "Rocking the torso to lift the weight — momentum, not biceps / sit still and move only your forearm.",
      "Stopping the descent early — skips the stretch / straighten your arm completely each rep."
    ],
    "video": {
      "title": "How To Do Concentration Curls",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=llD6MImgqe8"
    }
  },
  "squat": {
    "setup": [
      "Set the rack hooks at mid-chest height; load the bar and set safety pins just below your bottom position.",
      "Grip slightly wider than shoulder width, pull your shoulder blades together, and place the bar on your upper traps.",
      "Stand up to unrack, take two or three steps back, and set feet shoulder-width, toes out 15-30 degrees."
    ],
    "execution": [
      "Inhale deep into your belly and brace your core hard, 360 degrees around your spine.",
      "Bend knees and hips together, sitting down and slightly back, keeping the bar over your mid-foot.",
      "Push your knees out so they track over your toes; keep your heels pressed into the floor.",
      "Descend under control until your hip crease drops below the top of your kneecap.",
      "Drive up through your whole foot, chest up, until your hips and knees fully lock out.",
      "Exhale at the top, re-brace, and repeat; walk the bar back into the hooks when finished."
    ],
    "breathing": "Big inhale and 360-degree brace at the top; hold your breath through the rep; exhale forcefully once past the sticking point.",
    "tempo": "Lower in 2-3 seconds, no bounce at the bottom, drive up in about 1 second.",
    "mistakes": [
      "Knees caving inward — stresses knee ligaments and leaks force / screw your feet into the floor and push knees out over your toes.",
      "Heels lifting — shifts weight onto your toes and tips you forward / sit back more and build ankle mobility; use small plates under heels meanwhile.",
      "Cutting depth — shallow squats build less muscle and strength / reduce the load until hip-below-knee depth is consistent every rep.",
      "Hips shooting up first — turns the squat into a back lift / brace harder and lead the ascent with your chest.",
      "Looking up at the ceiling — hyperextends your neck and disturbs balance / fix your eyes on the floor a few meters ahead."
    ],
    "safety": "Always squat inside a rack with safety pins or spotter arms set just below your deepest position, and keep your spine neutral under load.",
    "video": {
      "title": "Perfect Your Squat: Step-By-Step Form Guide",
      "channel": "Squat University",
      "url": "https://www.youtube.com/watch?v=8Kls95w2jFA"
    }
  },
  "front-squat": {
    "setup": [
      "Set the rack hooks at mid-chest height so you can unrack the bar with a small knee dip.",
      "Rest the bar on your front delts, touching your throat; fingertips under the bar just outside your shoulders.",
      "Drive your elbows up until your upper arms are near parallel to the floor.",
      "Stand to unrack, step back twice, and set feet shoulder-width with toes out 15-30 degrees."
    ],
    "execution": [
      "Inhale and brace your core while standing tall; keep elbows high before you move.",
      "Squat straight down with a vertical torso, letting your knees travel forward over your toes.",
      "Descend until your hip crease sits below the top of your knee, heels flat.",
      "Drive up through mid-foot, leading with your chest and elbows, to full lockout.",
      "Exhale near the top, re-brace, and repeat before racking the bar."
    ],
    "breathing": "Inhale and brace at the top before each rep; hold during the descent and drive; exhale as you finish the lockout.",
    "tempo": "2-3 seconds down, smooth turnaround, drive up in about 1 second — no dive-bombing.",
    "mistakes": [
      "Elbows dropping — the bar rolls forward and strains your wrists / re-set with upper arms parallel and cue 'elbows to the wall ahead'.",
      "Full-grip squeezing — tight wrists take the load and ache / open your hands; the bar rests on shoulders, fingertips only guide it.",
      "Torso tipping forward — you lose the rack position and dump the bar / lighten the load and slow the descent.",
      "Heels rising — limited ankle mobility shifts you onto your toes / wear lifting shoes or elevate heels slightly while mobility improves."
    ],
    "safety": "If you lose the bar forward, push it away and step back rather than chasing it; set rack safeties just below your bottom position.",
    "video": {
      "title": "HOW TO FRONT SQUAT: Build Bigger Quads & A Stronger Squat",
      "channel": "Jeff Nippard",
      "url": "https://www.youtube.com/watch?v=v-mQm_droHg"
    }
  },
  "goblet-squat": {
    "setup": [
      "Hold one dumbbell vertically by its top head, pressed against your sternum, elbows pointing down.",
      "Stand with feet shoulder-width or slightly wider, toes turned out 15-30 degrees."
    ],
    "execution": [
      "Inhale, brace your core, and keep the dumbbell glued to your chest.",
      "Squat down between your knees, keeping your torso tall and heels planted.",
      "Reach depth where your elbows lightly brush inside your knees, hip crease below your kneecap.",
      "Gently pry your knees out over your toes at the bottom.",
      "Push through your whole foot to stand tall, exhaling on the way up."
    ],
    "breathing": "Inhale and brace before each descent; exhale steadily as you drive back up.",
    "tempo": "2-3 seconds down, optional 1-second pause at depth, about 1 second up.",
    "mistakes": [
      "Dumbbell drifting off your chest — becomes a front raise and tips you forward / pin it to your sternum with elbows down.",
      "Knees collapsing inward — stresses the knees and weakens the drive / track knees over toes and use elbows to pry them out.",
      "Heels lifting — weight rolls onto your toes and depth suffers / sit back slightly and grip the floor with your whole foot.",
      "Upper back rounding — dumps load onto your spine / lift your chest and keep your eyes forward."
    ],
    "safety": "Pick the dumbbell up from the floor with a flat back, and lower it by squatting down, not by bending over.",
    "video": {
      "title": "Dan John: Goblet Squat",
      "channel": "Laree Draper",
      "url": "https://www.youtube.com/watch?v=u7-HLtOglfM"
    }
  },
  "leg-press": {
    "setup": [
      "Adjust the seat back so your knees can bend past 90 degrees without your hips curling off the pad.",
      "Place feet shoulder-width apart, centered on the platform, toes turned slightly out.",
      "Check the safety catches and keep both hands on the side handles."
    ],
    "execution": [
      "Press the platform up, release the safety locks, and stop just short of locked knees.",
      "Inhale and lower the sled under control until your knees reach at least 90 degrees.",
      "Stop the descent the moment your glutes or lower back start curling off the pad.",
      "Keep your knees tracking in line with your toes throughout.",
      "Exhale and press through your whole foot, emphasizing the heels, to just short of lockout.",
      "After the final rep, re-engage the safety locks fully before releasing pressure."
    ],
    "breathing": "Inhale during the 2-3 second descent; exhale forcefully as you press; never hold your breath to grind a rep.",
    "tempo": "2-3 seconds down, smooth reversal, about 1 second up; no bouncing at the bottom.",
    "mistakes": [
      "Butt tucking at the bottom — rounds your lumbar spine under load / shorten the range or widen your stance slightly.",
      "Slamming into knee lockout — transfers load into the joint / stop just short of straight with quads still tensed.",
      "Knees caving inward — strains knee ligaments / lighten the sled and push your knees out over your toes.",
      "Stacking plates for quarter reps — tiny stimulus, big ego / use a load you can lower to 90 degrees under control.",
      "Pushing on your knees with your hands — hides a too-heavy load / grip the handles and drop the weight."
    ],
    "safety": "Never fully straighten and relax your knees mid-set, and always confirm the safety catches are locked before climbing out.",
    "video": {
      "title": "Leg Press",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=yZmx_Ac3880"
    }
  },
  "leg-extension": {
    "setup": [
      "Adjust the backrest so your knee joints line up with the machine's pivot point.",
      "Set the shin pad on your lower shins, just above your ankles.",
      "Sit tall with your back against the pad and grip the side handles."
    ],
    "execution": [
      "Exhale and extend both knees to a fully straight position in about one second.",
      "Pause for one second at the top, squeezing your quads hard.",
      "Inhale and lower for 2-3 seconds back toward 90 degrees of knee bend.",
      "Stop the stack just short of resting between reps to keep tension."
    ],
    "breathing": "Exhale as you extend, inhale as you lower; keep a steady rhythm rather than breath-holding.",
    "tempo": "1 second up, 1-second squeeze at full extension, 2-3 seconds down.",
    "mistakes": [
      "Hips lifting off the seat — momentum replaces quad work / lighten the load and keep your back pinned to the pad.",
      "Stopping short of full extension — misses the quads' hardest range / straighten completely, even if the weight must drop.",
      "Knee behind the machine pivot — misalignment shears the joint / re-set the backrest until your knee and the cam axis match.",
      "Letting the stack slam down — loses tension and yanks the knee into flexion / control every centimeter of the lowering."
    ],
    "safety": "If you have knee issues, start from no deeper than 90 degrees of bend and increase the load gradually.",
    "video": {
      "title": "Leg Extension",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=m0FOpMEgero"
    }
  },
  "bulgarian-split-squat": {
    "setup": [
      "Set a bench about knee height behind you; hold dumbbells at your sides.",
      "Stand one long step in front; place your rear foot on the bench, laces down.",
      "Keep your feet hip-width apart sideways, like train tracks, for balance."
    ],
    "execution": [
      "Inhale, brace, and lower straight down for 2-3 seconds.",
      "Descend until your rear knee hovers just above the floor and front thigh is near parallel.",
      "Keep the front knee tracking over your toes and the front heel planted.",
      "Lean your torso slightly forward to bias glutes, or stay upright to bias quads.",
      "Exhale and drive through the front whole foot to stand; the rear leg only balances.",
      "Finish all reps, then switch legs and match the count."
    ],
    "breathing": "Inhale on the way down, exhale as you drive up; keep a light brace throughout.",
    "tempo": "2-3 seconds down, brief pause just off the floor, about 1 second up.",
    "mistakes": [
      "Front foot too close to the bench — the knee slams past the toes and the heel lifts / lengthen your stance until the shin stays near vertical.",
      "Pushing off the rear leg — steals work from the front leg / rest only the shoelaces on the bench and keep that leg passive.",
      "Wobbling sideways — feet lined up on a tightrope / widen your side-to-side stance to hip width.",
      "Torso collapsing forward — loads the lower back, not the legs / brace your core and keep your chest proud."
    ],
    "safety": "Learn the movement with bodyweight first; with dumbbells, drop them to the sides rather than fighting a lost balance.",
    "video": {
      "title": "How To: Bulgarian Split Squat",
      "channel": "ScottHermanFitness",
      "url": "https://www.youtube.com/watch?v=2C-uNgKwPLE"
    }
  },
  "walking-lunge": {
    "setup": [
      "Hold dumbbells at your sides with a clear, flat runway of at least ten meters.",
      "Stand tall, feet hip-width apart, shoulders back, eyes ahead."
    ],
    "execution": [
      "Step forward about one full stride length, keeping your feet hip-width apart sideways.",
      "Bend both knees to roughly 90 degrees as you lower.",
      "Stop your rear knee 2-5 centimeters above the floor, torso tall.",
      "Keep the front knee tracking over your toes and the front heel down.",
      "Drive through the front foot and bring the rear leg through into the next lunge.",
      "Continue alternating legs with even step lengths for the set distance."
    ],
    "breathing": "Inhale as you lower into each lunge; exhale as you drive up and step through.",
    "tempo": "About 2 seconds down, no knee slam, then a smooth continuous stride up — walking rhythm, not a race.",
    "mistakes": [
      "Steps too short — the front knee shoots far past the toes and the heel lifts / lengthen the stride until both knees reach 90 degrees.",
      "Narrow tightrope steps — you tip sideways / keep your feet on separate 'train tracks' at hip width.",
      "Banging the rear knee on the floor — painful and uncontrolled / stop 2-5 centimeters short on every rep.",
      "Leaning and twisting the torso — the dumbbells swing and balance goes / brace your core and keep shoulders level."
    ],
    "safety": "Use dumbbells rather than a barbell until your balance is reliable, and drop them to the sides if you stumble.",
    "video": {
      "title": "Dumbbell Walking Lunge",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=eFWCn5iEbTU"
    }
  },
  "hack-squat": {
    "setup": [
      "Lie back with hips and lower back flat on the pad, shoulders snug under the shoulder pads.",
      "Place feet shoulder-width apart around the middle of the platform, toes slightly out.",
      "Stand up to take the weight and rotate the safety handles open."
    ],
    "execution": [
      "Inhale and brace, then lower the sled under control for 2-3 seconds.",
      "Descend until your thighs are at least parallel, hips staying on the pad.",
      "Track your knees in line with your toes and keep both feet flat.",
      "Exhale and press through mid-foot and heel to just short of knee lockout.",
      "After the last rep, rotate the safeties closed before easing the sled down."
    ],
    "breathing": "Inhale on the descent, exhale while pressing up; re-brace your core before every rep.",
    "tempo": "2-3 seconds down, controlled turnaround, about 1 second up; stop short of a slammed lockout.",
    "mistakes": [
      "Hips curling off the pad at depth — rounds the lumbar spine / stop at the depth where your hips stay glued down.",
      "Feet too low on the platform — knees shoot far forward and heels lift / move your feet up until heels stay planted.",
      "Locking the knees hard at the top — shifts load into the joint / keep a soft knee and constant quad tension.",
      "Quarter reps with maximal plates — minimal growth stimulus / cut the load and own at least parallel depth."
    ],
    "safety": "Learn the safety-handle direction before loading plates, and never release the handles until your legs are braced against the platform.",
    "video": {
      "title": "Hack Squat",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=rYgNArpwE7E"
    }
  },
  "step-up": {
    "setup": [
      "Choose a stable box at knee height or slightly lower; beginners start around mid-shin.",
      "Hold dumbbells at your sides and place your whole lead foot on the box, heel included."
    ],
    "execution": [
      "Shift your weight onto the lead leg and lean slightly toward it.",
      "Exhale and drive through the lead heel until that hip and knee fully extend.",
      "Keep the trail leg passive; do not hop or push off the floor.",
      "Tap the box lightly with the trail foot and stand tall on top.",
      "Step back down with the trail foot, lowering yourself over 2-3 controlled seconds.",
      "Complete all reps on one side, then switch legs."
    ],
    "breathing": "Exhale as you drive up; inhale as you lower back down under control.",
    "tempo": "Drive up in about 1 second, lower in 2-3 seconds; no bouncing between reps.",
    "mistakes": [
      "Kicking off the back foot — the calf boosts the rep and the target leg loafs / use the bottom toe as a light kickstand only.",
      "Box too high — hips hike and the knee grinds / lower the box until you can rise without lurching or pushing off.",
      "Half a foot on the box — the heel hangs and you press from the toes / plant the entire foot and drive through the heel.",
      "Dropping off the box — jarring landing and zero eccentric work / lower slowly with the top leg controlling the descent."
    ],
    "safety": "Use a box or bench rated for your weight that cannot slide or tip, and keep the dumbbells droppable at your sides.",
    "video": {
      "title": "How To Do A Dumbbell Step Up",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=DxUNi119Qzs"
    }
  },
  "bodyweight-squat": {
    "setup": [
      "Stand with feet slightly wider than hip width, toes turned out 15-30 degrees.",
      "Hold your arms at your sides, chest tall, eyes on a point a few meters ahead."
    ],
    "execution": [
      "Inhale and push your hips back as if reaching for a chair behind you.",
      "Bend knees and hips together, raising your arms forward as a counterbalance.",
      "Track your knees over your second toe, keeping heels and big toes pressed down.",
      "Lower until your thighs reach at least parallel while your back stays flat.",
      "Exhale and stand fully tall, squeezing your glutes at the top."
    ],
    "breathing": "Inhale on the way down, exhale on the way up; keep a light core brace throughout.",
    "tempo": "2 seconds down, brief pause at depth, 1 second up; use a 3-second descent for extra control work.",
    "mistakes": [
      "Knees caving inward — rehearses a harmful loading pattern / push your knees out over your second toe on every rep.",
      "Heels lifting at depth — balance rolls onto the toes / push your hips back further and stop where heels stay down.",
      "Torso collapsing forward — turns the squat into a back bend / reach your arms forward and keep your chest up.",
      "Fast shallow pulses — little strength or mobility gained / own full depth at a strict tempo before adding speed."
    ],
    "video": {
      "title": "How To Squat Correctly (FIX MISTAKES!)",
      "channel": "Squat University",
      "url": "https://www.youtube.com/watch?v=QDVdLGWktPU"
    }
  },
  "standing-calf-raise": {
    "setup": [
      "Adjust the shoulder pads so the machine unlocks when you stand tall with straight legs.",
      "Place the balls of both feet on the platform edge, heels hanging free, feet hip-width apart.",
      "Point your toes forward and stand up to release the safety catch."
    ],
    "execution": [
      "Lower your heels for 2-3 seconds until you feel a deep stretch below the platform.",
      "Hold the stretched bottom position for 1-2 seconds without bouncing.",
      "Exhale and drive up onto your tiptoes as high as possible in about one second.",
      "Squeeze the top contraction for a full second before the next descent.",
      "Keep your knees straight but not snapped into hyperextension throughout."
    ],
    "breathing": "Exhale as you rise, inhale as you lower into the stretch; keep breathing through the pauses.",
    "tempo": "2-3 seconds down, 1-2 second stretch hold, 1 second up, 1-second top squeeze.",
    "mistakes": [
      "Bouncing out of the bottom — the Achilles rebounds and the calves loaf / pause 1-2 seconds dead-still in the stretch.",
      "Half-rep pulses at the top — skips the growth-driving stretch / lower until your heels sink clearly below the platform.",
      "Bending and straightening the knees — the legs press the weight up / lock your knee angle so the ankles do all the work.",
      "Rushing the set — momentum hides weak calves / count the tempo silently on every rep."
    ],
    "safety": "Warm up with a lighter set first; a loaded deep calf stretch is productive but unforgiving on a cold Achilles tendon.",
    "video": {
      "title": "Calf Machine",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=N3awlEyTY98"
    }
  },
  "seated-calf-raise": {
    "setup": [
      "Sit with the balls of your feet on the platform, heels hanging off, knees bent 90 degrees.",
      "Adjust the knee pad snug on your lower thighs, just above the knees, never on kneecaps.",
      "Press up slightly onto your toes and swing the safety catch clear."
    ],
    "execution": [
      "Lower your heels for 2-3 seconds to the deepest stretch you can reach.",
      "Hold that stretch for 1-2 seconds without bouncing.",
      "Exhale and press up onto your toes as high as possible.",
      "Squeeze at the top for one second, then begin the next rep.",
      "Re-hook the safety catch before sliding your legs out."
    ],
    "breathing": "Exhale pressing up, inhale sinking into the stretch; keep breathing through both holds.",
    "tempo": "2-3 seconds down, 1-2 second bottom hold, 1 second up, 1-second squeeze.",
    "mistakes": [
      "Bouncing reps — tendon rebound does the work, not the soleus / pause completely at the bottom stretch on each rep.",
      "Pad set too loose or high — the unit rocks and power leaks / re-fit it tight to your thighs before starting.",
      "Tiny range of motion — the soleus never works through length / go full stretch to full tiptoe, every rep.",
      "Pad resting on the kneecaps — painful compression / position it on the lower thighs and re-check before loading."
    ],
    "safety": "Because the knees are bent, this targets the soleus; keep loads moderate and the range full rather than stacking plates for partials.",
    "video": {
      "title": "Seated Calf Raise | GI Exercise Guide",
      "channel": "Generation Iron Fitness & Bodybuilding Network",
      "url": "https://www.youtube.com/watch?v=7FCBISNuU9s"
    }
  },
  "single-leg-calf-raise": {
    "setup": [
      "Stand with the ball of one foot on the edge of a step, heel hanging free.",
      "Hook the other foot behind your working ankle.",
      "Rest your fingertips on a wall or rail for balance only."
    ],
    "execution": [
      "Lower your heel for 2-3 seconds until you reach a maximal stretch.",
      "Hold the bottom stretch for 1-2 seconds, completely still.",
      "Exhale and press up through the big-toe side to your tallest tiptoe.",
      "Squeeze for one second at the top, keeping the knee straight.",
      "Match reps on both legs; let the weaker side set the count."
    ],
    "breathing": "Exhale as you press up, inhale as you lower; never hold your breath during the stretch pause.",
    "tempo": "2-3 seconds down, 1-2 second stretch hold, 1 second up, 1-second top squeeze.",
    "mistakes": [
      "Pulling with the support hand — hides the strength gap / touch fingertips for balance, never for lift.",
      "Ankle rolling outward — strains the lateral ankle / press through the big toe and keep pressure centered.",
      "Bouncing the stretch — rebound robs the calf of tension / pause dead-still at the bottom before rising.",
      "Bending the knee — the quads help and the range shrinks / keep the leg straight with motion at the ankle only."
    ],
    "safety": "Use a stable step with support within reach; if balance is shaky, work from the floor before adding the deficit.",
    "video": {
      "title": "Single Leg Stair Claves",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=_gEx2ijsmNM"
    }
  },
  "treadmill-run": {
    "setup": [
      "Clip the emergency-stop safety cord to your waistband before touching the belt.",
      "Straddle the belt, start it at 2-3 mph, then step on walking.",
      "Set the incline to 1 percent to mimic outdoor air resistance."
    ],
    "execution": [
      "Warm up 5 minutes: brisk walk, then raise speed 0.5 mph each minute to jogging pace.",
      "Run tall with a slight whole-body forward lean; eyes ahead, not on the console.",
      "Land each foot under your hips with quick, light steps around 170-180 per minute.",
      "Swing your arms front-to-back at about 90 degrees; never grab the handrails while moving.",
      "Hold a conversational effort for 20-30 minutes as your main block.",
      "Cool down 3-5 minutes walking at 2-3 mph before stopping the belt."
    ],
    "breathing": "Breathe rhythmically, about two steps per inhale and two per exhale at easy paces; you should manage full sentences.",
    "tempo": "Cadence around 170-180 steps per minute; change speed only 0.5 mph at a time as form holds.",
    "mistakes": [
      "Holding the handrails — kills posture and falsifies the effort / slow the belt until you can run hands-free.",
      "Overstriding ahead of your hips — braking forces jar shins and knees / shorten your steps and raise cadence.",
      "Staring down at the console — the head drops and form collapses / keep eyes level and glance at metrics only briefly.",
      "Skipping the warm-up — cold calves and hamstrings strain at speed / walk five minutes first, every session.",
      "Chasing speed over form — sloppy mechanics rehearse injury / drop 0.5-1.0 mph whenever form frays."
    ],
    "safety": "Use the safety clip every session, never step off a moving belt, and straddle the belt before restarting it.",
    "video": {
      "title": "Treadmill Workout For Beginners",
      "channel": "The Run Experience",
      "url": "https://www.youtube.com/watch?v=9xL1aOqeVZw"
    }
  },
  "outdoor-run": {
    "setup": [
      "Wear cushioned running shoes and plan a flat, well-lit route for early sessions.",
      "Warm up with 3-5 minutes of brisk walking plus 20 meters each of high knees and butt kicks."
    ],
    "execution": [
      "Run tall and lean slightly forward from the ankles, not the waist.",
      "Land each foot under your center of mass with quick, light steps.",
      "Hold a cadence of roughly 170-180 steps per minute.",
      "Bend your arms about 90 degrees and drive them front-to-back, never across your chest.",
      "Keep an easy conversational pace 20-30 minutes; beginners alternate 1 minute running, 1 minute walking.",
      "Finish with 5 minutes of easy walking to cool down."
    ],
    "breathing": "Use a relaxed rhythm such as three steps in, two steps out; on easy runs you should speak full sentences.",
    "tempo": "Cadence about 170-180 steps per minute at an easy-run effort of 3-4 out of 10.",
    "mistakes": [
      "Overstriding with a big heel-first reach — brakes every step and loads the knees / land closer to your hips with quicker steps.",
      "Starting too fast — you blow up mid-run / make the first 10 minutes the slowest of the session.",
      "Arms swinging across the body — wastes energy and twists the torso / drive your elbows straight back.",
      "Bouncing stride — vertical motion adds impact, not speed / run over the ground and keep your head level."
    ],
    "safety": "Run facing traffic, wear reflective gear in low light, and add weekly running time gradually to avoid overuse injuries.",
    "video": {
      "title": "5 Minute Running Form Fix",
      "channel": "The Run Experience",
      "url": "https://www.youtube.com/watch?v=ZaGgtiTo3m0"
    }
  },
  "walking": {
    "setup": [
      "Wear supportive, flexible shoes; stand tall with ears, shoulders, and hips stacked.",
      "Begin with 5 minutes at an easy pace to warm up."
    ],
    "execution": [
      "Keep your head up and eyes forward, not on your feet.",
      "Lightly tighten your stomach and keep your back straight, not arched.",
      "Roll each step smoothly from heel through mid-foot, pushing off the toes.",
      "Swing your arms freely from the shoulders with a slight elbow bend.",
      "Walk briskly, about 100 steps per minute or 3-4 mph, for 20-40 minutes.",
      "Finish with 3-5 minutes easy plus a brief calf and hip flexor stretch."
    ],
    "breathing": "Breathe naturally and steadily; at a brisk pace you should be able to talk but not sing.",
    "tempo": "Brisk means roughly 100-130 steps per minute; add 1-minute fast intervals to raise intensity.",
    "mistakes": [
      "Staring at your phone — the head drops and posture folds / eyes up, phone away until the cooldown.",
      "Overstriding — long steps jar the heels and knees / take quicker, shorter steps landing under your body.",
      "Slouched shoulders and limp arms — cuts pace and power / bend the elbows slightly and swing with purpose.",
      "Strolling and calling it a workout — too easy to drive adaptation / hold the talk-but-not-sing effort for most minutes."
    ],
    "safety": "Face traffic when there is no sidewalk, wear reflective gear at night, and increase weekly time by only a few minutes.",
    "video": {
      "title": "How to Power Walk",
      "channel": "LIVESTRONG",
      "url": "https://www.youtube.com/watch?v=gknH0pQwohc"
    }
  },
  "cycling": {
    "setup": [
      "Set saddle height so your heel just reaches the pedal at the bottom with a straight leg.",
      "With the ball of your foot on the pedal, the extended knee should keep a 25-35 degree bend.",
      "With the pedals level, your forward kneecap should sit roughly over the pedal spindle.",
      "Set the bars at saddle height or higher and strap the balls of your feet over the pedal axles."
    ],
    "execution": [
      "Warm up with 5 minutes of easy spinning at low resistance.",
      "Pedal smooth circles at 80-90 rpm with your hips steady on the saddle.",
      "Keep a neutral spine, relaxed shoulders, and a light grip on the bars.",
      "Ride 20-30 minutes at conversational effort, or alternate 1 minute hard, 2 minutes easy, six times.",
      "Cool down with 5 minutes of easy spinning before dismounting."
    ],
    "breathing": "Breathe deep and steady; at endurance effort you should hold a conversation, during intervals only short phrases.",
    "tempo": "Hold 80-90 rpm and add resistance, not just speed, once your cadence is smooth.",
    "mistakes": [
      "Saddle too low — the knee stays overbent and aches at the front / raise it until the heel-on-pedal leg is straight.",
      "Saddle too high — hips rock and the back of the knee strains / lower it until your pelvis sits still.",
      "Mashing heavy resistance at low cadence — grinds the knees / spin 80-90 rpm and build resistance gradually.",
      "Death-gripping the bars — numb hands and tense shoulders / rest your hands lightly and keep elbows soft."
    ],
    "safety": "Secure straps or cleats before hard efforts, and stop to re-check your fit if you feel sharp knee pain.",
    "video": {
      "title": "15 Minute Beginner Indoor Cycling Session",
      "channel": "GCN Training",
      "url": "https://www.youtube.com/watch?v=fQqndzvURAU"
    }
  },
  "rowing-machine": {
    "setup": [
      "Set the damper between 3 and 5 for smooth, boat-like resistance, not maximal drag.",
      "Strap each foot so the strap crosses the ball of the foot; heels can lift freely.",
      "Set the monitor at eye level; grip the handle overhand, hands shoulder-width, wrists flat."
    ],
    "execution": [
      "Catch: shins vertical, arms long, chest up, torso hinged slightly forward to eleven o'clock.",
      "Drive: push with the legs first, holding your body angle and straight arms.",
      "As the handle passes your knees, swing the torso back to one o'clock.",
      "Finish: pull the handle to your lower ribs with legs flat and elbows past your sides.",
      "Recovery: arms away first, then hinge from the hips, then bend the knees, rolling forward slowly.",
      "Make the recovery twice as long as the drive, holding 18-24 strokes per minute.",
      "Session: 3-5 minutes easy, then 3 x 5 minutes steady with 1-2 minutes paddle, 3 minutes cool-down."
    ],
    "breathing": "Exhale through the drive, inhale on the recovery; never hold your breath as the rate climbs.",
    "tempo": "Drive about 1 second, recovery about 2 seconds — a 1:2 ratio at 18-24 strokes per minute.",
    "mistakes": [
      "Pulling with the arms early — the legs-body-arms order breaks and the back strains / finish the leg push before the arms bend.",
      "Sitting bolt upright at the catch with hips under shoulders — shortens every stroke / hinge forward from the hips with a proud chest.",
      "Leaning back past one o'clock — turns rowing into a sit-up / stop the layback at a slight recline.",
      "Throwing the head back off the catch — whips the spine / keep your eyes level on the monitor all stroke.",
      "Rushing the slide forward — the rhythm collapses and rest disappears / count 'one-two' up the slide on every recovery."
    ],
    "safety": "Keep your spine neutral, never rounded, at the catch; if your lower back aches, shorten the layback and lower the stroke rate.",
    "video": {
      "title": "How to Row Better By Doing NOTHING (Fundamentals: Part 1)",
      "channel": "Dark Horse Rowing",
      "url": "https://www.youtube.com/watch?v=jAXxIJMYFqI"
    }
  },
  "jump-rope": {
    "setup": [
      "Size the rope: stand on its middle; the handle tops should reach just below your armpits.",
      "Pick a forgiving surface such as rubber flooring or wood, never bare concrete, plus cushioned shoes.",
      "Hold the handles loosely at hip height, elbows tucked to your ribs, hands angled 45 degrees out."
    ],
    "execution": [
      "Stand tall with eyes forward; start the rope with a wrist flick, not the arms.",
      "Jump only 2-5 centimeters, one jump per rope turn, counting an even rhythm.",
      "Land softly on the balls of your feet, knees slightly bent, heels barely touching.",
      "Keep your elbows pinned and let the wrists spin the rope like ball bearings.",
      "Session: 30 seconds jumping, 10-30 seconds rest, for 6-10 rounds, 2-3 times weekly at first."
    ],
    "breathing": "Breathe in an even rhythm with the bounce; if you are gasping, slow the rope or lengthen the rest intervals.",
    "tempo": "One low bounce per turn at a steady beat, building toward roughly two turns per second.",
    "mistakes": [
      "Double-bouncing between turns — caps your speed ceiling / count single skips in a steady one-two-three-four rhythm.",
      "Jumping too high — extra impact and mistimed turns / clear the rope by only 2-5 centimeters.",
      "Swinging from the shoulders — the rope shortens and snags your feet / pin the elbows in and rotate from the wrists.",
      "Landing on your heels — invites shin splints / stay on the balls of your feet with soft knees.",
      "Rope too long — it drags overhead and trips you / re-size until the handles reach just below the armpits."
    ],
    "safety": "Jump on forgiving surfaces and cut volume at the first sign of shin or calf pain — it is a high-frequency impact skill.",
    "video": {
      "title": "Top 5 Beginner Jump Rope Mistakes",
      "channel": "Jump Rope Dudes",
      "url": "https://www.youtube.com/watch?v=z7sVfmzdUxk"
    }
  },
  "elliptical": {
    "setup": [
      "Place each foot fully on its pedal, toward the inside edges, with weight spread evenly.",
      "Set resistance low and incline mid-range while you learn the stride; adjust later.",
      "Stand tall and take a light grip on the moving handles or the fixed rails."
    ],
    "execution": [
      "Warm up for 5 minutes at easy resistance before real effort.",
      "Push through the whole foot, keeping your heels down as much as possible.",
      "Pump the handles at the same tempo as your legs, shoulders relaxed.",
      "Stay upright with eyes forward; no leaning on the console or locked elbows.",
      "Main block: 20-30 minutes conversational, or intervals of 30 seconds hard, 15 seconds easy.",
      "Add short backward-pedaling blocks to shift work toward hamstrings and glutes.",
      "Cool down 3-5 minutes at low resistance before stepping off."
    ],
    "breathing": "Keep breathing steady and conversational at moderate effort; intervals should still allow short phrases.",
    "tempo": "Cruise around 120-140 strides per minute at moderate resistance; raise resistance before you raise speed.",
    "mistakes": [
      "Slumping onto the handrails — unloads the legs and wrecks posture / stand tall with fingertip contact only.",
      "Riding on tiptoes — numb feet and burning calves / press your heels down through each stride.",
      "Spinning fast on near-zero resistance — momentum, not muscle / add resistance until every stride requires a push.",
      "Hunching toward the screen — strains neck and lower back / eyes forward with ears stacked over shoulders."
    ],
    "safety": "The pedals keep moving as you finish, so hold a rail while stepping on or off.",
    "video": {
      "title": "How to use an Elliptical Machine | Planet Fitness",
      "channel": "Planet Fitness",
      "url": "https://www.youtube.com/watch?v=sHMemwz_HPU"
    }
  },
  "stair-climber": {
    "setup": [
      "Start the machine at a slow step rate, roughly 40-50 steps per minute.",
      "Place at least your mid-foot on each pedal and rest fingertips on the rails for balance."
    ],
    "execution": [
      "Warm up 3-5 minutes at the easy rate before increasing.",
      "Stand tall with a slight natural hip hinge; never drape over the console.",
      "Take full, deliberate steps, pressing through heel and mid-foot each stride.",
      "Avoid letting the pedals hit their top or bottom stops.",
      "Main block: 15-25 minutes at a talk-test pace, or 1 minute brisk, 2 minutes easy, six times.",
      "Lower the rate for 3 minutes to cool down, then step off carefully."
    ],
    "breathing": "Breathe rhythmically; a steady climb should allow short sentences — if you cannot speak, drop the step rate.",
    "tempo": "Steady work sits around 40-70 steps per minute; beginners start with 5-10 minute sessions and build.",
    "mistakes": [
      "Hanging on the rails with locked arms — offloads bodyweight and inflates the calorie readout / fingertips only, posture tall.",
      "Quick shallow toe-taps — calves burn while glutes idle / take slower, fuller steps with heels pressing down.",
      "Hunching over the console — hips drift back and the lower back rounds / chest up, eyes ahead.",
      "Choosing a rate you cannot hold upright — form dies fast / drop levels until posture survives the whole block."
    ],
    "safety": "The pedals continue moving as you slow down — press stop and hold the rail before stepping off.",
    "video": {
      "title": "How to use a Stair Climber | Planet Fitness",
      "channel": "Planet Fitness",
      "url": "https://www.youtube.com/watch?v=ST-5lD69XqU"
    }
  },
  "romanian-deadlift": {
    "setup": [
      "Take the bar from a rack at mid-thigh height, or deadlift it once from the floor to standing.",
      "Grip double-overhand just outside your thighs; feet hip-width apart, toes forward.",
      "Stand tall with the bar touching your thighs; pull it into your legs to lock your lats."
    ],
    "execution": [
      "Unlock your knees to roughly 15 degrees and freeze that angle for the entire rep.",
      "Push your hips straight back and hinge forward, sliding the bar down your thighs.",
      "Keep the bar in skin contact, shins vertical, back flat, neck neutral.",
      "Stop between kneecap and mid-shin — the point where your hips can't travel further back without rounding.",
      "Drive your hips forward into the bar and stand tall; squeeze your glutes at lockout."
    ],
    "breathing": "Inhale and brace your core at the top before each descent; exhale as your hips drive through to lockout.",
    "tempo": "3 seconds down, no bottom pause, 1 second up — the slow eccentric is the growth stimulus.",
    "mistakes": [
      "Bar drifting off your thighs — every centimeter forward multiplies lumbar load / drag it down your legs the whole descent.",
      "Bending knees as you lower — the movement becomes a half squat and unloads the hamstrings / freeze 15 degrees, hips back only.",
      "Rounding your back chasing depth — spinal flexion under load risks disc injury / end the rep at your flat-back limit.",
      "Leaning back at the top — lumbar hyperextension isn't lockout / finish stacked: ribs down, glutes squeezed."
    ],
    "safety": "Your depth is set by hamstring flexibility, not the floor. Start near 50% of your deadlift weight and add slowly.",
    "video": {
      "title": "Do THIS For Proper RDL Technique",
      "channel": "Squat University",
      "url": "https://www.youtube.com/watch?v=KecWzqYscYc"
    }
  },
  "lying-leg-curl": {
    "setup": [
      "Adjust the machine so your kneecaps line up with its pivot point.",
      "Lie face down with the roller pad across your lower calves, just above your heels.",
      "Grip the handles, press your hips into the bench, and start from straight knees."
    ],
    "execution": [
      "Curl your heels toward your glutes until the pad touches, or nearly touches, them.",
      "Keep your hips glued to the bench — only your knees move.",
      "Squeeze hard for one second at full bend.",
      "Lower under control until your knees are almost straight, keeping tension on the hamstrings."
    ],
    "breathing": "Exhale as you curl your heels up; inhale during the slow lowering phase.",
    "tempo": "1 second up, 1 second squeeze, 3 seconds down — the eccentric does most of the muscle-building.",
    "mistakes": [
      "Hips lifting off the bench — hip flexion fakes extra range and unloads the hamstrings / lighten the load, pin your hips.",
      "Stopping short of straight knees — you skip the stretched position that drives growth / finish each lower nearly straight.",
      "Kicking the weight up — momentum robs tension and jars the knee / cut the load and curl smoothly.",
      "Pad set at mid-calf — it levers against the calf and limits knee bend / reposition just above your heels."
    ],
    "video": {
      "title": "Lying Leg Curl",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=n5WDXD_mpVY"
    }
  },
  "seated-leg-curl": {
    "setup": [
      "Adjust the seat so your kneecaps align with the machine's pivot point.",
      "Set the roller pad just above your heels; clamp the thigh pad snug above your knees.",
      "Sit fully back against the pad, legs straight, and grip the handles."
    ],
    "execution": [
      "Curl your heels down and under the seat as far as the machine allows.",
      "Keep your thighs and back pinned — movement happens only at the knees.",
      "Hold the fully curled position for one second.",
      "Return slowly all the way to straight knees, controlling the top stretch."
    ],
    "breathing": "Exhale as you curl down; inhale as the pad returns to the stretched position.",
    "tempo": "1 second curl, 1 second hold, 3 seconds back — never bounce out of the stretched top.",
    "mistakes": [
      "Thigh pad too loose — your hips lift and the hamstrings lose leverage / clamp it down before rep one.",
      "Cutting the return short — the stretched half of the rep grows the most muscle / extend to straight knees.",
      "Hauling on the handles with a rounding back — the pelvis tucks and slackens the hamstrings / sit tall, chest up.",
      "Bouncing at full stretch — ballistic reversals at long muscle length invite strains / pause, then curl."
    ],
    "video": {
      "title": "Seated Leg Curl",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=Orxowest56U"
    }
  },
  "stiff-leg-deadlift": {
    "setup": [
      "Hold dumbbells against the front of your thighs, palms facing your legs.",
      "Stand feet hip-width, toes forward, knees unlocked at about 10 degrees.",
      "Pull your shoulders back and brace your core before the first descent."
    ],
    "execution": [
      "Hinge at the hips, pushing them straight back while the dumbbells slide down your thighs.",
      "Hold the 10-degree knee bend — stiffer than a Romanian deadlift — through the whole rep.",
      "Lower to mid-shin, or stop the moment your lower back begins to round.",
      "Drag the dumbbells back up your legs as you drive your hips forward to stand."
    ],
    "breathing": "Inhale and brace before you hinge; hold through the descent; exhale driving up past mid-thigh.",
    "tempo": "3 seconds down, brief pause in the stretch, 1 second up — no rebound at the bottom.",
    "mistakes": [
      "Knees bending deeper as you descend — it becomes an RDL and shortens the stretch / freeze the knee angle.",
      "Dumbbells swinging ahead of your feet — the load shifts onto your lower back / keep them grazing your legs.",
      "Forcing the weights to the floor — depth with a rounded spine is worthless / stop at your flat-back limit.",
      "Snapping out of the bottom — fast reversals at peak stretch are how hamstrings strain / pause, then drive."
    ],
    "safety": "This lift loads hamstrings at their longest: warm up thoroughly, progress weight slowly, and never round your back to gain depth.",
    "video": {
      "title": "Dumbbell Stiff Legged Deadlift",
      "channel": "Renaissance Periodization",
      "url": "https://www.youtube.com/watch?v=cYKYGwcg0U8"
    }
  },
  "good-morning": {
    "setup": [
      "Set rack hooks at armpit height; safety pins just below the bar’s lowest point mid-rep.",
      "Pin your shoulder blades together to build a muscle shelf; grip just outside shoulder width.",
      "Step back, set feet shoulder-width with toes slightly out, and unlock your knees."
    ],
    "execution": [
      "Breathe, brace, then push your hips straight back to start the hinge.",
      "Tip your chest forward as one piece; shins vertical, back rigid.",
      "Descend until your torso nears parallel to the floor, or your hamstrings halt the hip travel.",
      "Drive your hips forward to raise your torso; the bar rises because your hips extend."
    ],
    "breathing": "Big inhale and full brace standing tall; hold it down and up; exhale only past the sticking point.",
    "tempo": "2–3 seconds down, no bounce, 1 second up. Reset your breath every rep.",
    "mistakes": [
      "Bar rolling onto your neck — it strains the cervical spine and pitches you forward / squeeze it into your rear delts.",
      "Knees traveling forward — the lift turns into a squat and the hamstrings unload / keep shins vertical, hips back.",
      "Rounding the back — the long torso lever multiplies spinal load / cut depth where rigidity ends.",
      "Loading it like a squat — heavy good mornings punish position errors brutally / start with the empty bar, add 2.5 kg jumps."
    ],
    "safety": "Set rack safety pins just below your bottom position — a failed good morning cannot be dumped safely.",
    "video": {
      "title": "How To Do A Barbell Good Morning",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=nWyx81AfTos"
    }
  },
  "nordic-curl": {
    "setup": [
      "Kneel on a pad and anchor your ankles under a loaded barbell, bench strap, or partner's hands.",
      "Set knees hip-width and stack shoulders over hips — one straight line down to your knees.",
      "Squeeze your glutes to lock your hips open; raise hands ready to catch."
    ],
    "execution": [
      "Lower your torso toward the floor as slowly as possible, pivoting only at the knees.",
      "Drive your heels up into the anchor and keep hips fully extended throughout.",
      "When the hamstrings give out, catch yourself in a push-up position.",
      "Push off the floor just enough for your hamstrings to pull you back upright."
    ],
    "breathing": "Exhale slowly and steadily through the descent; breathe in as you push back to the start.",
    "tempo": "Fight for a 3–5 second lower every rep — the eccentric is the exercise. Assist the return as needed.",
    "mistakes": [
      "Breaking at the hips — folding forward dumps hamstring tension and fakes progress / glutes tight, body one rigid line.",
      "Free-falling the second half — no stimulus, harsh landing / shorten the range: lower to a box or add band assistance.",
      "Too much volume too soon — Nordics cause severe soreness / start with 2–3 sets of 3–5 reps, twice weekly.",
      "A shaky ankle anchor — feet slipping mid-rep means a face-first fall / load-test the anchor before every set."
    ],
    "safety": "Brutal eccentric soreness is normal at first. Build from partial range over several weeks, and skip it during an active hamstring strain.",
    "video": {
      "title": "How to Set Up, Perform, & Program Nordic Hamstring Curls (Progressions | Regressions | Alternatives)",
      "channel": "E3 Rehab",
      "url": "https://www.youtube.com/watch?v=_e9vFU9-tkc"
    }
  },
  "hip-thrust": {
    "setup": [
      "Sit on the floor with your upper back against a bench about 40 cm (16 in) high.",
      "Roll a padded barbell into your hip crease and hold it centered with both hands.",
      "Set the bench edge just below your shoulder blades.",
      "Place feet hip-to-shoulder width so your shins finish vertical at the top."
    ],
    "execution": [
      "Tuck your chin and fix your eyes on the bar — hold that gaze all rep.",
      "Drive through your heels and extend your hips until your torso is parallel to the floor.",
      "Lock out with a posterior pelvic tilt — ribs down, tailbone tucked — and squeeze glutes one full second.",
      "Lower with control until the plates touch the floor, bar still centered over your hips."
    ],
    "breathing": "Inhale and brace before each rep; exhale forcefully as you squeeze through the lockout.",
    "tempo": "1 second up, 1 second tilted squeeze at the top, 2 seconds down.",
    "mistakes": [
      "Arching the lower back at lockout — lumbar hyperextension impersonates hip extension / tuck the pelvis, keep ribs down.",
      "Head thrown back, eyes on the ceiling — it drives rib flare and arching / stay chin-tucked, eyes on the bar.",
      "Feet set too far forward — hamstrings take over the drive / pull heels back until shins are vertical up top.",
      "Stopping shy of full extension — the lockout squeeze is the entire point / drop weight until you own it.",
      "Rising onto your toes — force leaks away from the glutes / screw your heels into the floor."
    ],
    "safety": "Pad the bar over your hip bones and wedge the bench against a rack or wall so it cannot slide.",
    "video": {
      "title": "Everything You Need to Know About the Hip Thrust",
      "channel": "Bret Contreras Glute Guy",
      "url": "https://www.youtube.com/watch?v=cBrHdatPj9g"
    }
  },
  "glute-bridge": {
    "setup": [
      "Lie on your back with knees bent, feet flat, hip-width apart, toes forward.",
      "Place your heels 15–20 cm (6–8 in) from your glutes.",
      "Rest arms at your sides, ribs down, lower back gently flattened toward the floor."
    ],
    "execution": [
      "Squeeze your glutes first, then press through your heels to lift your hips.",
      "Stop when knees, hips, and shoulders form one straight line — never arch higher.",
      "Hold the top for two seconds at maximum glute squeeze.",
      "Lower slowly until your glutes hover just above the floor, then go again."
    ],
    "breathing": "Exhale as your hips rise and through the top squeeze; inhale on the way down.",
    "tempo": "1 second up, 2-second squeeze, 2 seconds down — the hold is non-negotiable.",
    "mistakes": [
      "Arching past the straight line — the lower back finishes what the glutes should / stop at knee-hip-shoulder alignment.",
      "Hamstrings cramping and taking over — heels are too far out / walk your feet closer to your glutes.",
      "Driving through your toes — quads steal the movement / push through heels and let the toes go light.",
      "Pumping fast reps — momentum erases the squeeze that makes it work / own two full seconds up top."
    ],
    "video": {
      "title": "How to do a Glute bridge",
      "channel": "National Academy of Sports Medicine (NASM)",
      "url": "https://www.youtube.com/watch?v=SKOMwg1JLrU"
    }
  },
  "cable-kickback": {
    "setup": [
      "Strap the ankle cuff on and clip it to the lowest pulley setting.",
      "Face the stack, step back about half a meter, and hold the frame with both hands.",
      "Hinge your torso forward about 20 degrees and shift weight onto your support leg."
    ],
    "execution": [
      "Keeping a soft knee, drive your leg straight back from the hip.",
      "Stop at 20–30 degrees behind your body — full glute squeeze, zero lower-back arch.",
      "Hold one second with hips square to the machine.",
      "Return under control, letting your knee travel slightly forward for a stretch.",
      "Complete all reps, then switch the cuff to the other ankle."
    ],
    "breathing": "Exhale as you kick back; inhale as the leg returns forward.",
    "tempo": "1 second back, 1 second squeeze, 2 seconds return — the stack never touches down between reps.",
    "mistakes": [
      "Arching the back to kick higher — the extra height is lumbar motion, not glute / end the rep at the squeeze.",
      "Swinging the leg — momentum strips glute tension / slow both directions and shorten the arc.",
      "Hip rotating open — the turnout shifts load off the glute max / keep both hip bones facing the stack.",
      "Standing bolt upright — it caps your hip-extension range / hold the 20-degree forward hinge."
    ],
    "video": {
      "title": "How to: Cable Kickback (Glute Max) | Form Tutorial for Bigger Glutes",
      "channel": "Physique Development",
      "url": "https://www.youtube.com/watch?v=bVrmtCI00Ys"
    }
  },
  "sumo-deadlift": {
    "setup": [
      "Set feet 1.5–2 times shoulder width, toes turned out 30–45 degrees.",
      "Position the bar over mid-foot, roughly 3 cm (1 in) from your shins.",
      "Push your knees out over your toes and drop hips until your shins are vertical, viewed from the front.",
      "Grip the bar at shoulder width, arms hanging straight between your legs.",
      "Chest up, back flat — pull the slack out of the bar until it clicks against the plates."
    ],
    "execution": [
      "Take a big breath, brace, and drive the floor apart with both feet.",
      "Rise with hips and chest together, the bar dragging up your shins.",
      "Past the knees, squeeze your glutes and push your hips through to the bar.",
      "Stand fully tall — hips and knees straight, ribs stacked, no lean-back.",
      "Return by pushing hips back first, then bending knees once the bar passes them."
    ],
    "breathing": "Inhale and brace hard before the bar breaks the floor; hold through the rep; exhale after the plates land.",
    "tempo": "Patient 1–2 second grind off the floor, strong hip finish, 2-second controlled lower. Reset every rep.",
    "mistakes": [
      "Hips shooting up first — the back becomes the lever and knees cave / spread the floor from the first millimeter.",
      "Knees caving inward — lost external rotation strains the knees and kills leg drive / track knees over toes throughout.",
      "Yanking a slack bar — the jolt rounds your spine before liftoff / tension the bar until it clicks, then push.",
      "Bar drifting forward — distance from your shins multiplies lumbar load / drag it up your legs.",
      "Hyperextending the lockout — leaning back adds spinal stress, not completion / finish tall with glutes squeezed."
    ],
    "safety": "The bar travels up your shins — wear long socks or pants. End the set the moment your back position degrades.",
    "video": {
      "title": "How To Sumo Deadlift - feat. Mark Bell and Silent Mike",
      "channel": "Alan Thrall (Untamed Strength)",
      "url": "https://www.youtube.com/watch?v=aa0Y9y5ZAo4"
    }
  },
  "kettlebell-swing": {
    "setup": [
      "Stand feet shoulder-width or slightly wider, kettlebell about 30 cm (12 in) ahead of your toes.",
      "Hinge back with vertical shins and grip the handle with both hands, tilting the bell toward you.",
      "Flatten your back and pull your shoulders away from your ears — lats loaded."
    ],
    "execution": [
      "Hike the bell back high between your thighs, like a football snap.",
      "Snap your hips forward to stand in a hard plank — glutes and abs locked, knees straight.",
      "Let the bell float to chest height on straight, loose arms; your hips launched it.",
      "Stay tall until your forearms touch your hips, then hinge back for the next rep.",
      "Keep shins near vertical every cycle — hips travel back, knees barely bend.",
      "To finish, let a final hike settle to the floor with a flat back."
    ],
    "breathing": "Sniff air in through your nose during the hinge; exhale a sharp 'tss' timed exactly with each hip snap.",
    "tempo": "Explosive and rhythmic — about one swing every 1.5 seconds. Stop the set the moment power drops.",
    "mistakes": [
      "Squatting the swing — knees bending forward kills the hip snap / hinge: hips back, shins vertical, chest over knees.",
      "Lifting with the arms — the bell must be launched, not raised / arms stay loose ropes; snap the hips harder instead.",
      "Leaning back at the top — rib flare and lumbar hyperextension replace the plank / stand tall, glutes and abs braced.",
      "Hinging too early on the descent — the bell drops low and yanks your back / wait for forearm-to-hip contact."
    ],
    "safety": "Swing in clear space with dry hands, hardstyle to chest height only. If your lower back fatigues before your glutes, stop and regroove the hinge.",
    "video": {
      "title": "StrongFirst Kettlebell Swing: Timing the Hinge",
      "channel": "StrongFirst",
      "url": "https://www.youtube.com/watch?v=fvQoQsDk40M"
    }
  },
  "hip-abduction": {
    "setup": [
      "Sit down, back against the pad, feet on the footrests, pads outside your knees.",
      "Select a start width that gives slight tension with your knees together.",
      "Sit tall gripping the handles — or hinge slightly forward to bias the upper glutes."
    ],
    "execution": [
      "Press your knees apart as far as your hips allow.",
      "Keep your pelvis still and back against the pad — no rocking for extra range.",
      "Hold the widest position for one second, feeling the outer glutes.",
      "Return with control, stopping just before the weight stack touches down."
    ],
    "breathing": "Exhale as you press the pads apart; inhale as your knees return together.",
    "tempo": "1 second out, 1 second hold, 2 seconds back — constant tension, no stack slap.",
    "mistakes": [
      "Rocking the torso for range — momentum does the abductors' job / lighten the load and pin your back.",
      "Letting the stack slam between reps — tension resets to zero / stop a centimeter above touchdown.",
      "Heavy partial reps — short arcs undertrain the glute medius / choose a load that allows the full spread."
    ],
    "video": {
      "title": "How To Use The Seated Hip Abductor (Outer Thigh)",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=G_8LItOiZ0Q"
    }
  },
  "plank": {
    "setup": [
      "Place your forearms on the floor with elbows directly under your shoulders, hands flat or lightly clasped.",
      "Step both feet back to hip width, balancing on your toes.",
      "Form one straight line from head through heels; look at the floor to keep your neck neutral."
    ],
    "execution": [
      "Exhale hard and pull your ribcage down toward your pelvis, flattening your lower back.",
      "Squeeze your glutes and quads hard, tucking your pelvis slightly so your hips stay level.",
      "Press the floor away through your forearms so your upper back stays broad, not collapsed.",
      "Hold this rigid line, breathing steadily; end the set the moment your hips sag or hike."
    ],
    "breathing": "Never hold your breath: inhale through your nose and exhale slowly while keeping ribs pulled down and abs braced.",
    "tempo": "Static hold. Accumulate 20-60 seconds per set with full tension; add time only when your hips never sag.",
    "mistakes": [
      "Hips sagging — dumps stress into the lumbar spine / squeeze glutes, tuck pelvis, and stop when the line breaks.",
      "Ribs flared, back arched — abs disengage / exhale and pull your ribcage down toward your pelvis before and during the hold.",
      "Hips hiked high — turns the plank into a rest position / lower hips until shoulders, hips and heels align.",
      "Collapsing between the shoulder blades — shoulders bear the load passively / push the floor away, elbows under shoulders.",
      "Grinding out sloppy multi-minute holds — rehearses bad posture / do shorter, harder holds with ribcage-down, glutes-on tension."
    ],
    "safety": "If you feel pinching in your lower back, stop, reset the ribcage-down and glutes-on position, or regress to an incline plank.",
    "video": {
      "title": "How To Plank (Proper Form | Cues | Progressions)",
      "channel": "E3 Rehab",
      "url": "https://www.youtube.com/watch?v=A2b2EmIg0dA"
    }
  },
  "side-plank": {
    "setup": [
      "Lie on your side, propped on one forearm, elbow directly under your shoulder.",
      "Stack your feet, or place the top foot just in front of the bottom foot for balance.",
      "Rest your top hand on your hip or reach it toward the ceiling."
    ],
    "execution": [
      "Brace your core, then lift your hips until head, shoulders, hips and ankles form one straight line.",
      "Push the floor away through your forearm so the bottom shoulder stays tall, not sunken.",
      "Keep shoulders and hips stacked vertically — squeeze the bottom obliques and glutes so nothing rolls forward or back.",
      "Hold, lower with control, and repeat on the other side; end the set when your hips start dropping."
    ],
    "breathing": "Breathe steadily through the hold — slow nasal inhales, long exhales — without letting your ribs flare or your torso rotate.",
    "tempo": "Static hold. Start with 15-30 seconds per side and build toward 60 seconds with unbroken alignment.",
    "mistakes": [
      "Hips sagging toward the floor — the obliques stop working and the spine side-bends / lift until ankles, hips and shoulders align.",
      "Rolling the chest forward or backward — load shifts off the obliques / stack shoulders over hips; top hand on hip as a check.",
      "Elbow ahead of or behind the shoulder — strains the shoulder joint / reset the elbow directly beneath the shoulder.",
      "Sinking into the support shoulder — the joint collapses / press the forearm down and push the shoulder away from your ear."
    ],
    "safety": "If the bottom shoulder aches, shorten the holds or regress to a knees-bent side plank before returning to the full position.",
    "video": {
      "title": "Side Plank",
      "channel": "[P]rehab",
      "url": "https://www.youtube.com/watch?v=jfK1l-mNm3Q"
    }
  },
  "crunch": {
    "setup": [
      "Lie on your back with knees bent about 90 degrees, feet flat and hip-width apart.",
      "Place fingertips lightly behind your ears or cross your arms over your chest — never pull on your head.",
      "Press your lower back gently into the floor and tuck your chin slightly."
    ],
    "execution": [
      "Exhale and curl your shoulder blades three to four inches off the floor, ribs moving toward your pelvis.",
      "Keep your lower back glued to the floor; only the upper back rounds up.",
      "Pause one second at the top and squeeze your abs hard.",
      "Inhale and lower your shoulder blades back down with control — no bouncing off the floor."
    ],
    "breathing": "Exhale as you curl up, finishing the breath at the top squeeze; inhale as you lower back down.",
    "tempo": "1-2 seconds up, 1-second squeeze, 2 seconds down — no flinging the head or arms for momentum.",
    "mistakes": [
      "Pulling on the neck — strains the cervical spine and cheats the abs / keep fingertips light and lead with your chest.",
      "Sitting all the way up — hip flexors take over past the first few inches / stop once your shoulder blades clear the floor.",
      "Lower back lifting off the floor — the crunch becomes a lumbar-loading sit-up / keep it pressed down for every rep.",
      "Bouncing reps with momentum — tension leaves the abs / slow the lowering phase to a strict two-count."
    ],
    "safety": "With a history of neck pain, keep the chin gently tucked and the range small; stop if symptoms appear.",
    "video": {
      "title": "How to Do Crunches",
      "channel": "LIVESTRONG",
      "url": "https://www.youtube.com/watch?v=Xyd_fa5zoEU"
    }
  },
  "cable-crunch": {
    "setup": [
      "Attach a rope to a high pulley and kneel two to three feet back, facing the stack.",
      "Hold the rope ends against the sides of your head and keep them pinned there for the whole set.",
      "Set your hips as the fixed point: thighs near vertical, hips slightly flexed, spine neutral to slightly extended."
    ],
    "execution": [
      "Exhale and crunch your ribcage toward your pelvis, rounding your spine until your elbows travel toward mid-thigh.",
      "Keep your hips completely still — all movement comes from your spine flexing, not from hips rocking back.",
      "Hold the peak contraction for one second with abs fully shortened.",
      "Inhale and uncurl with control to the start, feeling a slight ab stretch, without letting the stack touch down."
    ],
    "breathing": "Exhale forcefully as you crunch down, finishing at peak contraction; inhale while returning to the stretched position.",
    "tempo": "1-2 seconds down into the crunch, 1-second hold, 2 seconds back up; keep constant cable tension.",
    "mistakes": [
      "Rocking at the hips — hip flexors and bodyweight move the load instead of abs / freeze hips and flex only your spine.",
      "Pulling with the arms — the rep becomes a pulldown / lock hands against your head; elbows just ride along with the torso.",
      "Going too heavy — jerky reps shift work to the lower back / pick a load allowing 12-15 slow, controlled reps.",
      "Cutting the top stretch short — abs lose their loaded lengthened position / uncurl fully to a slight stretch every rep."
    ],
    "safety": "Do not let the weight yank you into hard lumbar hyperextension at the top; return only to neutral or slightly extended.",
    "video": {
      "title": "Cable Crunch",
      "channel": "Jim Stoppani, PhD",
      "url": "https://www.youtube.com/watch?v=vsAPBAg0fb0"
    }
  },
  "hanging-knee-raise": {
    "setup": [
      "Grip a pull-up bar at shoulder width, palms forward, arms fully extended.",
      "Hang with shoulders lightly engaged — shoulder blades drawn slightly down — rather than sagging passively.",
      "Set a hollow position: exhale, ribs down, pelvis tucked, legs together and slightly in front of your body."
    ],
    "execution": [
      "Raise your knees at least to hip height — ideally toward your chest — curling your pelvis up at the end.",
      "Move only your legs and pelvis; the torso stays quiet with zero swing between reps.",
      "Lower your knees with control until your legs hang just short of vertical, keeping the hollow position.",
      "Pause briefly at the bottom to kill momentum before the next rep."
    ],
    "breathing": "Exhale as the knees rise, inhale as the legs lower; keep the abs braced so the ribs stay down throughout.",
    "tempo": "1 second up, brief squeeze, 2 seconds down, dead stop at the bottom — strict beats fast every time.",
    "mistakes": [
      "Swinging or kipping — momentum does the lifting and grooves sloppy reps / hang motionless at the bottom before each rep.",
      "Stopping at hip height — abs only load once the pelvis curls / keep pulling the knees up toward your chest.",
      "Hanging passively through the shoulders — the joint strains and the body swings / keep shoulder blades set and lats engaged.",
      "Arching the lower back at the bottom — the hollow is lost and abs unload / stop the legs just short of vertical."
    ],
    "safety": "End the set once you can no longer prevent swinging; flailing reps mostly stress your shoulders and lower back.",
    "video": {
      "title": "Hanging Leg Raises: Beginner To Advanced",
      "channel": "Tom Merrick",
      "url": "https://www.youtube.com/watch?v=or7KtIgxchE"
    }
  },
  "hanging-leg-raise": {
    "setup": [
      "Hang from a pull-up bar with a shoulder-width overhand grip, arms straight, feet clear of the floor.",
      "Engage your shoulders by drawing the shoulder blades slightly down — do not hang slack.",
      "Set a hollow body line: exhale, ribs down, pelvis tucked, legs together with toes slightly in front."
    ],
    "execution": [
      "Raise your straight legs in one controlled sweep until they are at least parallel — hips bent 90 degrees.",
      "Curl your pelvis toward your ribs at the top; that final tilt is what loads the abs hardest.",
      "Lower under strict control, resisting the whole way, until your legs return just short of vertical.",
      "Hold the bottom hollow motionless for a beat and start the next rep from a dead stop — no swing."
    ],
    "breathing": "Exhale through pursed lips as the legs rise and the pelvis curls; inhale on the descent without losing rib position.",
    "tempo": "1-2 seconds up, squeeze at the top, 2-3 seconds down; every rep starts from a still hang.",
    "mistakes": [
      "Swinging into each rep — momentum replaces ab work / dead-stop at the bottom and re-brace the hollow before every rep.",
      "Knees bending as you fatigue — the lever and range quietly shrink / end the set or switch to strict knee raises.",
      "No pelvic curl at the top — the lift stays all hip flexors / finish each rep by tilting the pelvis toward the ribs.",
      "Arching the lower back at the bottom — the hollow is lost and the lumbar strains / stop the legs short of vertical.",
      "Pulling with bent arms — the rep becomes a half pull-up / keep elbows locked and shoulders set; arms stay passive."
    ],
    "safety": "Earn this after strict hanging knee raises; tight hamstrings or a lost hollow force the lower back to round under load.",
    "video": {
      "title": "Hanging Leg Raise (Toes to Bar) Tutorial | Beginner Calisthenics Tutorials",
      "channel": "Simonster Strength",
      "url": "https://www.youtube.com/watch?v=S2QVHl6DoNo"
    }
  },
  "russian-twist": {
    "setup": [
      "Sit on the floor with knees bent and heels resting lightly on the ground.",
      "Lean your straight torso back about 45 degrees until your abs engage; chest proud, spine long.",
      "Hold your hands together at chest height, elbows wide.",
      "To progress, lift both feet so your shins are parallel to the floor."
    ],
    "execution": [
      "Rotate your ribcage to the right until your hands travel beside your hip, eyes following your hands.",
      "Keep hips and knees pointing straight up — rotation comes from the trunk, not from knees flopping sideways.",
      "Reverse through center and rotate left with the same control, counting one rep per side."
    ],
    "breathing": "Exhale as you rotate to each side; inhale passing back through center. Never hold your breath.",
    "tempo": "Deliberate — about one second per side; speed turns trunk rotation into low-back swinging.",
    "mistakes": [
      "Slumping into a rounded back — the lumbar spine absorbs the twist / hinge back with a long spine and lifted chest.",
      "Twisting from the lower back — it tolerates little rotation under load / rotate the ribcage over locked, stable hips.",
      "Knees swaying side to side — fakes rotation with zero oblique work / pin the knees skyward and move only the torso.",
      "Swinging only the arms — hands travel but the trunk never turns / rotate shoulders and chest as one unit with the hands."
    ],
    "safety": "With disc-related back issues keep the heels down, reduce the lean-back, and shorten the arc — or substitute dead bugs.",
    "video": {
      "title": "How To Do A Russian Twist",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=99T1EfpMwPA"
    }
  },
  "bicycle-crunch": {
    "setup": [
      "Lie on your back with fingertips resting lightly behind your ears, elbows wide.",
      "Lift your legs to tabletop — hips and knees at 90 degrees, shins parallel to the floor.",
      "Exhale, set your ribs down, press your lower back gently into the floor, and curl the shoulder blades just off it."
    ],
    "execution": [
      "Extend your right leg to about 45 degrees while drawing the left knee toward your chest.",
      "Rotate your torso so the right shoulder — not just the elbow — travels toward the left knee.",
      "Pause for a beat at the contraction; do not let the extended leg pull your lower back into an arch.",
      "Switch sides in one smooth pedal motion, alternating with control rather than speed."
    ],
    "breathing": "Exhale each time a shoulder rotates toward the opposite knee; take a quick inhale as you switch sides.",
    "tempo": "Slow pedaling — about one second per side with a distinct squeeze; racing reps just yank the neck.",
    "mistakes": [
      "Yanking the head with the hands — strains the neck and hides weak abs / touch the ears lightly and lead with the shoulder.",
      "Flapping elbows across — elbows move while the trunk never rotates / turn the ribcage so the shoulder chases the knee.",
      "Lower back arching as the leg extends — hip flexors take over and the lumbar loads / extend the leg higher until the back stays down.",
      "Sprinting through reps — momentum replaces oblique tension / slow to a one-second-per-side cadence with a pause."
    ],
    "safety": "If your lower back lifts off the floor, raise the extended-leg angle or shorten the set — back-pressed-down is the standard.",
    "video": {
      "title": "How To Do Bicycle Crunches For Beginners - The Proper Form, Muscle Building Benefits & Routine",
      "channel": "Fit Father Project - Fitness For Busy Fathers",
      "url": "https://www.youtube.com/watch?v=PAEo-zRSanM"
    }
  },
  "ab-wheel-rollout": {
    "setup": [
      "Kneel on a pad with the wheel on the floor, hands on the handles, arms straight.",
      "From tall kneeling, hinge slightly forward so your shoulders stack over the wheel.",
      "Before moving, brace hard: exhale, tuck your pelvis, squeeze your glutes, pull your ribs down — no lumbar arch."
    ],
    "execution": [
      "Roll the wheel forward slowly, arms and hips traveling together, holding the braced line from knees to head.",
      "Go only as far as you can with zero lower-back sag — range is earned, not forced.",
      "Pause one beat at your end range with everything tight.",
      "Drag the wheel back under your shoulders using abs and lats, exhaling as you return.",
      "Progress by rolling to a floor mark; move it a hand-width further only when every rep stays arch-free."
    ],
    "breathing": "Inhale and brace before each rollout; exhale steadily through the pull back; reset the breath every rep.",
    "tempo": "2-3 seconds out, brief pause, 2 seconds back for 6-10 reps. Master the full kneeling range before trying standing rollouts.",
    "mistakes": [
      "Reaching out before bracing — the lumbar arches the instant load hits / lock in ribs-down, glutes-on before the wheel moves.",
      "Hips sagging or lagging behind — back extensors take over / move hips and shoulders forward as one rigid unit.",
      "Rolling too far too soon — form collapses at end range / add distance gradually and stop where the brace holds.",
      "Bending the elbows to pull back — the rep becomes an arm exercise / keep arms straight; the abs bring you home.",
      "Piking the hips on the return — shortens the lever to cheat / hold the straight knee-to-head line while dragging back."
    ],
    "safety": "Sharp lower-back pain means the extension beat your brace — shorten the range immediately or roll toward a wall as a stop.",
    "video": {
      "title": "How To Do Ab Wheel Roll Outs",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=SBO5aFR09D4"
    }
  },
  "mountain-climber": {
    "setup": [
      "Set a high plank: hands under shoulders, arms straight, body one line from head to heels.",
      "Exhale, pull your ribs down, and squeeze your glutes so your trunk is locked before any leg moves.",
      "Spread your fingers and grip the floor; shoulders stay stacked over wrists the whole time."
    ],
    "execution": [
      "Drive one knee toward your chest, the foot hovering or tapping lightly under your hip.",
      "Switch legs in the air — extend the front leg back as the other knee drives in.",
      "Keep hips level with, or slightly below, your shoulders; the trunk stays dead still while the legs alternate.",
      "Move only as fast as your plank survives — feet land light and quiet."
    ],
    "breathing": "Breathe rhythmically with a short exhale on each knee drive; never hold your breath as the pace climbs.",
    "tempo": "Work in 20-30 second bursts. Earn speed: slow, silent switches first, then add tempo without bouncing the hips.",
    "mistakes": [
      "Hips piking upward — the core unloads and the knee drive shrinks / drop the hips until the shoulder-hip-heel line returns.",
      "Hips sagging with an arched back — the lumbar spine takes the impact / re-brace ribs down and glutes on, or slow down.",
      "Shoulders drifting back behind the wrists — trunk demand disappears / keep shoulders stacked directly over the wrists on every switch.",
      "Loud, bouncing feet — impact travels up through the spine as form unravels / land softly; quiet feet prove a stable trunk."
    ],
    "safety": "If your wrists complain, elevate your hands on a bench — the incline version keeps the pattern with less wrist extension.",
    "video": {
      "title": "How To Do Mountain Climbers",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=kLh-uczlPLg"
    }
  },
  "dead-bug": {
    "setup": [
      "Lie on your back with arms reaching straight up over your shoulders.",
      "Lift your legs to tabletop: hips and knees bent 90 degrees, shins parallel to the floor.",
      "Exhale hard to draw your ribs down and press your lower back flat into the floor — this position is the exercise."
    ],
    "execution": [
      "Slowly lower your right arm overhead while extending your left leg until both hover just above the floor.",
      "Travel only as far as the lower back stays pressed down; shrink the range the moment it lifts.",
      "Exhale and return the arm and leg to the start with the same control.",
      "Switch to the left arm and right leg, alternating sides for the full set."
    ],
    "breathing": "Use one long exhale for each extension — it keeps ribs down and back flat; inhale while returning to tabletop.",
    "tempo": "3-4 seconds per extension, 6-10 flawless reps per side; slower always beats bigger range.",
    "mistakes": [
      "Lower back lifting off the floor — the core disengaged and the lumbar takes over / shorten the range so contact never breaks.",
      "Ribs flaring as the arm goes overhead — the brace leaks / exhale first and reach only as far as the ribs stay down.",
      "Fast, jerky limb movement — momentum masks instability / slow every extension to a 3-4 second count.",
      "Holding your breath — pressure spikes and the brace collapses / sync a long, audible exhale with every extension."
    ],
    "video": {
      "title": "How to do a Dead Bug | Proper Form & Technique | NASM",
      "channel": "National Academy of Sports Medicine (NASM)",
      "url": "https://www.youtube.com/watch?v=bxn9FBrt4-A"
    }
  },
  "lying-leg-raise": {
    "setup": [
      "Lie on your back with legs straight, hands at your sides or palms down under your hips.",
      "Exhale, tilt your pelvis back slightly, and press your lower back into the floor before the first rep.",
      "Press your legs together, knees straight or very slightly bent — they stay that way all set."
    ],
    "execution": [
      "Raise your legs as one piece until they point at the ceiling — hips at about 90 degrees.",
      "Lower them slowly, resisting gravity the entire way down.",
      "Stop the descent just before your lower back peels off the floor — hover, never rest the heels.",
      "Start the next rep from that hover with zero bounce."
    ],
    "breathing": "Exhale as the legs rise; inhale slowly on the descent while keeping the lower back pressed down and ribs quiet.",
    "tempo": "1-2 seconds up, 3 seconds down — the lowering half is the rep, so never drop it.",
    "mistakes": [
      "Lower back arching as the legs descend — hip flexors tug the pelvis and the lumbar strains / shorten the descent or soften the knees.",
      "Resting heels on the floor between reps — all tension dies / hover an inch up and reverse from there.",
      "Bouncing the legs up with momentum — abs never actually load / lift from a dead hover at a strict tempo.",
      "Straining the head and neck upward — the neck starts bracing / keep the head down and chin gently tucked."
    ],
    "safety": "If your back arches even with hands under your hips, regress to bent-knee raises or dead bugs and rebuild the pressed-flat standard.",
    "video": {
      "title": "How to Do Leg Raises",
      "channel": "LIVESTRONG",
      "url": "https://www.youtube.com/watch?v=JB2oyawG9KI"
    }
  },
  "burpee": {
    "setup": [
      "Stand with feet shoulder-width apart, arms at your sides, weight over mid-foot.",
      "Clear one body-length of floor in front of you.",
      "Brace your core lightly — every rep passes through a plank."
    ],
    "execution": [
      "Squat down and plant both palms flat on the floor, just inside shoulder width, arms straight.",
      "Jump or step both feet back into a strong plank — hips level, lower back never sagging.",
      "Chest-to-floor version: lower your whole body to touch the floor, then press back up to the plank.",
      "Squat-thrust version: skip the floor touch and hold the plank — the faster, lower-impact standard.",
      "Jump both feet back toward your hands, landing flat-footed in a deep crouch.",
      "Stand and finish with a small vertical jump, arms overhead; land softly with knees tracking over toes.",
      "Flow straight into the next rep at an even pace."
    ],
    "breathing": "One breath per phase: inhale jumping back, exhale pressing up, inhale as the feet return, exhale on the jump.",
    "tempo": "A steady metronome pace you can hold for the whole set — clean plank and soft landing on every single rep.",
    "mistakes": [
      "Hips sagging in the plank phase — the lumbar spine absorbs every rep / brace before kicking back and hold one straight line.",
      "Worming up chest-first — the chest and hips rise separately and the back hinges / press up as one rigid unit, or use the squat-thrust version.",
      "Stiff, loud landings — knees and back eat the force / land soft and flat-footed with bent knees.",
      "Planting hands far ahead of the feet — shoulders overload as you kick back / plant palms roughly under your shoulders.",
      "Sprinting the first five reps — the pace crash destroys form / pick a speed you can repeat for the entire set."
    ],
    "safety": "If impact bothers your knees or back, step the feet back and forward instead of jumping and skip the hop — same pattern, less impact.",
    "video": {
      "title": "The Burpee",
      "channel": "CrossFit",
      "url": "https://www.youtube.com/watch?v=auBLPXO8Fww"
    }
  },
  "thruster": {
    "setup": [
      "Clean two dumbbells to your shoulders — one end resting on each shoulder, elbows lifted slightly forward.",
      "Set your feet shoulder-width apart, toes slightly out, weight over mid-foot.",
      "Take a breath and brace so your torso stays vertical in the squat."
    ],
    "execution": [
      "Squat with an upright torso until your thighs reach at least parallel, elbows staying lifted.",
      "Stand up explosively through the whole foot, using leg drive to launch the dumbbells off your shoulders.",
      "As the hips finish extending, press the dumbbells straight up to lockout — biceps beside your ears.",
      "Lower the dumbbells back to your shoulders and descend straight into the next squat in one rhythm."
    ],
    "breathing": "Inhale and brace at the top before each squat; exhale as you drive up and press to lockout.",
    "tempo": "One continuous rhythm — controlled descent, explosive drive, no pause between squat and press; the press rides the leg momentum.",
    "mistakes": [
      "Splitting it into squat, stop, then press — the pause kills the leg drive / start pressing the instant the hips snap open.",
      "Elbows dropping in the squat — the dumbbells tip forward and the back rounds / keep elbows lifted with weights racked solidly.",
      "Locking out with the dumbbells forward of your head — shoulders and lower back strain / finish with biceps beside the ears.",
      "Heels lifting during the drive — power leaks and the knees wobble / keep the whole foot down until the press begins.",
      "Leaning back to finish the press — the lumbar hyperextends under load / squeeze glutes and keep ribs down at lockout."
    ],
    "safety": "If you cannot lock out overhead without arching your lower back, lighten the load and address overhead mobility first.",
    "video": {
      "title": "Movement Demo - The Dumbbell Thruster",
      "channel": "Rogue Fitness",
      "url": "https://www.youtube.com/watch?v=1KYPZ-Jzo3w"
    }
  },
  "clean-and-press": {
    "setup": [
      "Stand with the bar over mid-foot, feet hip-width, shins about an inch from the bar.",
      "Grip just outside your legs with a pronated grip; hips higher than knees, shoulders slightly ahead of the bar.",
      "Flatten your back, set your lats, and take a big brace before pulling."
    ],
    "execution": [
      "Push the floor away, keeping the bar close and your back angle constant, until the bar passes the knees.",
      "Explode — extend hips, knees and ankles like a jump, shrugging hard as the bar accelerates.",
      "Pull yourself under, whip the elbows around and forward, and rack the bar on your front shoulders, knees bending to absorb.",
      "Stand tall, reset your feet under your hips, and re-brace.",
      "Press the bar overhead in a straight line, pulling your head back slightly, to full lockout over mid-foot.",
      "Lower the bar to your shoulders, then hinge it down to the floor with a flat back."
    ],
    "breathing": "Big breath and brace before the pull, hold it through the clean, re-breathe at the rack, exhale finishing the press.",
    "tempo": "Every rep from a still bar — fast, aggressive extension, crisp rack, controlled press; no touch-and-go rebounding.",
    "mistakes": [
      "Bar drifting away from the body — the load levers onto the lower back / drag it close enough to graze the thighs.",
      "Arms pulling early — the biceps steal the hip drive / arms hang like ropes until the hips finish extending.",
      "Reverse-curling the bar up — arms cannot clean meaningful load / extend the hips violently, then rack with elbows forward.",
      "Rounding the back off the floor — the spine, not the hips, lifts the bar / reset a flat back and constant angle first.",
      "Leaning far back in the press — the lumbar takes the load / squeeze glutes, keep ribs down, press in a straight line."
    ],
    "safety": "Learn this lift light and technique-first; a rounded-back pull or a soft, collapsing rack position gets injurious quickly under load.",
    "video": {
      "title": "How To Clean And Press",
      "channel": "PureGym",
      "url": "https://www.youtube.com/watch?v=KCe8l86-alA"
    }
  }
};

export function guideFor(exerciseId: string): ExerciseGuide | undefined {
  return EXERCISE_GUIDES[exerciseId];
}
