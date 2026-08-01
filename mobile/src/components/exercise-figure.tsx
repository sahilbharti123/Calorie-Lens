/**
 * Stylized movement-path animations (secondary aid — real photos and a
 * verified video are the primary demo for every built-in exercise).
 *
 * Every template was graded by a strict visual review for instant
 * readability: grounded contact points (feet, palms, forearms actually meet
 * the floor line), correct joint directions, distinct keyframes, and
 * equipment context (box, steps, dip bars, sled, wheel, pedals) where the
 * movement is meaningless without it.
 *
 * Convention: all segment angles are absolute, in degrees.
 *   0 = straight down · 90 = forward (facing direction) · 180 = straight up
 *   negative = backward.  endpoint = start + length × (sin a, cos a).
 */

import { useEffect, useRef, useState } from 'react';
import Svg, { Circle, G, Line, Path, Rect } from 'react-native-svg';

import type { FigureGear, FigureTemplate } from '@/src/lib/exercises';
import { useReducedMotion } from '@/src/lib/accessibility';
import { palette } from '@/src/theme';

type ArmPose = [number, number];
type LegPose = [number, number];

export type Pose = {
  hip: [number, number];
  torso: number;
  head?: number;
  armF: ArmPose;
  armB?: ArmPose;
  legF: LegPose;
  legB?: LegPose;
  footF?: number;
  footB?: number;
  /** Multiplies torso length (shrug etc.). */
  torsoScale?: number;
  /** Multiplies neck length (shrug shortens it). */
  neckScale?: number;
  /** Draw back limbs at full opacity (side plank top arm etc.). */
  boldBack?: boolean;
};

type Prop =
  | 'box'          // step-up box (right side)
  | 'steps'        // two ascending steps
  | 'dip-bars'     // parallel-bar hint at hand height
  | 'sled'         // 45° leg-press plate
  | 'seat'         // simple seat block under hips
  | 'wheel'        // ab wheel circle at front wrist
  | 'pedals'       // crank circle for cycling
  | 'pad-45'       // 45° back-extension pad
  | 'rope'         // jump-rope arc under the feet
  | 'arm-pad'      // preacher-curl arm pad
  | 'foot-block';  // raised foot platform (seated calf machine)

type Template = {
  frames: Pose[];
  /** ms per transition between frames (loops back to the first). */
  speed?: number;
  bench?: 'flat' | 'incline' | 'low';
  props?: Prop[];
};

const L = { torso: 25, neck: 3.4, head: 6.2, upperArm: 12.5, foreArm: 11.5, thigh: 16, shin: 15, foot: 6 };
const GROUND = 93.5;

function dir(angle: number): [number, number] {
  const r = (angle * Math.PI) / 180;
  return [Math.sin(r), Math.cos(r)];
}

function add(p: [number, number], angle: number, len: number): [number, number] {
  const [dx, dy] = dir(angle);
  return [p[0] + dx * len, p[1] + dy * len];
}

// ---------------------------------------------------------------------------
// Pose library — v2, strict-review corrected
// ---------------------------------------------------------------------------

const SQUAT_TOP: Pose = { hip: [48, 61], torso: 176, armF: [257, 108], legF: [2, 0], legB: [0, 0] };
const SQUAT_BOTTOM: Pose = { hip: [42, 76], torso: 152, armF: [255, 110], legF: [78, -18], legB: [74, -18], footF: 90 };
const FRONT_SQUAT_TOP: Pose = { ...SQUAT_TOP, torso: 178, armF: [55, 178] };
const FRONT_SQUAT_BOTTOM: Pose = { ...SQUAT_BOTTOM, torso: 172, armF: [65, 182] };
const GOBLET_TOP: Pose = { ...SQUAT_TOP, armF: [38, 112] };
const GOBLET_BOTTOM: Pose = { ...SQUAT_BOTTOM, torso: 160, armF: [46, 108] };

const HINGE_TOP: Pose = { hip: [48, 61], torso: 176, armF: [4, 1], legF: [2, 0], legB: [-1, 0] };
const HINGE_BOTTOM: Pose = { hip: [44, 68], torso: 118, armF: [3, 1], legF: [22, -12], legB: [20, -12] };
const GM_TOP: Pose = { hip: [48, 61], torso: 176, armF: [256, 104], legF: [2, 0], legB: [0, 0] };
const GM_BOTTOM: Pose = { hip: [44, 66], torso: 112, armF: [228, 96], legF: [16, -8], legB: [14, -8] };

const LUNGE_TOP: Pose = { hip: [46, 61], torso: 178, armF: [5, 2], legF: [4, 0], legB: [-2, 0] };
const LUNGE_BOTTOM: Pose = { hip: [47, 70], torso: 176, armF: [5, 2], legF: [55, -8], legB: [-20, -60], footB: -80 };

const STEP_BOTTOM: Pose = { hip: [44, 61], torso: 174, armF: [10, 6], legF: [56, 58], legB: [2, 0], footF: 92 };
const STEP_TOP: Pose = { hip: [64, 47], torso: 178, armF: [8, 4], legF: [3, 0], legB: [-2, 0] };

const BENCH_DOWN: Pose = { hip: [44, 69], torso: 90, head: 92, armF: [-50, 174], armB: [-50, 174], legF: [55, -15], legB: [50, -15] };
const BENCH_UP: Pose = { ...BENCH_DOWN, armF: [180, 180], armB: [180, 180] };
const INCLINE_DOWN: Pose = { hip: [42, 74], torso: 135, head: 140, armF: [-60, 170], armB: [-60, 170], legF: [75, -20], legB: [70, -20] };
const INCLINE_UP: Pose = { ...INCLINE_DOWN, armF: [178, 182], armB: [178, 182] };
const FLY_OPEN: Pose = { hip: [44, 69], torso: 90, head: 92, armF: [122, 118], armB: [122, 118], legF: [55, -15], legB: [50, -15] };
const FLY_CLOSED: Pose = { ...FLY_OPEN, armF: [181, 179], armB: [181, 179] };

const PUSHUP_TOP: Pose = { hip: [52, 74], torso: 100, head: 104, armF: [4, 2], armB: [4, 2], legF: [-58, -50], legB: [-58, -50], footF: -90, footB: -90 };
const PUSHUP_BOTTOM: Pose = { hip: [52, 80], torso: 99, head: 104, armF: [55, -35], armB: [55, -35], legF: [-64, -55], legB: [-64, -55], footF: -90, footB: -90 };

const DIP_TOP: Pose = { hip: [48, 50], torso: 174, armF: [4, 2], armB: [4, 2], legF: [15, -110], legB: [12, -110], footF: -150, footB: -150 };
const DIP_BOTTOM: Pose = { hip: [48, 58], torso: 168, armF: [-50, 55], armB: [-50, 55], legF: [15, -110], legB: [12, -110], footF: -150, footB: -150 };

const OHP_START: Pose = { hip: [48, 61], torso: 179, armF: [42, 190], armB: [42, 190], legF: [2, 0], legB: [-2, 0] };
const OHP_TOP: Pose = { hip: [48, 61], torso: 180, armF: [172, 182], armB: [172, 182], legF: [2, 0], legB: [-2, 0] };

const LATERAL_DOWN: Pose = { hip: [48, 61], torso: 179, armF: [10, 6], armB: [-8, -4], legF: [2, 0], legB: [-2, 0] };
const LATERAL_UP: Pose = { ...LATERAL_DOWN, armF: [86, 82], armB: [-86, -82], boldBack: true };
const FRONT_RAISE_DOWN: Pose = { hip: [48, 61], torso: 179, armF: [10, 6], legF: [2, 0], legB: [-2, 0] };
const FRONT_RAISE_UP: Pose = { ...FRONT_RAISE_DOWN, armF: [92, 90] };

const REAR_FLY_DOWN: Pose = { hip: [46, 64], torso: 118, armF: [3, 1], armB: [3, 1], legF: [16, -10], legB: [14, -10] };
const REAR_FLY_UP: Pose = { ...REAR_FLY_DOWN, armF: [-95, -88], armB: [-95, -88] };

const FACE_PULL_START: Pose = { hip: [48, 62], torso: 176, armF: [88, 86], armB: [88, 86], legF: [8, -4], legB: [-6, 0] };
const FACE_PULL_END: Pose = { ...FACE_PULL_START, armF: [98, -52], armB: [98, -52] };

const UPRIGHT_START: Pose = { hip: [48, 61], torso: 178, armF: [8, 3], armB: [8, 3], legF: [2, 0], legB: [-2, 0] };
const UPRIGHT_TOP: Pose = { ...UPRIGHT_START, armF: [58, -52], armB: [58, -52] };

const PULLUP_BOTTOM: Pose = { hip: [50, 62], torso: 178, armF: [172, 176], armB: [186, 190], legF: [6, -105], legB: [3, -105], footF: -160, footB: -160 };
const PULLUP_TOP: Pose = { hip: [50, 44], torso: 176, armF: [-62, 175], armB: [-58, 179], legF: [8, -108], legB: [5, -108], footF: -160, footB: -160 };

const PULLDOWN_START: Pose = { hip: [46, 74], torso: 172, armF: [165, 192], armB: [165, 192], legF: [85, -5], legB: [82, -5] };
const PULLDOWN_END: Pose = { ...PULLDOWN_START, torso: 168, armF: [98, 262], armB: [98, 262] };

const STRAIGHT_ARM_START: Pose = { hip: [46, 62], torso: 158, armF: [125, 125], armB: [125, 125], legF: [10, -6], legB: [6, -4] };
const STRAIGHT_ARM_END: Pose = { ...STRAIGHT_ARM_START, armF: [16, 16], armB: [16, 16] };

const ROW_DOWN: Pose = { hip: [46, 64], torso: 124, armF: [45, 15], armB: [45, 15], legF: [18, -12], legB: [16, -12] };
const ROW_UP: Pose = { ...ROW_DOWN, armF: [-25, 65], armB: [-25, 65] };

const SEATED_ROW_START: Pose = { hip: [44, 68], torso: 168, armF: [92, 88], armB: [92, 88], legF: [66, -32], legB: [64, -32] };
const SEATED_ROW_END: Pose = { ...SEATED_ROW_START, torso: 178, armF: [35, -25], armB: [35, -25] };

const SINGLE_ROW_DOWN: Pose = { hip: [46, 60], torso: 106, armF: [8, 4], armB: [25, 15], legF: [12, -8], legB: [78, -80] };
const SINGLE_ROW_UP: Pose = { ...SINGLE_ROW_DOWN, armF: [-35, 92] };

const SHRUG_DOWN: Pose = { hip: [48, 61], torso: 178, armF: [8, 3], armB: [8, 3], legF: [2, 0], legB: [-2, 0], torsoScale: 0.93, neckScale: 1.8 };
const SHRUG_UP: Pose = { hip: [48, 61], torso: 179, armF: [8, 3], armB: [8, 3], legF: [2, 0], legB: [-2, 0], torsoScale: 1.2, neckScale: 0.25 };

const CURL_DOWN: Pose = { hip: [48, 61], torso: 177, armF: [10, 6], armB: [10, 6], legF: [2, 0], legB: [-2, 0] };
const CURL_UP: Pose = { ...CURL_DOWN, armF: [10, 148], armB: [10, 148] };
const PREACHER_DOWN: Pose = { hip: [46, 68], torso: 168, armF: [55, 62], armB: [55, 62], legF: [72, -70], legB: [70, -70] };
const PREACHER_UP: Pose = { ...PREACHER_DOWN, armF: [55, 168], armB: [55, 168] };

const PUSHDOWN_START: Pose = { hip: [48, 61], torso: 174, armF: [15, 115], armB: [15, 115], legF: [4, 0], legB: [-3, 0] };
const PUSHDOWN_END: Pose = { ...PUSHDOWN_START, armF: [15, 8], armB: [15, 8] };

const OH_TRI_START: Pose = { hip: [48, 61], torso: 179, head: 172, armF: [-172, -55], armB: [-172, -55], legF: [3, 0], legB: [-3, 0] };
const OH_TRI_END: Pose = { ...OH_TRI_START, armF: [172, 178], armB: [180, 186] };

const SKULL_DOWN: Pose = { hip: [44, 69], torso: 90, head: 92, armF: [-155, -262], armB: [-155, -262], legF: [55, -15], legB: [50, -15] };
const SKULL_UP: Pose = { ...SKULL_DOWN, armF: [176, 174], armB: [176, 174] };

const WRIST_DOWN: Pose = { hip: [44, 74], torso: 118, armF: [-30, 35], armB: [-30, 35], legF: [78, 4], legB: [76, 4], footF: 90, footB: 90 };
const WRIST_UP: Pose = { ...WRIST_DOWN, armF: [-30, 125], armB: [-30, 125] };

const LEGPRESS_IN: Pose = { hip: [38, 72], torso: 147, armF: [-30, -10], armB: [-30, -10], legF: [100, 118], legB: [96, 118], footF: 35, footB: 35 };
const LEGPRESS_OUT: Pose = { ...LEGPRESS_IN, torso: 147, legF: [124, 120], legB: [120, 120], footF: 28, footB: 28 };

const LEGEXT_DOWN: Pose = { hip: [42, 68], torso: 172, armF: [45, 55], armB: [45, 55], legF: [92, 25], legB: [90, 25] };
const LEGEXT_UP: Pose = { ...LEGEXT_DOWN, legF: [92, 95], legB: [90, 25] };

const LEGCURL_LYING_START: Pose = { hip: [46, 69], torso: 90, head: 94, armF: [15, 60], armB: [15, 60], legF: [-90, -90], legB: [-90, -90] };
const LEGCURL_LYING_END: Pose = { ...LEGCURL_LYING_START, legF: [-90, -178], legB: [-90, -178] };

const LEGCURL_SEATED_START: Pose = { hip: [42, 68], torso: 172, armF: [45, 55], armB: [45, 55], legF: [92, 80], legB: [90, 80] };
const LEGCURL_SEATED_END: Pose = { ...LEGCURL_SEATED_START, legF: [92, -48], legB: [90, -48] };

const CALF_DOWN: Pose = { hip: [48, 61], torso: 178, armF: [8, 4], armB: [8, 4], legF: [2, 0], legB: [-2, 0], footF: 68, footB: 68 };
const CALF_UP: Pose = { hip: [48, 56], torso: 179, armF: [8, 4], armB: [8, 4], legF: [2, 0], legB: [-2, 0], footF: 142, footB: 142 };
const CALF_SEATED_DOWN: Pose = { hip: [44, 72], torso: 170, armF: [55, 65], armB: [55, 65], legF: [96, 22], legB: [94, 22], footF: 68, footB: 68 };
const CALF_SEATED_UP: Pose = { ...CALF_SEATED_DOWN, legF: [85, 22], legB: [83, 22], footF: 135, footB: 135 };

const HIPTHRUST_DOWN: Pose = { hip: [46, 76], torso: 122, head: 152, armF: [130, 165], armB: [130, 165], legF: [42, -55], legB: [38, -55] };
const HIPTHRUST_UP: Pose = { hip: [46, 63], torso: 102, head: 116, armF: [148, 175], armB: [148, 175], legF: [28, -80], legB: [24, -80] };
const BRIDGE_DOWN: Pose = { hip: [46, 89], torso: 89, head: 132, armF: [80, 85], armB: [80, 85], legF: [140, 14], legB: [136, 14] };
const BRIDGE_UP: Pose = { hip: [47, 78], torso: 60, head: 128, armF: [84, 88], armB: [84, 88], legF: [102, -6], legB: [98, -6] };

const KICKBACK_IN: Pose = { hip: [48, 61], torso: 168, armF: [55, 40], armB: [55, 40], legF: [4, 0], legB: [-2, 0] };
const KICKBACK_OUT: Pose = { ...KICKBACK_IN, torso: 160, legF: [-75, -85], legB: [-2, 0], footF: -130, boldBack: true };

const SIDE_LEG_IN: Pose = { hip: [48, 61], torso: 176, armF: [12, 6], armB: [12, 6], legF: [3, 0], legB: [-2, 0] };
const SIDE_LEG_OUT: Pose = { ...SIDE_LEG_IN, legF: [-58, -58], legB: [-2, 0], footF: -110, boldBack: true };

const SWING_BOTTOM: Pose = { hip: [46, 66], torso: 128, armF: [42, 20], armB: [42, 20], legF: [20, -14], legB: [18, -14] };
const SWING_TOP: Pose = { hip: [48, 61], torso: 176, armF: [88, 82], armB: [88, 82], legF: [2, 0], legB: [-2, 0] };

const CARRY_A: Pose = { hip: [48, 61], torso: 177, armF: [5, 1], armB: [-3, 1], legF: [16, -2], legB: [-14, -18], footB: 72, boldBack: true };
const CARRY_B: Pose = { hip: [48, 60], torso: 177, armF: [5, 1], armB: [-3, 1], legF: [3, -35], legB: [-2, 0], footF: 130, boldBack: true };

const HANG_POSE: Pose = { hip: [50, 62], torso: 178, armF: [173, 177], armB: [185, 189], legF: [6, -105], legB: [3, -105], footF: -160, footB: -160 };
const HANG_POSE_B: Pose = { ...HANG_POSE, hip: [50, 64.5], torsoScale: 1.04, legF: [10, -85], legB: [7, -85] };

const PLANK_A: Pose = { hip: [50, 83], torso: 94, head: 99, armF: [0, 90], armB: [0, 90], legF: [-79, -68], legB: [-79, -68], footF: -60, footB: -60 };
const PLANK_B: Pose = { ...PLANK_A, hip: [50, 82.2] };
const SIDE_PLANK_A: Pose = { hip: [50, 83], torso: 96, head: 101, armF: [0, 90], armB: [-172, -176], legF: [-79, -68], legB: [-79, -68], footF: -60, footB: -60, boldBack: true };
const SIDE_PLANK_B: Pose = { ...SIDE_PLANK_A, hip: [50, 82.2] };

const CRUNCH_DOWN: Pose = { hip: [46, 88], torso: 91, head: 94, armF: [-215, -95], armB: [-215, -95], legF: [130, 25], legB: [126, 25] };
const CRUNCH_UP: Pose = { hip: [46, 88], torso: 104, head: 138, armF: [-230, -104], armB: [-230, -104], legF: [130, 25], legB: [126, 25] };

const LEGRAISE_DOWN: Pose = { hip: [46, 88], torso: 91, head: 94, armF: [-95, -92], armB: [-95, -92], legF: [-91, -91], legB: [-91, -91] };
const LEGRAISE_UP: Pose = { ...LEGRAISE_DOWN, legF: [-178, -178], legB: [-178, -178] };

const HKR_DOWN: Pose = { hip: [50, 62], torso: 178, armF: [173, 177], armB: [185, 189], legF: [5, -60], legB: [2, -60], footF: -140, footB: -140 };
const HKR_UP: Pose = { ...HKR_DOWN, legF: [88, 3], legB: [84, 3], footF: -30, footB: -30 };

const TWIST_A: Pose = { hip: [48, 86], torso: 138, armF: [105, 100], armB: [105, 100], legF: [105, 40], legB: [101, 40] };
const TWIST_B: Pose = { ...TWIST_A, armF: [-20, -30], armB: [-20, -30] };

const ABWHEEL_IN: Pose = { hip: [40, 80], torso: 125, head: 132, armF: [30, 22], armB: [30, 22], legF: [40, -95], legB: [38, -95] };
const ABWHEEL_OUT: Pose = { hip: [48, 78], torso: 105, head: 112, armF: [40, 55], armB: [40, 55], legF: [13, -75], legB: [11, -75] };

const MTN_A: Pose = { hip: [52, 74], torso: 100, head: 104, armF: [4, 2], armB: [4, 2], legF: [95, -35], legB: [-58, -50], footB: -90 };
const MTN_B: Pose = { hip: [52, 74], torso: 100, head: 104, armF: [4, 2], armB: [4, 2], legF: [-58, -50], legB: [95, -35], footF: -90, boldBack: true };

const DEADBUG_A: Pose = { hip: [48, 89.5], torso: 91, head: 134, armF: [94, 92], armB: [174, 166], legF: [-95, -91], legB: [178, -88], boldBack: true };
const DEADBUG_B: Pose = { hip: [48, 89.5], torso: 91, head: 134, armF: [174, 166], armB: [174, 166], legF: [178, -88], legB: [174, -86], boldBack: true };

const BACKEXT_DOWN: Pose = { hip: [52, 74], torso: 55, head: 65, armF: [110, 205], armB: [110, 205], legF: [-42, -42], legB: [-44, -42] };
const BACKEXT_UP: Pose = { hip: [52, 74], torso: 138, head: 142, armF: [222, 100], armB: [222, 100], legF: [-42, -42], legB: [-44, -42] };

const NORDIC_UP: Pose = { hip: [46, 72], torso: 174, armF: [35, 65], armB: [35, 65], legF: [40, -95], legB: [38, -95] };
const NORDIC_DOWN: Pose = { hip: [50, 74], torso: 118, armF: [72, 85], armB: [72, 85], legF: [30, -92], legB: [28, -92] };

const BURPEE_STAND: Pose = { hip: [48, 61], torso: 178, armF: [8, 4], legF: [2, 0], legB: [-2, 0] };
const BURPEE_CROUCH: Pose = { hip: [44, 76], torso: 120, head: 126, armF: [30, 15], armB: [30, 15], legF: [80, -22], legB: [76, -22] };
const BURPEE_PLANK: Pose = { ...PUSHUP_BOTTOM };
const BURPEE_JUMP: Pose = { hip: [48, 52], torso: 180, armF: [172, 176], armB: [184, 188], legF: [6, -12], legB: [-6, -4], footF: 170, footB: 170 };

const THRUSTER_BOTTOM: Pose = { ...FRONT_SQUAT_BOTTOM };
const THRUSTER_TOP: Pose = { hip: [48, 61], torso: 180, armF: [172, 182], armB: [172, 182], legF: [2, 0], legB: [-2, 0] };

const RUN_A: Pose = { hip: [48, 59], torso: 168, armF: [55, 140], armB: [-40, 55], legF: [42, -60], legB: [-38, -35], footB: -110 };
const RUN_B: Pose = { hip: [48, 59], torso: 168, armF: [-40, 55], armB: [55, 140], legF: [-38, -35], legB: [42, -60], footF: -110, boldBack: true };
const WALK_A: Pose = { hip: [48, 61], torso: 175, armF: [24, 25], armB: [-18, 10], legF: [22, -10], legB: [-20, 8] };
const WALK_B: Pose = { hip: [48, 61], torso: 175, armF: [-18, 10], armB: [24, 25], legF: [-20, 8], legB: [22, -10], boldBack: true };

const CYCLE_A: Pose = { hip: [42, 58], torso: 142, armF: [82, 62], armB: [82, 62], legF: [72, 8], legB: [38, 25], footF: 95, footB: 95 };
const CYCLE_B: Pose = { hip: [42, 58], torso: 142, armF: [82, 62], armB: [82, 62], legF: [38, 25], legB: [72, 8], footF: 95, footB: 95, boldBack: true };

const ROWING_CATCH: Pose = { hip: [40, 70], torso: 150, armF: [88, 85], armB: [88, 85], legF: [82, -70], legB: [80, -70] };
const ROWING_FINISH: Pose = { hip: [46, 70], torso: 196, armF: [55, -30], armB: [55, -30], legF: [58, -12], legB: [56, -12] };

const JUMPROPE_A: Pose = { hip: [48, 61], torso: 178, armF: [38, 68], armB: [-25, -58], legF: [3, -6], legB: [-3, -6], boldBack: true };
const JUMPROPE_B: Pose = { hip: [48, 54], torso: 178, armF: [42, 78], armB: [-30, -66], legF: [4, -14], legB: [-4, -14], footF: 155, footB: 155, boldBack: true };

const STAIR_A: Pose = { hip: [44, 58], torso: 172, armF: [25, 20], armB: [-18, 8], legF: [52, -25], legB: [-4, 2] };
const STAIR_B: Pose = { hip: [58, 48], torso: 174, armF: [-18, 8], armB: [25, 20], legF: [4, 0], legB: [5, -100], footB: -150, boldBack: true };

const TEMPLATES: Record<FigureTemplate, Template> = {
  'squat': { frames: [SQUAT_TOP, SQUAT_BOTTOM] },
  'front-squat': { frames: [FRONT_SQUAT_TOP, FRONT_SQUAT_BOTTOM] },
  'goblet-squat': { frames: [GOBLET_TOP, GOBLET_BOTTOM] },
  'hinge': { frames: [HINGE_TOP, HINGE_BOTTOM] },
  'good-morning': { frames: [GM_TOP, GM_BOTTOM] },
  'lunge': { frames: [LUNGE_TOP, LUNGE_BOTTOM] },
  'step-up': { frames: [STEP_BOTTOM, STEP_TOP], speed: 800, props: ['box'] },
  'bench-press': { frames: [BENCH_DOWN, BENCH_UP], bench: 'flat' },
  'incline-press': { frames: [INCLINE_DOWN, INCLINE_UP], bench: 'incline' },
  'fly': { frames: [FLY_OPEN, FLY_CLOSED], bench: 'flat' },
  'pushup': { frames: [PUSHUP_TOP, PUSHUP_BOTTOM] },
  'dip': { frames: [DIP_TOP, DIP_BOTTOM], props: ['dip-bars'] },
  'overhead-press': { frames: [OHP_START, OHP_TOP] },
  'lateral-raise': { frames: [LATERAL_DOWN, LATERAL_UP] },
  'front-raise': { frames: [FRONT_RAISE_DOWN, FRONT_RAISE_UP] },
  'rear-fly': { frames: [REAR_FLY_DOWN, REAR_FLY_UP] },
  'face-pull': { frames: [FACE_PULL_START, FACE_PULL_END] },
  'upright-row': { frames: [UPRIGHT_START, UPRIGHT_TOP] },
  'pullup': { frames: [PULLUP_BOTTOM, PULLUP_TOP] },
  'pulldown': { frames: [PULLDOWN_START, PULLDOWN_END], props: ['seat'] },
  'straight-arm-pulldown': { frames: [STRAIGHT_ARM_START, STRAIGHT_ARM_END] },
  'bent-row': { frames: [ROW_DOWN, ROW_UP] },
  'seated-row': { frames: [SEATED_ROW_START, SEATED_ROW_END], props: ['seat'] },
  'single-arm-row': { frames: [SINGLE_ROW_DOWN, SINGLE_ROW_UP], bench: 'low' },
  'shrug': { frames: [SHRUG_DOWN, SHRUG_UP], speed: 600 },
  'curl': { frames: [CURL_DOWN, CURL_UP] },
  'preacher-curl': { frames: [PREACHER_DOWN, PREACHER_UP], props: ['seat', 'arm-pad'] },
  'pushdown': { frames: [PUSHDOWN_START, PUSHDOWN_END] },
  'overhead-triceps': { frames: [OH_TRI_START, OH_TRI_END] },
  'skullcrusher': { frames: [SKULL_DOWN, SKULL_UP], bench: 'flat' },
  'wrist-curl': { frames: [WRIST_DOWN, WRIST_UP], speed: 550, props: ['seat'] },
  'leg-press': { frames: [LEGPRESS_IN, LEGPRESS_OUT], props: ['sled', 'seat'] },
  'leg-extension': { frames: [LEGEXT_DOWN, LEGEXT_UP], props: ['seat'] },
  'leg-curl-lying': { frames: [LEGCURL_LYING_START, LEGCURL_LYING_END], bench: 'flat' },
  'leg-curl-seated': { frames: [LEGCURL_SEATED_START, LEGCURL_SEATED_END], props: ['seat'] },
  'calf-raise': { frames: [CALF_DOWN, CALF_UP], speed: 550 },
  'calf-seated': { frames: [CALF_SEATED_DOWN, CALF_SEATED_UP], speed: 550, props: ['seat', 'foot-block'] },
  'hip-thrust': { frames: [HIPTHRUST_DOWN, HIPTHRUST_UP], bench: 'low' },
  'glute-bridge': { frames: [BRIDGE_DOWN, BRIDGE_UP] },
  'kickback': { frames: [KICKBACK_IN, KICKBACK_OUT] },
  'side-leg-raise': { frames: [SIDE_LEG_IN, SIDE_LEG_OUT] },
  'swing': { frames: [SWING_BOTTOM, SWING_TOP], speed: 600 },
  'carry': { frames: [CARRY_A, CARRY_B], speed: 500 },
  'hang': { frames: [HANG_POSE, HANG_POSE_B], speed: 900 },
  'plank': { frames: [PLANK_A, PLANK_B], speed: 1000 },
  'side-plank': { frames: [SIDE_PLANK_A, SIDE_PLANK_B], speed: 1000 },
  'crunch': { frames: [CRUNCH_DOWN, CRUNCH_UP] },
  'leg-raise-lying': { frames: [LEGRAISE_DOWN, LEGRAISE_UP] },
  'hanging-knee-raise': { frames: [HKR_DOWN, HKR_UP] },
  'russian-twist': { frames: [TWIST_A, TWIST_B], speed: 550 },
  'ab-wheel': { frames: [ABWHEEL_IN, ABWHEEL_OUT], props: ['wheel'] },
  'mountain-climber': { frames: [MTN_A, MTN_B], speed: 380 },
  'dead-bug': { frames: [DEADBUG_A, DEADBUG_B], speed: 900 },
  'back-extension': { frames: [BACKEXT_DOWN, BACKEXT_UP], props: ['pad-45'] },
  'nordic-curl': { frames: [NORDIC_UP, NORDIC_DOWN], speed: 900 },
  'burpee': { frames: [BURPEE_STAND, BURPEE_CROUCH, BURPEE_PLANK, BURPEE_CROUCH, BURPEE_JUMP], speed: 450 },
  'thruster': { frames: [THRUSTER_BOTTOM, THRUSTER_TOP] },
  'run': { frames: [RUN_A, RUN_B], speed: 320 },
  'walk': { frames: [WALK_A, WALK_B], speed: 500 },
  'cycle': { frames: [CYCLE_A, CYCLE_B], speed: 400, props: ['pedals', 'seat'] },
  'rowing': { frames: [ROWING_CATCH, ROWING_FINISH], speed: 650, props: ['seat'] },
  'jump-rope': { frames: [JUMPROPE_A, JUMPROPE_B], speed: 300, props: ['rope'] },
  'stair': { frames: [STAIR_A, STAIR_B], speed: 480, props: ['steps'] },
};

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function easeInOut(t: number) {
  return 0.5 - Math.cos(Math.PI * t) / 2;
}

function mixPose(a: Pose, b: Pose, t: number): Pose {
  const armBA = a.armB ?? a.armF;
  const armBB = b.armB ?? b.armF;
  const legBA = a.legB ?? a.legF;
  const legBB = b.legB ?? b.legF;
  return {
    hip: [lerp(a.hip[0], b.hip[0], t), lerp(a.hip[1], b.hip[1], t)],
    torso: lerp(a.torso, b.torso, t),
    head: lerp(a.head ?? a.torso, b.head ?? b.torso, t),
    armF: [lerp(a.armF[0], b.armF[0], t), lerp(a.armF[1], b.armF[1], t)],
    armB: [lerp(armBA[0], armBB[0], t), lerp(armBA[1], armBB[1], t)],
    legF: [lerp(a.legF[0], b.legF[0], t), lerp(a.legF[1], b.legF[1], t)],
    legB: [lerp(legBA[0], legBB[0], t), lerp(legBA[1], legBB[1], t)],
    footF: lerp(a.footF ?? 90, b.footF ?? 90, t),
    footB: lerp(a.footB ?? 90, b.footB ?? 90, t),
    torsoScale: lerp(a.torsoScale ?? 1, b.torsoScale ?? 1, t),
    neckScale: lerp(a.neckScale ?? 1, b.neckScale ?? 1, t),
    boldBack: t > 0.5 ? b.boldBack : a.boldBack,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ExerciseFigure({
  template,
  gear = 'none',
  size = 200,
  tint = palette.ink,
  accent = palette.lime,
  paused = false,
}: {
  template: FigureTemplate;
  gear?: FigureGear;
  size?: number;
  tint?: string;
  accent?: string;
  paused?: boolean;
}) {
  const reducedMotion = useReducedMotion();
  const spec = TEMPLATES[template] ?? TEMPLATES.squat;
  const [pose, setPose] = useState<Pose>(() => mixPose(spec.frames[0], spec.frames[0], 0));
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    const active = TEMPLATES[template] ?? TEMPLATES.squat;
    const speed = active.speed ?? 850;
    const pauseAtEnds = 240;
    const legDuration = speed + pauseAtEnds;
    const count = active.frames.length;
    const total = legDuration * count;
    let start: number | null = null;
    if (paused || reducedMotion) {
      setPose(mixPose(active.frames[0], active.frames[1] ?? active.frames[0], 0.5));
      return;
    }
    let last = 0;
    const tick = (now: number) => {
      if (start === null) start = now;
      if (now - last > 33) {
        last = now;
        const elapsed = (now - start) % total;
        const leg = Math.floor(elapsed / legDuration);
        const within = elapsed - leg * legDuration;
        const t = Math.min(1, within / speed);
        const from = active.frames[leg % count];
        const to = active.frames[(leg + 1) % count];
        setPose(mixPose(from, to, easeInOut(t)));
      }
      frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    };
  }, [template, paused, reducedMotion]);

  // Forward kinematics ------------------------------------------------------
  const hip = pose.hip;
  const torsoLen = L.torso * (pose.torsoScale ?? 1);
  const neckLen = L.neck * (pose.neckScale ?? 1);
  const shoulder = add(hip, pose.torso, torsoLen);
  const headBase = add(shoulder, pose.head ?? pose.torso, neckLen);
  const headCenter = add(headBase, pose.head ?? pose.torso, L.head);
  const armB = pose.armB ?? pose.armF;
  const legB = pose.legB ?? pose.legF;
  const elbowF = add(shoulder, pose.armF[0], L.upperArm);
  const wristF = add(elbowF, pose.armF[1], L.foreArm);
  const elbowB = add(shoulder, armB[0], L.upperArm);
  const wristB = add(elbowB, armB[1], L.foreArm);
  const kneeF = add(hip, pose.legF[0], L.thigh);
  const ankleF = add(kneeF, pose.legF[1], L.shin);
  const toeF = add(ankleF, pose.footF ?? 90, L.foot);
  const kneeB = add(hip, legB[0], L.thigh);
  const ankleB = add(kneeB, legB[1], L.shin);
  const toeB = add(ankleB, pose.footB ?? 90, L.foot);

  const stroke = { strokeWidth: 3.4, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  const thin = { strokeWidth: 2.2, strokeLinecap: 'round' as const };
  const backOpacity = pose.boldBack ? 0.95 : 0.38;

  const gripMid: [number, number] = [(wristF[0] + wristB[0]) / 2, (wristF[1] + wristB[1]) / 2];
  const spec2 = TEMPLATES[template] ?? TEMPLATES.squat;
  const props = spec2.props ?? [];

  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      {/* stage */}
      <Line x1={6} y1={GROUND} x2={94} y2={GROUND} stroke={palette.lineHi} strokeWidth={2.4} strokeLinecap="round" />
      {spec2.bench === 'flat' ? (
        <Rect x={24} y={72} width={50} height={7} rx={3} fill={palette.lineHi} />
      ) : null}
      {spec2.bench === 'incline' ? (
        <Path d="M30 92 L46 92 L64 58 L58 53 Z" fill={palette.lineHi} />
      ) : null}
      {spec2.bench === 'low' ? (
        <>
          <Rect x={58} y={78} width={30} height={6} rx={3} fill={palette.lineHi} />
          <Line x1={62} y1={84} x2={62} y2={GROUND} stroke={palette.lineHi} strokeWidth={2.2} />
          <Line x1={84} y1={84} x2={84} y2={GROUND} stroke={palette.lineHi} strokeWidth={2.2} />
        </>
      ) : null}
      {props.includes('box') ? (
        <Rect x={52} y={79} width={32} height={GROUND - 79} fill={palette.lineHi} rx={1.5} />
      ) : null}
      {props.includes('steps') ? (
        <>
          <Rect x={54} y={80} width={40} height={GROUND - 80} fill={palette.lineHi} />
          <Rect x={70} y={66} width={24} height={GROUND - 66} fill={palette.lineHi} />
        </>
      ) : null}
      {props.includes('dip-bars') ? (
        <>
          <Line x1={36} y1={49} x2={64} y2={49} stroke={accent} strokeWidth={2.6} strokeLinecap="round" />
          <Line x1={40} y1={49} x2={40} y2={GROUND} stroke={palette.lineHi} strokeWidth={2.2} />
          <Line x1={60} y1={49} x2={60} y2={GROUND} stroke={palette.lineHi} strokeWidth={2.2} />
        </>
      ) : null}
      {props.includes('sled') ? (
        <Line x1={82} y1={38} x2={56} y2={80} stroke={palette.lineHi} strokeWidth={5} strokeLinecap="round" />
      ) : null}
      {props.includes('seat') ? (
        <Rect x={hip[0] - 9} y={hip[1] + 3} width={18} height={Math.max(4, GROUND - hip[1] - 3)} fill={palette.lineHi} rx={1.5} />
      ) : null}
      {props.includes('pedals') ? (
        <>
          <Circle cx={57} cy={83} r={8.5} fill="none" stroke={palette.lineHi} strokeWidth={2} />
          <Line x1={78} y1={46} x2={72} y2={78} stroke={palette.lineHi} strokeWidth={2.4} strokeLinecap="round" />
          <Line x1={74} y1={44} x2={82} y2={48} stroke={palette.lineHi} strokeWidth={2.6} strokeLinecap="round" />
        </>
      ) : null}
      {props.includes('pad-45') ? (
        <Path d={`M26 ${GROUND} L52 ${GROUND} L52 80 L38 80 Z`} fill={palette.lineHi} />
      ) : null}
      {props.includes('rope') ? (
        <Path d={`M ${hip[0] - 21} 84 Q ${hip[0]} ${GROUND + 4.5} ${hip[0] + 21} 84`} fill="none" stroke={accent} strokeWidth={1.6} />
      ) : null}
      {props.includes('arm-pad') ? (
        <Line x1={52} y1={63} x2={64} y2={72} stroke={palette.lineHi} strokeWidth={5.5} strokeLinecap="round" />
      ) : null}
      {props.includes('foot-block') ? (
        <Rect x={62} y={86.5} width={16} height={GROUND - 86.5} fill={palette.lineHi} rx={1} />
      ) : null}
      {gear === 'bar-overhead' ? (
        <Line x1={16} y1={13} x2={84} y2={13} stroke={accent} strokeWidth={3} strokeLinecap="round" />
      ) : null}
      {gear === 'cable-high' ? (
        <>
          <Rect x={88} y={6} width={7} height={10} rx={2} fill={palette.lineHi} />
          <Line x1={91} y1={12} x2={wristF[0]} y2={wristF[1]} stroke={accent} {...thin} strokeWidth={1.6} />
        </>
      ) : null}
      {gear === 'cable-low' ? (
        <>
          <Rect x={88} y={84} width={7} height={10} rx={2} fill={palette.lineHi} />
          <Line x1={91} y1={88} x2={wristF[0]} y2={wristF[1]} stroke={accent} {...thin} strokeWidth={1.6} />
        </>
      ) : null}

      {/* back limbs (depth) */}
      <G opacity={backOpacity}>
        <Line x1={shoulder[0]} y1={shoulder[1]} x2={elbowB[0]} y2={elbowB[1]} stroke={tint} {...stroke} />
        <Line x1={elbowB[0]} y1={elbowB[1]} x2={wristB[0]} y2={wristB[1]} stroke={tint} {...stroke} />
        <Line x1={hip[0]} y1={hip[1]} x2={kneeB[0]} y2={kneeB[1]} stroke={tint} {...stroke} />
        <Line x1={kneeB[0]} y1={kneeB[1]} x2={ankleB[0]} y2={ankleB[1]} stroke={tint} {...stroke} />
        <Line x1={ankleB[0]} y1={ankleB[1]} x2={toeB[0]} y2={toeB[1]} stroke={tint} {...stroke} strokeWidth={2.6} />
      </G>

      {/* torso */}
      <Line x1={hip[0]} y1={hip[1]} x2={shoulder[0]} y2={shoulder[1]} stroke={tint} {...stroke} strokeWidth={4.2} />

      {/* front limbs */}
      <Line x1={hip[0]} y1={hip[1]} x2={kneeF[0]} y2={kneeF[1]} stroke={tint} {...stroke} />
      <Line x1={kneeF[0]} y1={kneeF[1]} x2={ankleF[0]} y2={ankleF[1]} stroke={tint} {...stroke} />
      <Line x1={ankleF[0]} y1={ankleF[1]} x2={toeF[0]} y2={toeF[1]} stroke={tint} {...stroke} strokeWidth={2.6} />
      <Line x1={shoulder[0]} y1={shoulder[1]} x2={elbowF[0]} y2={elbowF[1]} stroke={tint} {...stroke} />
      <Line x1={elbowF[0]} y1={elbowF[1]} x2={wristF[0]} y2={wristF[1]} stroke={tint} {...stroke} />

      {/* joint dots for readability (skipped on locked-straight limbs) */}
      {Math.abs(pose.legF[0] - pose.legF[1]) > 3 ? <Circle cx={kneeF[0]} cy={kneeF[1]} r={1.4} fill={tint} /> : null}
      {Math.abs(pose.armF[0] - pose.armF[1]) > 3 ? <Circle cx={elbowF[0]} cy={elbowF[1]} r={1.4} fill={tint} /> : null}

      {/* head last, filled — masks overhead limbs so the face stays clean */}
      <Circle cx={headCenter[0]} cy={headCenter[1]} r={L.head} fill={palette.surface} stroke={tint} strokeWidth={3} />

      {/* gear at hands */}
      {props.includes('wheel') ? (
        <Circle cx={wristF[0]} cy={Math.min(wristF[1] + 2.5, GROUND - 4)} r={4} fill="none" stroke={accent} strokeWidth={2.4} />
      ) : null}
      {gear === 'barbell' ? (
        <>
          <Circle cx={gripMid[0]} cy={gripMid[1]} r={5.4} fill="none" stroke={accent} strokeWidth={2.6} />
          <Circle cx={gripMid[0]} cy={gripMid[1]} r={1.5} fill={accent} />
        </>
      ) : null}
      {gear === 'barbell-back' ? (
        <>
          <Circle cx={shoulder[0] - 2.5} cy={shoulder[1] - 2} r={4.6} fill="none" stroke={accent} strokeWidth={2.4} />
          <Circle cx={shoulder[0] - 2.5} cy={shoulder[1] - 2} r={1.3} fill={accent} />
        </>
      ) : null}
      {gear === 'barbell-front' ? (
        <>
          <Circle cx={shoulder[0] + 4.5} cy={shoulder[1] + 1} r={4.6} fill="none" stroke={accent} strokeWidth={2.4} />
          <Circle cx={shoulder[0] + 4.5} cy={shoulder[1] + 1} r={1.3} fill={accent} />
        </>
      ) : null}
      {gear === 'dumbbells' ? (
        <>
          <Line x1={wristF[0] - 3.4} y1={wristF[1]} x2={wristF[0] + 3.4} y2={wristF[1]} stroke={accent} strokeWidth={4.6} strokeLinecap="round" />
          <Line x1={wristB[0] - 3.4} y1={wristB[1]} x2={wristB[0] + 3.4} y2={wristB[1]} stroke={accent} strokeWidth={4.6} strokeLinecap="round" opacity={pose.boldBack ? 0.8 : 0.4} />
        </>
      ) : null}
      {gear === 'kettlebell' ? (
        <>
          <Path
            d={`M ${gripMid[0] - 3} ${gripMid[1]} a 3.2 3.2 0 0 1 6 0`}
            fill="none"
            stroke={accent}
            strokeWidth={1.8}
          />
          <Circle cx={gripMid[0]} cy={gripMid[1] + 4.6} r={4.2} fill={accent} />
        </>
      ) : null}
      {gear === 'machine' && !props.length ? (
        <Rect x={12} y={86} width={20} height={4} rx={2} fill={palette.lineHi} />
      ) : null}
    </Svg>
  );
}
