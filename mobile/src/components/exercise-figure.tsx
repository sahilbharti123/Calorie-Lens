/**
 * Animated exercise demonstrations.
 *
 * Every exercise in the library maps to a pose template: a set of skeleton
 * keyframes (joint angles in world space) that are interpolated on a loop.
 * The figure is a stylized side-view skeleton so users can see joint paths —
 * hip hinge vs squat, elbow pin vs swing — without licensing videos.
 *
 * Convention: all segment angles are absolute, in degrees.
 *   0 = straight down · 90 = forward (facing direction) · 180 = straight up
 *   270 (or -90) = backward.
 * A segment endpoint = start + length × (sin a, cos a) in SVG coordinates.
 */

import { useEffect, useRef, useState } from 'react';
import Svg, { Circle, G, Line, Path, Rect } from 'react-native-svg';

import type { FigureGear, FigureTemplate } from '@/src/lib/exercises';
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
};

type Template = {
  frames: Pose[];
  /** ms per transition between frames (loops back to the first). */
  speed?: number;
  bench?: 'flat' | 'incline' | 'low';
  ground?: boolean;
};

const L = { torso: 25, neck: 3.4, head: 6.2, upperArm: 12.5, foreArm: 11.5, thigh: 16, shin: 15, foot: 6 };

function dir(angle: number): [number, number] {
  const r = (angle * Math.PI) / 180;
  return [Math.sin(r), Math.cos(r)];
}

function add(p: [number, number], angle: number, len: number): [number, number] {
  const [dx, dy] = dir(angle);
  return [p[0] + dx * len, p[1] + dy * len];
}

// ---------------------------------------------------------------------------
// Pose library
// ---------------------------------------------------------------------------

const SQUAT_TOP: Pose = { hip: [48, 61], torso: 176, armF: [235, 140], legF: [2, 0], legB: [0, 0] };
const SQUAT_BOTTOM: Pose = { hip: [42, 76], torso: 152, armF: [242, 128], legF: [78, -18], legB: [74, -18], footF: 90 };
const FRONT_SQUAT_TOP: Pose = { ...SQUAT_TOP, armF: [55, 178] };
const FRONT_SQUAT_BOTTOM: Pose = { ...SQUAT_BOTTOM, torso: 162, armF: [70, 185] };
const GOBLET_TOP: Pose = { ...SQUAT_TOP, armF: [30, 120] };
const GOBLET_BOTTOM: Pose = { ...SQUAT_BOTTOM, torso: 158, armF: [40, 118] };

const HINGE_TOP: Pose = { hip: [48, 61], torso: 176, armF: [6, 2], legF: [2, 0], legB: [-1, 0] };
const HINGE_BOTTOM: Pose = { hip: [44, 68], torso: 118, armF: [52, 8], legF: [22, -12], legB: [20, -12] };
const GM_BOTTOM: Pose = { hip: [44, 66], torso: 112, armF: [245, 135], legF: [16, -8], legB: [14, -8] };

const LUNGE_TOP: Pose = { hip: [48, 60], torso: 178, armF: [6, 3], legF: [14, -4], legB: [-14, 4] };
const LUNGE_BOTTOM: Pose = { hip: [48, 74], torso: 176, armF: [6, 3], legF: [62, -35], legB: [-38, 40], footB: 130 };

const STEP_TOP: Pose = { hip: [50, 52], torso: 178, armF: [10, 6], legF: [4, 0], legB: [-4, 0] };
const STEP_BOTTOM: Pose = { hip: [44, 66], torso: 168, armF: [10, 6], legF: [58, -50], legB: [-8, 2] };

const BENCH_DOWN: Pose = { hip: [46, 58], torso: 92, head: 96, armF: [178, 262], armB: [178, 262], legF: [28, -62], legB: [22, -62] };
const BENCH_UP: Pose = { ...BENCH_DOWN, armF: [200, 178], armB: [200, 178] };
const INCLINE_DOWN: Pose = { hip: [45, 62], torso: 128, head: 140, armF: [195, 285], armB: [195, 285], legF: [40, -45], legB: [34, -45] };
const INCLINE_UP: Pose = { ...INCLINE_DOWN, armF: [212, 195], armB: [212, 195] };
const FLY_OPEN: Pose = { ...BENCH_DOWN, armF: [130, 220], armB: [130, 220] };
const FLY_CLOSED: Pose = { ...BENCH_DOWN, armF: [192, 175], armB: [192, 175] };

const PUSHUP_TOP: Pose = { hip: [50, 64], torso: 99, head: 104, armF: [172, 178], armB: [172, 178], legF: [-82, 2], legB: [-82, 2], footF: 12, footB: 12 };
const PUSHUP_BOTTOM: Pose = { hip: [50, 72], torso: 101, head: 106, armF: [128, 245], armB: [128, 245], legF: [-80, 2], legB: [-80, 2], footF: 12, footB: 12 };

const DIP_TOP: Pose = { hip: [48, 58], torso: 170, armF: [4, 2], armB: [4, 2], legF: [10, -55], legB: [6, -55] };
const DIP_BOTTOM: Pose = { hip: [48, 66], torso: 158, armF: [-42, 82], armB: [-42, 82], legF: [16, -62], legB: [12, -62] };

const OHP_START: Pose = { hip: [48, 61], torso: 179, armF: [42, 190], armB: [42, 190], legF: [2, 0], legB: [-2, 0] };
const OHP_TOP: Pose = { hip: [48, 61], torso: 180, armF: [172, 182], armB: [172, 182], legF: [2, 0], legB: [-2, 0] };

const LATERAL_DOWN: Pose = { hip: [48, 61], torso: 179, armF: [14, 8], armB: [14, 8], legF: [2, 0], legB: [-2, 0] };
const LATERAL_UP: Pose = { ...LATERAL_DOWN, armF: [96, 88], armB: [-64, -60] };
const FRONT_RAISE_UP: Pose = { ...LATERAL_DOWN, armF: [92, 90], armB: [14, 8] };

const REAR_FLY_DOWN: Pose = { hip: [46, 64], torso: 122, armF: [78, 40], armB: [78, 40], legF: [16, -10], legB: [14, -10] };
const REAR_FLY_UP: Pose = { ...REAR_FLY_DOWN, armF: [148, 110], armB: [-10, -35] };

const FACE_PULL_START: Pose = { hip: [48, 62], torso: 172, armF: [82, 95], armB: [82, 95], legF: [8, -4], legB: [-6, 0] };
const FACE_PULL_END: Pose = { ...FACE_PULL_START, armF: [118, 22], armB: [118, 22] };

const UPRIGHT_START: Pose = { hip: [48, 61], torso: 178, armF: [10, 5], armB: [10, 5], legF: [2, 0], legB: [-2, 0] };
const UPRIGHT_TOP: Pose = { ...UPRIGHT_START, armF: [78, -25], armB: [78, -25] };

const PULLUP_BOTTOM: Pose = { hip: [50, 56], torso: 176, armF: [178, 182], armB: [178, 182], legF: [4, -18], legB: [0, -18] };
const PULLUP_TOP: Pose = { hip: [50, 40], torso: 174, armF: [128, 258], armB: [128, 258], legF: [8, -26], legB: [4, -26] };

const PULLDOWN_START: Pose = { hip: [48, 66], torso: 172, armF: [162, 195], armB: [162, 195], legF: [78, -75], legB: [74, -75] };
const PULLDOWN_END: Pose = { ...PULLDOWN_START, torso: 166, armF: [105, 275], armB: [105, 275] };

const STRAIGHT_ARM_START: Pose = { hip: [46, 62], torso: 158, armF: [128, 132], armB: [128, 132], legF: [10, -6], legB: [6, -4] };
const STRAIGHT_ARM_END: Pose = { ...STRAIGHT_ARM_START, armF: [30, 28], armB: [30, 28] };

const ROW_DOWN: Pose = { hip: [46, 64], torso: 124, armF: [62, 30], armB: [62, 30], legF: [18, -12], legB: [16, -12] };
const ROW_UP: Pose = { ...ROW_DOWN, armF: [36, -55], armB: [36, -55] };

const SEATED_ROW_START: Pose = { hip: [44, 68], torso: 168, armF: [92, 88], armB: [92, 88], legF: [68, -40], legB: [66, -40] };
const SEATED_ROW_END: Pose = { ...SEATED_ROW_START, torso: 176, armF: [40, -18], armB: [40, -18] };

const SINGLE_ROW_DOWN: Pose = { hip: [46, 60], torso: 106, armF: [24, 12], armB: [95, 92], legF: [12, -8], legB: [78, -80] };
const SINGLE_ROW_UP: Pose = { ...SINGLE_ROW_DOWN, armF: [12, -68] };

const SHRUG_DOWN: Pose = { hip: [48, 61.5], torso: 177, armF: [8, 3], armB: [8, 3], legF: [2, 0], legB: [-2, 0] };
const SHRUG_UP: Pose = { hip: [48, 60], torso: 179, head: 176, armF: [8, 3], armB: [8, 3], legF: [2, 0], legB: [-2, 0] };

const CURL_DOWN: Pose = { hip: [48, 61], torso: 177, armF: [10, 6], armB: [10, 6], legF: [2, 0], legB: [-2, 0] };
const CURL_UP: Pose = { ...CURL_DOWN, armF: [10, 148], armB: [10, 148] };
const PREACHER_DOWN: Pose = { hip: [46, 68], torso: 168, armF: [62, 78], armB: [62, 78], legF: [72, -70], legB: [70, -70] };
const PREACHER_UP: Pose = { ...PREACHER_DOWN, armF: [62, 172], armB: [62, 172] };

const PUSHDOWN_START: Pose = { hip: [48, 61], torso: 174, armF: [22, 118], armB: [22, 118], legF: [4, 0], legB: [-3, 0] };
const PUSHDOWN_END: Pose = { ...PUSHDOWN_START, armF: [20, 26], armB: [20, 26] };

const OH_TRI_START: Pose = { hip: [48, 61], torso: 179, armF: [175, 262], armB: [175, 262], legF: [3, 0], legB: [-3, 0] };
const OH_TRI_END: Pose = { ...OH_TRI_START, armF: [176, 184], armB: [176, 184] };

const SKULL_DOWN: Pose = { ...BENCH_DOWN, armF: [162, 258], armB: [162, 258] };
const SKULL_UP: Pose = { ...BENCH_DOWN, armF: [168, 172], armB: [168, 172] };

const WRIST_DOWN: Pose = { hip: [46, 68], torso: 158, armF: [58, 84], armB: [58, 84], legF: [70, -72], legB: [68, -72] };
const WRIST_UP: Pose = { ...WRIST_DOWN, armF: [58, 66], armB: [58, 66] };

const LEGPRESS_IN: Pose = { hip: [42, 66], torso: 138, armF: [30, 40], armB: [30, 40], legF: [86, -68], legB: [82, -68] };
const LEGPRESS_OUT: Pose = { ...LEGPRESS_IN, legF: [62, -14], legB: [58, -14] };

const LEGEXT_DOWN: Pose = { hip: [44, 64], torso: 168, armF: [40, 55], armB: [40, 55], legF: [70, -78], legB: [68, -78] };
const LEGEXT_UP: Pose = { ...LEGEXT_DOWN, legF: [76, -4], legB: [68, -78] };

const LEGCURL_LYING_START: Pose = { hip: [48, 62], torso: 96, head: 100, armF: [140, 210], armB: [140, 210], legF: [-84, -4], legB: [-84, -4] };
const LEGCURL_LYING_END: Pose = { ...LEGCURL_LYING_START, legF: [-84, -118], legB: [-84, -118] };

const LEGCURL_SEATED_START: Pose = { ...LEGEXT_DOWN, legF: [72, -12], legB: [72, -12] };
const LEGCURL_SEATED_END: Pose = { ...LEGEXT_DOWN, legF: [72, -85], legB: [72, -85] };

const CALF_DOWN: Pose = { hip: [48, 60], torso: 178, armF: [8, 4], armB: [8, 4], legF: [2, 0], legB: [-2, 0], footF: 74 };
const CALF_UP: Pose = { hip: [48, 56], torso: 179, armF: [8, 4], armB: [8, 4], legF: [2, 0], legB: [-2, 0], footF: 118 };
const CALF_SEATED_DOWN: Pose = { hip: [44, 66], torso: 172, armF: [45, 60], armB: [45, 60], legF: [74, -78], legB: [72, -78], footF: 70 };
const CALF_SEATED_UP: Pose = { ...CALF_SEATED_DOWN, hip: [44, 65], legF: [74, -82], legB: [72, -82], footF: 122 };

const HIPTHRUST_DOWN: Pose = { hip: [46, 74], torso: 122, head: 130, armF: [130, 165], armB: [130, 165], legF: [42, -55], legB: [38, -55] };
const HIPTHRUST_UP: Pose = { hip: [46, 62], torso: 102, head: 112, armF: [148, 175], armB: [148, 175], legF: [28, -80], legB: [24, -80] };
const BRIDGE_DOWN: Pose = { hip: [48, 80], torso: 112, head: 118, armF: [55, 40], armB: [55, 40], legF: [35, -55], legB: [31, -55] };
const BRIDGE_UP: Pose = { hip: [48, 70], torso: 96, head: 104, armF: [70, 55], armB: [70, 55], legF: [22, -78], legB: [18, -78] };

const KICKBACK_IN: Pose = { hip: [48, 61], torso: 164, armF: [65, 45], armB: [65, 45], legF: [2, 0], legB: [6, -4] };
const KICKBACK_OUT: Pose = { ...KICKBACK_IN, legB: [-42, 6] };

const SIDE_LEG_IN: Pose = { hip: [48, 61], torso: 176, armF: [12, 6], armB: [12, 6], legF: [2, 0], legB: [-2, 0] };
const SIDE_LEG_OUT: Pose = { ...SIDE_LEG_IN, legB: [-52, 10] };

const SWING_BOTTOM: Pose = { hip: [46, 66], torso: 128, armF: [42, 20], armB: [42, 20], legF: [20, -14], legB: [18, -14] };
const SWING_TOP: Pose = { hip: [48, 61], torso: 176, armF: [88, 82], armB: [88, 82], legF: [2, 0], legB: [-2, 0] };

const CARRY_A: Pose = { hip: [48, 61], torso: 178, armF: [7, 3], armB: [7, 3], legF: [16, -8], legB: [-14, 6] };
const CARRY_B: Pose = { hip: [48, 61], torso: 178, armF: [7, 3], armB: [7, 3], legF: [-14, 6], legB: [16, -8] };

const HANG_POSE: Pose = { hip: [50, 58], torso: 177, armF: [176, 180], armB: [176, 180], legF: [3, -8], legB: [-1, -8] };
const HANG_POSE_B: Pose = { ...HANG_POSE, hip: [50, 59] };

const PLANK_A: Pose = { hip: [50, 66], torso: 97, head: 102, armF: [155, 262], armB: [155, 262], legF: [-83, 2], legB: [-83, 2], footF: 12, footB: 12 };
const PLANK_B: Pose = { ...PLANK_A, hip: [50, 65] };
const SIDE_PLANK_A: Pose = { hip: [50, 68], torso: 104, head: 108, armF: [148, 268], armB: [200, 195], legF: [-78, 0], legB: [-78, 0], footF: 20 };
const SIDE_PLANK_B: Pose = { ...SIDE_PLANK_A, hip: [50, 67] };

const CRUNCH_DOWN: Pose = { hip: [46, 74], torso: 96, head: 108, armF: [125, 250], armB: [125, 250], legF: [35, -58], legB: [31, -58] };
const CRUNCH_UP: Pose = { ...CRUNCH_DOWN, torso: 122, head: 138 };

const LEGRAISE_DOWN: Pose = { hip: [46, 76], torso: 94, head: 100, armF: [40, 30], armB: [40, 30], legF: [-80, -4], legB: [-80, -4] };
const LEGRAISE_UP: Pose = { ...LEGRAISE_DOWN, legF: [-6, -2], legB: [-6, -2] };

const HKR_DOWN: Pose = { ...HANG_POSE };
const HKR_UP: Pose = { ...HANG_POSE, legF: [82, -80], legB: [78, -80] };

const TWIST_A: Pose = { hip: [46, 72], torso: 148, armF: [82, 62], armB: [82, 62], legF: [55, -40], legB: [51, -40] };
const TWIST_B: Pose = { ...TWIST_A, armF: [40, 118], armB: [40, 118] };

const ABWHEEL_IN: Pose = { hip: [42, 74], torso: 122, head: 130, armF: [110, 108], armB: [110, 108], legF: [-58, -85], legB: [-58, -85] };
const ABWHEEL_OUT: Pose = { hip: [46, 70], torso: 105, head: 110, armF: [138, 132], armB: [138, 132], legF: [-65, -70], legB: [-65, -70] };

const MTN_A: Pose = { ...PUSHUP_TOP, legF: [45, -95], legB: [-82, 2] };
const MTN_B: Pose = { ...PUSHUP_TOP, legF: [-82, 2], legB: [45, -95] };

const DEADBUG_A: Pose = { hip: [48, 76], torso: 94, head: 98, armF: [178, 182], armB: [95, 90], legF: [-8, -85], legB: [-55, -20] };
const DEADBUG_B: Pose = { hip: [48, 76], torso: 94, head: 98, armF: [95, 90], armB: [178, 182], legF: [-55, -20], legB: [-8, -85] };

const BACKEXT_DOWN: Pose = { hip: [48, 62], torso: 108, armF: [125, 245], armB: [125, 245], legF: [-12, 2], legB: [-14, 2] };
const BACKEXT_UP: Pose = { ...BACKEXT_DOWN, torso: 172 };

const NORDIC_UP: Pose = { hip: [46, 70], torso: 172, armF: [35, 65], armB: [35, 65], legF: [-88, -85], legB: [-88, -85] };
const NORDIC_DOWN: Pose = { hip: [50, 72], torso: 118, armF: [95, 105], armB: [95, 105], legF: [-88, -60], legB: [-88, -60] };

const BURPEE_STAND: Pose = { hip: [48, 61], torso: 178, armF: [8, 4], legF: [2, 0], legB: [-2, 0] };
const BURPEE_SQUAT: Pose = { hip: [44, 78], torso: 140, armF: [65, 40], legF: [80, -25], legB: [76, -25] };
const BURPEE_PLANK: Pose = { ...PUSHUP_BOTTOM };
const BURPEE_JUMP: Pose = { hip: [48, 52], torso: 180, armF: [175, 178], armB: [175, 178], legF: [6, -10], legB: [-6, -2] };

const THRUSTER_BOTTOM: Pose = { ...FRONT_SQUAT_BOTTOM };
const THRUSTER_TOP: Pose = { hip: [48, 61], torso: 180, armF: [172, 182], armB: [172, 182], legF: [2, 0], legB: [-2, 0] };

const RUN_A: Pose = { hip: [48, 59], torso: 168, armF: [55, 140], armB: [-40, 55], legF: [42, -60], legB: [-38, -35], footB: 40 };
const RUN_B: Pose = { hip: [48, 59], torso: 168, armF: [-40, 55], armB: [55, 140], legF: [-38, -35], legB: [42, -60], footF: 40 };
const WALK_A: Pose = { hip: [48, 61], torso: 175, armF: [24, 25], armB: [-18, 10], legF: [22, -10], legB: [-20, 8] };
const WALK_B: Pose = { hip: [48, 61], torso: 175, armF: [-18, 10], armB: [24, 25], legF: [-20, 8], legB: [22, -10] };

const CYCLE_A: Pose = { hip: [42, 60], torso: 140, armF: [72, 55], armB: [72, 55], legF: [65, -55], legB: [15, -30] };
const CYCLE_B: Pose = { hip: [42, 60], torso: 140, armF: [72, 55], armB: [72, 55], legF: [15, -30], legB: [65, -55] };

const ROWING_CATCH: Pose = { hip: [40, 70], torso: 150, armF: [88, 85], armB: [88, 85], legF: [82, -70], legB: [80, -70] };
const ROWING_FINISH: Pose = { hip: [46, 70], torso: 196, armF: [55, -30], armB: [55, -30], legF: [55, -8], legB: [53, -8] };

const JUMPROPE_A: Pose = { hip: [48, 61], torso: 178, armF: [28, 65], armB: [-15, -55], legF: [2, 0], legB: [-2, 0] };
const JUMPROPE_B: Pose = { hip: [48, 58], torso: 178, armF: [32, 75], armB: [-20, -62], legF: [4, -8], legB: [-4, -8], footF: 110, footB: 110 };

const STAIR_A: Pose = { hip: [48, 60], torso: 172, armF: [25, 20], armB: [-18, 8], legF: [48, -60], legB: [-6, 2] };
const STAIR_B: Pose = { hip: [48, 60], torso: 172, armF: [-18, 8], armB: [25, 20], legF: [-6, 2], legB: [48, -60] };

const TEMPLATES: Record<FigureTemplate, Template> = {
  'squat': { frames: [SQUAT_TOP, SQUAT_BOTTOM] },
  'front-squat': { frames: [FRONT_SQUAT_TOP, FRONT_SQUAT_BOTTOM] },
  'goblet-squat': { frames: [GOBLET_TOP, GOBLET_BOTTOM] },
  'hinge': { frames: [HINGE_TOP, HINGE_BOTTOM] },
  'good-morning': { frames: [{ ...SQUAT_TOP }, GM_BOTTOM] },
  'lunge': { frames: [LUNGE_TOP, LUNGE_BOTTOM] },
  'step-up': { frames: [STEP_BOTTOM, STEP_TOP], speed: 800 },
  'bench-press': { frames: [BENCH_DOWN, BENCH_UP], bench: 'flat' },
  'incline-press': { frames: [INCLINE_DOWN, INCLINE_UP], bench: 'incline' },
  'fly': { frames: [FLY_OPEN, FLY_CLOSED], bench: 'flat' },
  'pushup': { frames: [PUSHUP_TOP, PUSHUP_BOTTOM] },
  'dip': { frames: [DIP_TOP, DIP_BOTTOM] },
  'overhead-press': { frames: [OHP_START, OHP_TOP] },
  'lateral-raise': { frames: [LATERAL_DOWN, LATERAL_UP] },
  'front-raise': { frames: [LATERAL_DOWN, FRONT_RAISE_UP] },
  'rear-fly': { frames: [REAR_FLY_DOWN, REAR_FLY_UP] },
  'face-pull': { frames: [FACE_PULL_START, FACE_PULL_END] },
  'upright-row': { frames: [UPRIGHT_START, UPRIGHT_TOP] },
  'pullup': { frames: [PULLUP_BOTTOM, PULLUP_TOP] },
  'pulldown': { frames: [PULLDOWN_START, PULLDOWN_END] },
  'straight-arm-pulldown': { frames: [STRAIGHT_ARM_START, STRAIGHT_ARM_END] },
  'bent-row': { frames: [ROW_DOWN, ROW_UP] },
  'seated-row': { frames: [SEATED_ROW_START, SEATED_ROW_END] },
  'single-arm-row': { frames: [SINGLE_ROW_DOWN, SINGLE_ROW_UP], bench: 'low' },
  'shrug': { frames: [SHRUG_DOWN, SHRUG_UP], speed: 550 },
  'curl': { frames: [CURL_DOWN, CURL_UP] },
  'preacher-curl': { frames: [PREACHER_DOWN, PREACHER_UP] },
  'pushdown': { frames: [PUSHDOWN_START, PUSHDOWN_END] },
  'overhead-triceps': { frames: [OH_TRI_START, OH_TRI_END] },
  'skullcrusher': { frames: [SKULL_DOWN, SKULL_UP], bench: 'flat' },
  'wrist-curl': { frames: [WRIST_DOWN, WRIST_UP], speed: 550 },
  'leg-press': { frames: [LEGPRESS_IN, LEGPRESS_OUT] },
  'leg-extension': { frames: [LEGEXT_DOWN, LEGEXT_UP] },
  'leg-curl-lying': { frames: [LEGCURL_LYING_START, LEGCURL_LYING_END], bench: 'flat' },
  'leg-curl-seated': { frames: [LEGCURL_SEATED_START, LEGCURL_SEATED_END] },
  'calf-raise': { frames: [CALF_DOWN, CALF_UP], speed: 550 },
  'calf-seated': { frames: [CALF_SEATED_DOWN, CALF_SEATED_UP], speed: 550 },
  'hip-thrust': { frames: [HIPTHRUST_DOWN, HIPTHRUST_UP] },
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
  'ab-wheel': { frames: [ABWHEEL_IN, ABWHEEL_OUT] },
  'mountain-climber': { frames: [MTN_A, MTN_B], speed: 380 },
  'dead-bug': { frames: [DEADBUG_A, DEADBUG_B], speed: 900 },
  'back-extension': { frames: [BACKEXT_UP, BACKEXT_DOWN] },
  'nordic-curl': { frames: [NORDIC_UP, NORDIC_DOWN], speed: 900 },
  'burpee': { frames: [BURPEE_STAND, BURPEE_SQUAT, BURPEE_PLANK, BURPEE_SQUAT, BURPEE_JUMP], speed: 450 },
  'thruster': { frames: [THRUSTER_BOTTOM, THRUSTER_TOP] },
  'run': { frames: [RUN_A, RUN_B], speed: 320 },
  'walk': { frames: [WALK_A, WALK_B], speed: 500 },
  'cycle': { frames: [CYCLE_A, CYCLE_B], speed: 400 },
  'rowing': { frames: [ROWING_CATCH, ROWING_FINISH], speed: 650 },
  'jump-rope': { frames: [JUMPROPE_A, JUMPROPE_B], speed: 300 },
  'stair': { frames: [STAIR_A, STAIR_B], speed: 480 },
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
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ExerciseFigure({
  template,
  gear = 'none',
  size = 200,
  tint = palette.forest,
  accent = palette.limeDark,
  paused = false,
}: {
  template: FigureTemplate;
  gear?: FigureGear;
  size?: number;
  tint?: string;
  accent?: string;
  paused?: boolean;
}) {
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
    if (paused) {
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
  }, [template, paused]);

  // Forward kinematics ------------------------------------------------------
  const hip = pose.hip;
  const shoulder = add(hip, pose.torso, L.torso);
  const headBase = add(shoulder, pose.head ?? pose.torso, L.neck);
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

  const gripMid: [number, number] = [(wristF[0] + wristB[0]) / 2, (wristF[1] + wristB[1]) / 2];

  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      {/* stage */}
      <Line x1={8} y1={93.5} x2={92} y2={93.5} stroke={palette.line} strokeWidth={2} strokeLinecap="round" />
      {TEMPLATES[template]?.bench === 'flat' ? (
        <Rect x={26} y={72} width={48} height={7} rx={3} fill={palette.line} />
      ) : null}
      {TEMPLATES[template]?.bench === 'incline' ? (
        <Path d="M30 88 L44 88 L66 60 L60 55 Z" fill={palette.line} />
      ) : null}
      {TEMPLATES[template]?.bench === 'low' ? (
        <Rect x={52} y={78} width={34} height={6} rx={3} fill={palette.line} />
      ) : null}
      {gear === 'bar-overhead' ? (
        <Line x1={16} y1={13} x2={84} y2={13} stroke={accent} strokeWidth={3} strokeLinecap="round" />
      ) : null}
      {gear === 'cable-high' ? (
        <>
          <Rect x={88} y={6} width={7} height={10} rx={2} fill={palette.line} />
          <Line x1={91} y1={12} x2={wristF[0]} y2={wristF[1]} stroke={accent} {...thin} strokeWidth={1.6} />
        </>
      ) : null}
      {gear === 'cable-low' ? (
        <>
          <Rect x={88} y={84} width={7} height={10} rx={2} fill={palette.line} />
          <Line x1={91} y1={88} x2={wristF[0]} y2={wristF[1]} stroke={accent} {...thin} strokeWidth={1.6} />
        </>
      ) : null}

      {/* back limbs (depth) */}
      <G opacity={0.38}>
        <Line x1={shoulder[0]} y1={shoulder[1]} x2={elbowB[0]} y2={elbowB[1]} stroke={tint} {...stroke} />
        <Line x1={elbowB[0]} y1={elbowB[1]} x2={wristB[0]} y2={wristB[1]} stroke={tint} {...stroke} />
        <Line x1={hip[0]} y1={hip[1]} x2={kneeB[0]} y2={kneeB[1]} stroke={tint} {...stroke} />
        <Line x1={kneeB[0]} y1={kneeB[1]} x2={ankleB[0]} y2={ankleB[1]} stroke={tint} {...stroke} />
        <Line x1={ankleB[0]} y1={ankleB[1]} x2={toeB[0]} y2={toeB[1]} stroke={tint} {...stroke} strokeWidth={2.6} />
      </G>

      {/* torso + head */}
      <Line x1={hip[0]} y1={hip[1]} x2={shoulder[0]} y2={shoulder[1]} stroke={tint} {...stroke} strokeWidth={4.2} />
      <Circle cx={headCenter[0]} cy={headCenter[1]} r={L.head} fill="none" stroke={tint} strokeWidth={3} />

      {/* front limbs */}
      <Line x1={hip[0]} y1={hip[1]} x2={kneeF[0]} y2={kneeF[1]} stroke={tint} {...stroke} />
      <Line x1={kneeF[0]} y1={kneeF[1]} x2={ankleF[0]} y2={ankleF[1]} stroke={tint} {...stroke} />
      <Line x1={ankleF[0]} y1={ankleF[1]} x2={toeF[0]} y2={toeF[1]} stroke={tint} {...stroke} strokeWidth={2.6} />
      <Line x1={shoulder[0]} y1={shoulder[1]} x2={elbowF[0]} y2={elbowF[1]} stroke={tint} {...stroke} />
      <Line x1={elbowF[0]} y1={elbowF[1]} x2={wristF[0]} y2={wristF[1]} stroke={tint} {...stroke} />

      {/* gear at hands */}
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
          <Line x1={wristB[0] - 3.4} y1={wristB[1]} x2={wristB[0] + 3.4} y2={wristB[1]} stroke={accent} strokeWidth={4.6} strokeLinecap="round" opacity={0.4} />
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
      {gear === 'machine' ? (
        <Rect x={12} y={86} width={20} height={4} rx={2} fill={palette.line} />
      ) : null}
    </Svg>
  );
}
