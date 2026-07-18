/* BHR detector registry for the Experience Builder.
 *
 * HAND-SYNCED from engines/detectors/__init__.py REGISTRY — if a detector is
 * added/renamed there, mirror it here. `params` are the author-facing subset
 * with the same defaults as the rule modules' docstrings; the exporter passes
 * them through verbatim, so any param a rule reads may be added here.
 *
 * region: "required" — the window's drawn region is the detection target
 *         "optional" — region is used for preview clicks / on-screen indicator only
 */
window.EB = window.EB || {};

EB.DETECTORS = [
  { type: "point_region", label: "Point at region (L/R)", icon: "point_scan", region: "required",
    hint: "Arm raised toward the left or right half (pose wrist vs body midline); low side-reach accepted for low targets.",
    params: { hold_ms: 600, reject_open_palm: true, allow_low_reach: true } },

  { type: "point_target_held", label: "Point at target + hold", icon: "ads_click", region: "required",
    hint: "Extended arm with the hand point held inside the drawn region (pose-based).",
    params: { hold_ms: 700 } },

  { type: "forward_point", label: "Point into screen", icon: "touch_app", region: "required",
    hint: "Hand pushed toward the camera while over the region.",
    params: { hold_ms: 500 } },

  { type: "forward_reach", label: "Reach toward screen", icon: "front_hand", region: "optional",
    hint: "Wrist approaches the camera (real depth on the Gemini 335, pose-z fallback). Any hand pose counts.",
    params: { area_growth_threshold: 0.30, min_depth_delta_mm: 180 } },

  { type: "reach_and_close", label: "Reach + grab (fist)", icon: "back_hand", region: "optional",
    hint: "Hand reaches toward the screen then settles — the take-hold gesture (pose-based).",
    params: {} },

  { type: "presence_bilateral", label: "Raise both hands", icon: "waving_hand", region: "optional",
    hint: "Both hands above shoulder height, held briefly.",
    params: { y_threshold: "shoulder", hold_ms: 500 } },

  { type: "presence_bilateral_still", label: "Hold still, palms out", icon: "do_not_touch", region: "optional",
    hint: "Both hands present and nearly motionless, palms forward.",
    params: { hold_ms: 1000 } },

  { type: "directional_point", label: "Directional point", icon: "arrow_selector_tool", region: "optional",
    hint: "One extended arm pointing left / right / up / down (wrist→index vector).",
    params: { direction: "left", hold_ms: 500 } },

  { type: "directional_draw", label: "Draw stroke (direction)", icon: "gesture", region: "optional",
    hint: "One directional stroke — tracks BOTH pose wrists; ~30° tolerance. Chain several windows for multi-stroke traces.",
    params: { direction: "left", tolerance_deg: 30 } },

  { type: "rhythm_bilateral", label: "Knock (two pushes)", icon: "door_front", region: "optional",
    hint: "Two pushes toward the camera — real depth drop on the Gemini 335, pose-z fallback.",
    params: { knock_count: 2, knock_window_ms: 2500, min_growth_pct: 30 } },

  { type: "push_out", label: "Push forward (both hands)", icon: "open_with", region: "optional",
    hint: "Both wrists thrust toward the screen together (depth / pose-z).",
    params: { min_growth_pct: 50, window_ms: 250, min_depth_delta_mm: 200 } },

  { type: "throw", label: "Throw (overhand)", icon: "sports_handball", region: "optional",
    hint: "Wind up above the shoulder, then snap below it within the stroke window.",
    params: { max_stroke_ms: 600 } },

  { type: "survey", label: "Salute / shade eyes", icon: "visibility", region: "optional",
    hint: "Hand held at brow height (salute) for the hold time.",
    params: { salute_hold_ms: 500 } },

  { type: "touch_head", label: "Touch head / temple", icon: "psychology", region: "optional",
    hint: "Hand raised to the temple region.",
    params: { hold_ms: 500 } },

  { type: "arms_crossed", label: "Cross arms", icon: "close_fullscreen", region: "optional",
    hint: "Both wrists cross past the body centerline.",
    params: { hold_ms: 500 } },

  { type: "bilateral_lower", label: "Lower both hands", icon: "south", region: "optional",
    hint: "Both hands moving downward together (duck).",
    params: {} },

  { type: "bilateral_sweep", label: "Arm sweep", icon: "swipe", region: "optional",
    hint: "Extended arm sweeps across the screen at speed.",
    params: {} },

  { type: "bilateral_alternating", label: "Walk / march arms", icon: "directions_walk", region: "optional",
    hint: "Alternating forward arm motion, N cycles; cadence separates walk from run.",
    params: { min_cycles: 3 } },

  { type: "run_arms", label: "Run arms (fast pump)", icon: "directions_run", region: "optional",
    hint: "Fast alternating arm pump crossing the waist line.",
    params: {} },

  { type: "speed_bilateral", label: "Fast gather / shake", icon: "bolt", region: "optional",
    hint: "Hands moving above a velocity threshold in sustained bursts (screen units/second; metric mm/s with depth).",
    params: { velocity_multiplier: 2.0, min_active_ms: 250 } },

  { type: "bilateral_arcing", label: "Paddle strokes", icon: "rowing", region: "optional",
    hint: "Both wrists arc high-forward to low-back, N strokes.",
    params: { stroke_count: 4 } },

  { type: "paddle", label: "Paddle (hysteresis)", icon: "kayaking", region: "optional",
    hint: "Bilateral paddling with midline hysteresis so jitter can't rack up strokes.",
    params: { stroke_count: 4 } },

  { type: "bilateral_rotation", label: "Unravel (rotate out)", icon: "cached", region: "optional",
    hint: "Both wrists trace outward circles, N cycles.",
    params: { min_cycles: 2 } },

  { type: "unravel", label: "Unwind rope", icon: "rotate_right", region: "optional",
    hint: "Both hands rotating outward — the boat-unwinding gesture.",
    params: { min_cycles: 2 } },

  { type: "spread_arms", label: "Spread arms wide", icon: "expand_content", region: "optional",
    hint: "Both wrists outside the shoulders in a chest-height band (wingspan pose).",
    params: { hold_ms: 500 } },

  { type: "mouth_proximity_tip", label: "Drink (hand to mouth)", icon: "local_drink", region: "optional",
    hint: "Hand raised to the mouth with the wrist at/below the hand point — the natural drinking pose.",
    params: {} },

  { type: "directional_head_or_hand", label: "Look / scan direction", icon: "frame_person", region: "optional",
    hint: "Head turn OR hand shading eyes toward a direction.",
    params: { direction: "left" } },

  { type: "trail_follow", label: "Follow trail waypoints", icon: "route", region: "optional",
    hint: "Extended hand follows defined waypoints (≥60% match).",
    params: { waypoint_match: 0.60 } },

  { type: "voice", label: "Voice keyword", icon: "mic", region: "optional", voice: true,
    hint: "Vosk keyword spotting — fires when the player says the keyword during the window.",
    params: { keyword: "go" } },
];

EB.detectorByType = function (type) {
  return EB.DETECTORS.find(d => d.type === type) || null;
};
