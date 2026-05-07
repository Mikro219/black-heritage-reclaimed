"""
bilateral_rotation — both hands making outward circular motions, ≥ min_cycles.
Used for Scene 11 unravel CG (≥2 outward rotations).

Params:
  min_cycles (int): complete outward rotations per hand. Default 2.
  direction (str): "outward" | "inward". Default "outward".

Approach: track angle of wrist→middle-finger vector per hand across frames;
accumulate full 360° sweeps; both hands must complete within a short window.

Context keys: rotation_cycle_count, rotation_angles
"""


def detect(landmarks, params: dict, context: dict) -> bool:
    # TODO: compute per-hand rotation angle from landmark vectors,
    # accumulate full outward cycles, gate on min_cycles for both hands.
    if "rotation_cycle_count" not in context:
        context["rotation_cycle_count"] = {0: 0, 1: 0}
        context["rotation_angles"] = {0: None, 1: None}
    return False
