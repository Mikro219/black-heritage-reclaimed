from .rules.presence_bilateral import detect as presence_bilateral
from .rules.presence_bilateral_still import detect as presence_bilateral_still
from .rules.directional_point import detect as directional_point
from .rules.directional_head_or_hand import detect as directional_head_or_hand
from .rules.bilateral_sweep import detect as bilateral_sweep
from .rules.bilateral_lower import detect as bilateral_lower
from .rules.shape_match import detect as shape_match
from .rules.rhythm_bilateral import detect as rhythm_bilateral
from .rules.speed_bilateral import detect as speed_bilateral
from .rules.bilateral_alternating import detect as bilateral_alternating
from .rules.bilateral_arcing import detect as bilateral_arcing
from .rules.bilateral_rotation import detect as bilateral_rotation
from .rules.mouth_proximity_tip import detect as mouth_proximity_tip
from .rules.reach_and_close import detect as reach_and_close
from .rules.point_target_held import detect as point_target_held
from .rules.trail_follow import detect as trail_follow
from .rules.directional_draw import detect as directional_draw
from .rules.forward_reach import detect as forward_reach
from .rules.forward_point import detect as forward_point
from .rules.survey import detect as survey
from .rules.touch_head import detect as touch_head
from .rules.arms_crossed import detect as arms_crossed
from .rules.push_out import detect as push_out
from .rules.run_arms import detect as run_arms
from .rules.unravel import detect as unravel
from .rules.paddle import detect as paddle

REGISTRY = {
    "presence_bilateral": presence_bilateral,
    "presence_bilateral_still": presence_bilateral_still,
    "directional_point": directional_point,
    "directional_head_or_hand": directional_head_or_hand,
    "bilateral_sweep": bilateral_sweep,
    "bilateral_lower": bilateral_lower,
    "shape_match": shape_match,
    "rhythm_bilateral": rhythm_bilateral,
    "speed_bilateral": speed_bilateral,
    "bilateral_alternating": bilateral_alternating,
    "bilateral_arcing": bilateral_arcing,
    "bilateral_rotation": bilateral_rotation,
    "mouth_proximity_tip": mouth_proximity_tip,
    "reach_and_close": reach_and_close,
    "point_target_held": point_target_held,
    "trail_follow": trail_follow,
    "directional_draw": directional_draw,
    "forward_reach": forward_reach,
    "forward_point": forward_point,
    "survey": survey,
    "touch_head": touch_head,
    "arms_crossed": arms_crossed,
    "push_out": push_out,
    "run_arms": run_arms,
    "unravel": unravel,
    "paddle": paddle,
}
