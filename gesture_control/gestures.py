from collections import deque
from typing import Optional, Deque, Dict, Any, Tuple
import numpy as np
import time

from utils import norm_dist_xy
from config import PINCH_ON, PINCH_OFF, VOTE_WINDOW, VOTE_ON, THUMBS_COOLDOWN, SWIPE_VX, VELOCITY_BUF

class GestureState:
    def __init__(self):
        self.pinch_hist: Deque[bool] = deque(maxlen=VOTE_WINDOW)
        self.vx_hist: Deque[float] = deque(maxlen=VELOCITY_BUF)
        self.dragging = False
        self.last_space_ts = 0.0
        self.prev_idx: Optional[np.ndarray] = None

def pick_stable_hand_idx(multi) -> Optional[int]:
    if not multi: return None
    if len(multi) == 1: return 0
    vars_sum = []
    for hand in multi:
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        vars_sum.append(float(np.var(xs) + np.var(ys)))
    return int(np.argmin(vars_sum))

def detect(lm, state: GestureState) -> Dict[str, Any]:
    """Kembalikan dict status gesture saat ini."""
    info = {
        "idx_tip_xy": None,      # np.ndarray [x,y] (0..1)
        "pinch": False,
        "thumbs_up": False,
        "two_fingers": False,
        "swipe": None            # "left" | "right" | None
    }
    if lm is None:
        state.prev_idx = None
        state.vx_hist.clear()
        return info

    idx_tip = lm[8]; th_tip = lm[4]

    # pinch hysteresis + voting
    d = norm_dist_xy((th_tip.x, th_tip.y), (idx_tip.x, idx_tip.y))
    if not state.dragging:
        state.pinch_hist.append(d < PINCH_ON)
    else:
        state.pinch_hist.append(d < PINCH_OFF)
    pinch_now = (sum(state.pinch_hist) >= VOTE_ON)
    info["pinch"] = pinch_now

    # fingers up heuristic
    fingers_up = [
        False,
        lm[8].y  < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y
    ]
    thumb_up = (lm[4].y < lm[3].y) and (abs(lm[4].x - lm[2].x) > 0.06)
    info["thumbs_up"] = thumb_up and not any(fingers_up[1:])
    info["two_fingers"] = fingers_up[1] and fingers_up[2] and not any([fingers_up[3], fingers_up[4]])

    # index tip pos
    idx_tip_xy = np.array([float(idx_tip.x), float(idx_tip.y)], dtype=np.float32)
    info["idx_tip_xy"] = idx_tip_xy

    # kecepatan untuk swipe (pakai x)
    if state.prev_idx is not None:
        state.vx_hist.append(float(idx_tip_xy[0] - state.prev_idx[0]))
    state.prev_idx = idx_tip_xy.copy()

    if info["two_fingers"] and len(state.vx_hist) >= VELOCITY_BUF:
        mean_vx = float(np.mean(state.vx_hist))
        if mean_vx > SWIPE_VX:
            info["swipe"] = "right"; state.vx_hist.clear()
        elif mean_vx < -SWIPE_VX:
            info["swipe"] = "left"; state.vx_hist.clear()

    return info

def thumbs_up_ready(state: GestureState) -> bool:
    return (time.time() - state.last_space_ts) > THUMBS_COOLDOWN
