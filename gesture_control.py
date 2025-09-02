# --- silence TF/absl logs (letakkan paling atas) ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.FATAL)

import cv2, time, math, numpy as np, pyautogui as pag
from collections import deque
from typing import Optional, Tuple, List, Deque, cast

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import drawing_styles as mp_styles

# ================= One Euro Filter =================
class OneEuroFilter:
    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.freq: float = freq
        self.mincutoff: float = mincutoff
        self.beta: float = beta
        self.dcutoff: float = dcutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x.astype(np.float32).copy()
            self.dx_prev = np.zeros_like(self.x_prev)
            return self.x_prev.copy()

        dt: float = max(t - cast(float, self.t_prev), 1e-6)
        self.t_prev = t

        # pastikan bukan None untuk Pylance
        assert self.x_prev is not None
        assert self.dx_prev is not None

        # derivative estimate
        dx = (x - self.x_prev) / dt
        ad = self._alpha(self.dcutoff, dt)
        dx_hat = ad * dx + (1.0 - ad) * self.dx_prev

        # dynamic cutoff
        speed = float(np.linalg.norm(dx_hat))   # -> float murni
        cutoff = self.mincutoff + self.beta * speed
        a = self._alpha(float(cutoff), dt)

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# ================= Config =================
MIRROR_VIEW = True
CAM_RES = (640, 480)

# OneEuro params:
CURSOR_FREQ_INIT = 30.0
CURSOR_MINCUTOFF = 1.2
CURSOR_BETA      = 0.03
CURSOR_DCUTOFF   = 1.0

# Pinch hysteresis + voting
PINCH_ON   = 0.040
PINCH_OFF  = 0.055
VOTE_WINDOW = 7
VOTE_ON     = 5

# Gestures debounce
THUMBS_COOLDOWN = 0.6
SWIPE_VX = 0.028
VELOCITY_BUF = 6

# ================= Util =================
def norm_dist(a, b) -> float:
    dx, dy = (a.x - b.x), (a.y - b.y)
    return float((dx*dx + dy*dy) ** 0.5)

def map_to_screen(nx: float, ny: float, W: int, H: int) -> Tuple[int,int]:
    x = int(np.clip(nx, 0.0, 1.0) * W)
    y = int(np.clip(ny, 0.0, 1.0) * H)
    return x, y

def pick_stable_hand(multi_landmarks) -> Optional[int]:
    if not multi_landmarks: return None
    if len(multi_landmarks) == 1: return 0
    vars_sum: List[float] = []
    for hand in multi_landmarks:
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        vars_sum.append(float(np.var(xs) + np.var(ys)))
    return int(np.argmin(vars_sum))

# ================= Main =================
def main():
    pag.FAILSAFE = True
    scrW, scrH = pag.size()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES[1])
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    # smoothing filters
    of = OneEuroFilter(freq=CURSOR_FREQ_INIT,
                       mincutoff=CURSOR_MINCUTOFF,
                       beta=CURSOR_BETA,
                       dcutoff=CURSOR_DCUTOFF)

    pinch_hist: Deque[bool] = deque(maxlen=VOTE_WINDOW)
    vx_hist: Deque[float]   = deque(maxlen=VELOCITY_BUF)

    prev_idx: Optional[np.ndarray] = None
    prev_t = time.time()
    dragging = False
    cursor_mode = True
    last_space_ts = 0.0

    # HAND_CONNECTIONS: frozenset -> list untuk memuaskan Pylance
    hand_connections: List[Tuple[int,int]] = list(mp_hands.HAND_CONNECTIONS)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=1,
                        min_detection_confidence=0.75,
                        min_tracking_confidence=0.75) as hands:

        print("Gesture Control (smooth) — Q: quit, T: toggle cursor")
        while True:
            ok, frame = cap.read()
            if not ok: break
            if MIRROR_VIEW: frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            multi = getattr(res, "multi_hand_landmarks", None)
            if multi:
                for hand in multi:
                    mp_draw.draw_landmarks(
                        frame, hand, hand_connections,  # <— list, bukan frozenset
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            idx_tip_xy: Optional[np.ndarray] = None
            pinch_now_bool = False
            thumbs_up_now = False
            two_fingers = False

            if multi:
                i = pick_stable_hand(multi)
                if i is not None:
                    hand = multi[i]
                    lm = hand.landmark
                    idx_tip = lm[8]
                    th_tip  = lm[4]

                    # pinch hysteresis + voting
                    d = norm_dist(th_tip, idx_tip)
                    if not dragging:
                        pinch_hist.append(d < PINCH_ON)
                    else:
                        pinch_hist.append(d < PINCH_OFF)
                    pinch_now_bool = (sum(pinch_hist) >= VOTE_ON)

                    # thumbs up & two fingers (simple)
                    fingers_up = [
                        False,
                        lm[8].y  < lm[6].y,
                        lm[12].y < lm[10].y,
                        lm[16].y < lm[14].y,
                        lm[20].y < lm[18].y
                    ]
                    thumb_up = (lm[4].y < lm[3].y) and (abs(lm[4].x - lm[2].x) > 0.06)
                    thumbs_up_now = thumb_up and not any(fingers_up[1:])
                    two_fingers = fingers_up[1] and fingers_up[2] and not any([fingers_up[3], fingers_up[4]])

                    idx_tip_xy = np.array([float(idx_tip.x), float(idx_tip.y)], dtype=np.float32)

            # ===== Update filter freq & smooth cursor =====
            now = time.time()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            of.freq = 1.0 / dt

            if cursor_mode and idx_tip_xy is not None:
                x_hat_y_hat = of(idx_tip_xy, now)              # np.ndarray
                mx, my = map_to_screen(float(x_hat_y_hat[0]),  # cast -> float
                                       float(x_hat_y_hat[1]),
                                       scrW, scrH)
                pag.moveTo(mx, my, duration=0)

                if pinch_now_bool and not dragging:
                    pag.mouseDown(); dragging = True
                elif not pinch_now_bool and dragging:
                    pag.mouseUp(); dragging = False

            # ===== Media gestures =====
            if thumbs_up_now and (now - last_space_ts) > THUMBS_COOLDOWN:
                pag.press('space'); last_space_ts = now

            # ===== Swipe detection =====
            if idx_tip_xy is not None:
                if prev_idx is not None:
                    vx_hist.append(float(idx_tip_xy[0] - prev_idx[0]))
                prev_idx = idx_tip_xy.copy()

            if two_fingers and len(vx_hist) >= VELOCITY_BUF:
                mean_vx = float(np.mean(vx_hist))
                if mean_vx > SWIPE_VX:
                    pag.hotkey('ctrl', 'right'); vx_hist.clear()
                elif mean_vx < -SWIPE_VX:
                    pag.hotkey('ctrl', 'left');  vx_hist.clear()

            # ===== overlay =====
            cv2.putText(frame, f"Cursor: {'ON' if cursor_mode else 'OFF'}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if cursor_mode else (0,0,255), 2)
            cv2.imshow("Gesture-Based Interaction (Smooth)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('t'), ord('T')):
                cursor_mode = not cursor_mode
                if dragging:
                    pag.mouseUp(); dragging = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
