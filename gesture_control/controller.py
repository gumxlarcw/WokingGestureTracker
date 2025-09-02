from typing import Optional, Tuple
import cv2, time, numpy as np, pyautogui as pag

from config import (CAM_RES, MIRROR_VIEW, CURSOR_FREQ_INIT, CURSOR_MINCUTOFF, CURSOR_BETA, CURSOR_DCUTOFF)
from filters import OneEuroFilter
from utils import map_to_screen
from tracker import HandTracker
from gestures import GestureState, pick_stable_hand_idx, detect, thumbs_up_ready

class GestureController:
    def __init__(self):
        pag.FAILSAFE = True
        self.scrW, self.scrH = pag.size()
        self.state = GestureState()
        self.filter = OneEuroFilter(freq=CURSOR_FREQ_INIT,
                                    mincutoff=CURSOR_MINCUTOFF,
                                    beta=CURSOR_BETA,
                                    dcutoff=CURSOR_DCUTOFF)
        self.cursor_mode = True
        self._dt: float = 1/60.0   # <<< tambahkan baris ini


    def handle_actions(self, info):
        now = time.time()

        # kursor & drag
        if self.cursor_mode and info["idx_tip_xy"] is not None:
            self.filter.freq = 1.0 / max(1e-6, self._dt)
            smoothed = self.filter(info["idx_tip_xy"], now)
            mx, my = map_to_screen(float(smoothed[0]), float(smoothed[1]), self.scrW, self.scrH)
            pag.moveTo(mx, my, duration=0)

            if info["pinch"] and not self.state.dragging:
                pag.mouseDown(); self.state.dragging = True
            elif not info["pinch"] and self.state.dragging:
                pag.mouseUp(); self.state.dragging = False

        # thumbs up = space (debounce)
        if info["thumbs_up"] and thumbs_up_ready(self.state):
            pag.press('space')
            self.state.last_space_ts = now

        # swipe
        if info["swipe"] == "right":
            pag.hotkey('ctrl', 'right')
        elif info["swipe"] == "left":
            pag.hotkey('ctrl', 'left')

    def toggle_cursor(self):
        self.cursor_mode = not self.cursor_mode
        if self.state.dragging:
            pag.mouseUp(); self.state.dragging = False

def run_loop():
    tracker = HandTracker()
    ctrl = GestureController()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES[1])
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    print("Gesture Control (smooth) — Q: quit, T: toggle cursor")

    prev_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame, multi = tracker.process(frame)
            tracker.draw(frame, multi)

            lm = None
            if multi:
                idx = pick_stable_hand_idx(multi)
                if idx is not None:
                    lm = multi[idx].landmark

            info = detect(lm, ctrl.state)

            now = time.time()
            ctrl._dt = max(now - prev_t, 1e-6)
            prev_t = now

            ctrl.handle_actions(info)

            cv2.putText(frame, f"Cursor: {'ON' if ctrl.cursor_mode else 'OFF'}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if ctrl.cursor_mode else (0,0,255), 2)
            cv2.imshow("Gesture-Based Interaction (Smooth)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('t'), ord('T')):
                ctrl.toggle_cursor()
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
