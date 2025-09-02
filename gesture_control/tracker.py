from typing import Optional, List, Tuple
import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import drawing_styles as mp_styles

from config import (MIRROR_VIEW, MP_MAX_HANDS, MP_MODEL_COMPLEXITY, MP_MIN_DET_CONF, MP_MIN_TRK_CONF)

class HandTracker:
    def __init__(self):
        self._mp_hands = mp_hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_HANDS,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DET_CONF,
            min_tracking_confidence=MP_MIN_TRK_CONF
        )
        self._connections: List[Tuple[int,int]] = list(mp_hands.HAND_CONNECTIONS)

    def process(self, frame_bgr):
        if MIRROR_VIEW:
            frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        multi = getattr(res, "multi_hand_landmarks", None)
        return frame_bgr, multi

    def draw(self, frame_bgr, hand_landmarks):
        for hand in hand_landmarks or []:
            mp_draw.draw_landmarks(
                frame_bgr, hand, self._connections,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    def close(self):
        self._hands.close()
