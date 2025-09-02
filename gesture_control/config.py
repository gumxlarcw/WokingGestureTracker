# --- silence TF/absl logs (letakkan paling atas) ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# UI / kamera
MIRROR_VIEW = True
CAM_RES = (640, 480)

# OneEuro params
CURSOR_FREQ_INIT = 30.0
CURSOR_MINCUTOFF = 1.2
CURSOR_BETA      = 0.03
CURSOR_DCUTOFF   = 1.0

# Pinch hysteresis + voting
PINCH_ON   = 0.040
PINCH_OFF  = 0.055
VOTE_WINDOW = 7
VOTE_ON     = 5

# Gestures debounce & swipe
THUMBS_COOLDOWN = 0.6
SWIPE_VX = 0.028
VELOCITY_BUF = 6

# Mediapipe thresholds
MP_MAX_HANDS = 2
MP_MODEL_COMPLEXITY = 1
MP_MIN_DET_CONF = 0.75
MP_MIN_TRK_CONF = 0.75
