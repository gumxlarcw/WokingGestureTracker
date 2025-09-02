# --- silence TF/absl logs ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.FATAL)

import cv2, time, csv
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

from mediapipe.python.solutions import face_detection as mp_face
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import drawing_styles as mp_styles

# ====== config ======
MIRROR_VIEW = False          # set True kalau mau tampilan mirror
AUTO_CALIB_DURATION = 6.0    # detik pengambilan sampel untuk kalibrasi
ROI_MARGIN = 40              # px margin di sekitar klaster saat buat ROI

# ---------- util ----------
def landmarks_bbox(landmarks, w:int, h:int) -> Tuple[int,int,int,int]:
    xs=[lm.x for lm in landmarks.landmark]; ys=[lm.y for lm in landmarks.landmark]
    x0,x1=max(min(xs),0.0),min(max(xs),1.0); y0,y1=max(min(ys),0.0),min(max(ys),1.0)
    return int(x0*w), int(y0*h), int((x1-x0)*w), int((y1-y0)*h)

def hand_center_px(hand_lms, w:int, h:int) -> Tuple[int,int]:
    xs=[lm.x for lm in hand_lms.landmark]; ys=[lm.y for lm in hand_lms.landmark]
    cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
    return int(cx*w), int(cy*h)

def point_in_rect(px:int, py:int, rect: Optional[Tuple[int,int,int,int]]) -> bool:
    if rect is None: return False
    x1,y1,x2,y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

def format_hms(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def select_roi(win: str, frame) -> Optional[Tuple[int,int,int,int]]:
    r = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("ROI selector")
    if r is None or r == (0,0,0,0): return None
    x,y,w,h = r
    return (x, y, x+w, y+h)

def bbox_from_points(pts: np.ndarray, w:int, h:int, margin:int=0) -> Tuple[int,int,int,int]:
    # pts shape: (N,2)
    x1 = max(int(np.min(pts[:,0]) - margin), 0)
    y1 = max(int(np.min(pts[:,1]) - margin), 0)
    x2 = min(int(np.max(pts[:,0]) + margin), w-1)
    y2 = min(int(np.max(pts[:,1]) + margin), h-1)
    return (x1, y1, x2, y2)

# ---------- camera open ----------
def open_camera():
    tries = [
        (0, cv2.CAP_DSHOW), (1, cv2.CAP_DSHOW), (2, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),  (1, cv2.CAP_MSMF),  (2, cv2.CAP_MSMF),
        (0, 0), (1, 0), (2, 0)
    ]
    for idx, backend in tries:
        cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[OK] Kamera terbuka pada index {idx} backend {backend}")
                return cap
        cap.release()
    return None

# ---------- auto calibration ----------
def auto_calibrate(cap, hands, w:int, h:int) -> Tuple[Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]]]:
    """
    Rekam titik pusat tangan selama AUTO_CALIB_DURATION detik,
    klasterkan (k=2), klaster dengan centroid-x terbesar -> Mouse,
    lainnya -> Keyboard. Kembalikan ROI keyboard, ROI mouse.
    """
    print("[CAL] Mulai kalibrasi otomatis... Gerakkan tangan di area keyboard dan mouse seperti biasa.")
    t0 = time.time()
    samples = []

    while time.time() - t0 < AUTO_CALIB_DURATION:
        ok, frame = cap.read()
        if not ok: break
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        mh = getattr(res, "multi_hand_landmarks", None)
        if mh:
            for hand_lms in mh:
                cx, cy = hand_center_px(hand_lms, w, h)
                samples.append([cx, cy])
                cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)
        cv2.putText(frame, "Calibrating... move hands over keyboard & mouse", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("Work Tracker - Mouse/Keyboard", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    if len(samples) < 10:
        print("[CAL] Sampel terlalu sedikit. Gagal kalibrasi.")
        return None, None

    pts = np.array(samples, dtype=np.float32)

    # K-Means k=2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    compactness, labels, centers = cv2.kmeans(pts, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.reshape(2,2)
    # klaster dengan x terbesar → Mouse
    mouse_idx = int(np.argmax(centers[:,0]))
    keyb_idx  = 1 - mouse_idx

    mouse_pts = pts[labels.ravel() == mouse_idx]
    keyb_pts  = pts[labels.ravel() == keyb_idx]

    roi_mouse = bbox_from_points(mouse_pts, w, h, margin=ROI_MARGIN)
    roi_keyb  = bbox_from_points(keyb_pts,  w, h, margin=ROI_MARGIN)

    print(f"[CAL] ROI Mouse: {roi_mouse} | ROI Keyboard: {roi_keyb}")
    return roi_keyb, roi_mouse

def main():
    print("Start... (C=Auto-ROI, K=ROI Keyboard, M=ROI Mouse, R=Reset, S=Save CSV, Q=Quit)")
    cap = open_camera()
    if cap is None:
        print("[ERR] Tidak bisa membuka kamera."); time.sleep(2); return

    pose_connections: List[Tuple[int,int]]  = list(mp_pose.POSE_CONNECTIONS)
    hand_connections: List[Tuple[int,int]]  = list(mp_hands.HAND_CONNECTIONS)

    roi_keyboard: Optional[Tuple[int,int,int,int]] = None
    roi_mouse:    Optional[Tuple[int,int,int,int]] = None

    total_work_sec = 0.0
    last_ts = time.time()
    working = False
    session_start = datetime.now()

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_det, \
         mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Gagal baca frame."); break
            if MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # FACE (opsional tampil)
            face_res = face_det.process(rgb)
            detections = getattr(face_res, "detections", None)
            if detections:
                for det in detections:
                    mp_draw.draw_detection(frame, det)

            # POSE (opsional tampil)
            pose_res = pose.process(rgb)
            pose_landmarks = getattr(pose_res, "pose_landmarks", None)
            if pose_landmarks is not None:
                mp_draw.draw_landmarks(frame, pose_landmarks, pose_connections,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

            # HANDS
            hands_res   = hands.process(rgb)
            multi_hands = getattr(hands_res, "multi_hand_landmarks", None)

            # Gambar ROI
            if roi_keyboard:
                x1,y1,x2,y2 = roi_keyboard
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
                cv2.putText(frame, "Keyboard ROI", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            if roi_mouse:
                x1,y1,x2,y2 = roi_mouse
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "Mouse ROI", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Deteksi apakah ada tangan di ROI
            in_keyboard = False
            in_mouse = False
            if multi_hands:
                for hand_lms in multi_hands:
                    mp_draw.draw_landmarks(
                        frame, hand_lms, hand_connections,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style()
                    )
                    cx, cy = hand_center_px(hand_lms, w, h)
                    cv2.circle(frame, (cx, cy), 5, (0,255,255), -1)
                    if point_in_rect(cx, cy, roi_keyboard):
                        in_keyboard = True
                    if point_in_rect(cx, cy, roi_mouse):
                        in_mouse = True

            # Update timer
            now = time.time()
            dt = now - last_ts
            last_ts = now

            new_working = (in_mouse or in_keyboard)
            if working and new_working:
                total_work_sec += dt
            # perubahan state lain tidak menambah waktu
            working = new_working

            # Overlay info
            status = "WORKING" if working else "IDLE"
            cv2.putText(frame, f"Status: {status}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if working else (0,0,255), 2)
            cv2.putText(frame, f"Work Time: {format_hms(total_work_sec)}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            hint = "C: auto-calibrate | K/M: set ROI | R: reset | S: save CSV | Q: quit"
            cv2.putText(frame, hint, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            cv2.imshow("Work Tracker - Mouse/Keyboard", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')):
                total_work_sec = 0.0
            elif key in (ord('s'), ord('S')):
                with open("work_log.csv", "a", newline="", encoding="utf-8") as f:
                    wtr = csv.writer(f)
                    wtr.writerow([session_start.isoformat(timespec="seconds"),
                                  datetime.now().isoformat(timespec="seconds"),
                                  format_hms(total_work_sec)])
                print("[LOG] Disimpan ke work_log.csv")
            elif key in (ord('k'), ord('K')):
                roi_keyboard = select_roi("ROI selector", frame)
            elif key in (ord('m'), ord('M')):
                roi_mouse = select_roi("ROI selector", frame)
            elif key in (ord('c'), ord('C')):
                # auto calibration
                roi_keyboard, roi_mouse = auto_calibrate(cap, hands, w, h)

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")

if __name__ == "__main__":
    main()
