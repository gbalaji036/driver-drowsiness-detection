import os
import urllib.request
import time
import csv
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial import distance as dist
import pygame

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. DOWNLOAD MODEL FILE AUTOMATICALLY ---
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face_landmarker model (~29MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model downloaded successfully.")

# --- 2. AUDIO ALARM SETUP ---
pygame.mixer.init()

def trigger_beep():
    sample_rate = 44100
    duration = 0.25
    freq = 880
    n_samples = int(round(duration * sample_rate))
    buf = np.sin(2 * np.pi * np.arange(n_samples) * freq / sample_rate)
    sound_array = np.int16(buf * 32767)
    stereo_array = np.repeat(sound_array.reshape(-1, 1), 2, axis=1)
    sound = pygame.sndarray.make_sound(stereo_array)
    sound.play()

# --- 3. CSV EVENT LOGGER ---
CSV_FILE = "fatigue_log.csv"
with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Event_Type", "EAR", "MAR", "Pitch_Deg"])

def log_event(event_type, ear, mar, pitch):
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type,
            f"{ear:.2f}",
            f"{mar:.2f}",
            f"{pitch:.1f}"
        ])

# --- 4. FACIAL LANDMARK INDICES & 3D MODEL ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]
POSE_POINTS = [1, 152, 33, 263, 61, 291]

MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Mouth left
    (150.0, -150.0, -125.0)   # Mouth right
], dtype=np.float64)

def calculate_ear(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth_pts):
    A = dist.euclidean(mouth_pts[1], mouth_pts[7])
    B = dist.euclidean(mouth_pts[2], mouth_pts[6])
    C = dist.euclidean(mouth_pts[3], mouth_pts[5])
    D = dist.euclidean(mouth_pts[0], mouth_pts[4])
    return (A + B + C) / (2.0 * D)

def get_head_pitch(shape_2d, img_w, img_h):
    focal_length = img_w
    cam_center = (img_w / 2, img_h / 2)
    cam_matrix = np.array([
        [focal_length, 0, cam_center[0]],
        [0, focal_length, cam_center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        MODEL_3D, shape_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360

# --- 5. INITIALIZE MEDIAPIPE FACE LANDMARKER (NEW API) ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- 6. CONFIGURATION & STATE ---
EAR_THRESH = 0.22
EAR_CONSEC_FRAMES = 45   # ~1.5s at 30 FPS
MAR_THRESH = 0.65
MAR_CONSEC_FRAMES = 40   # ~1.3s
PITCH_NOD_THRESH = -15.0

ear_counter = 0
mar_counter = 0
last_beep_time = 0
last_log_time = 0

cap = cv2.VideoCapture(0)
print(f"[INFO] Camera started. Logging alerts to {CSV_FILE}. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    status = "ALERT & ATTENTIVE"
    status_color = (0, 255, 0)
    ear, mar, pitch = 0.0, 0.0, 0.0

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        coords = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])

        # Calculate metrics
        left_ear = calculate_ear(coords[LEFT_EYE])
        right_ear = calculate_ear(coords[RIGHT_EYE])
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(coords[MOUTH])

        pose_2d = np.array([coords[idx] for idx in POSE_POINTS], dtype=np.float64)
        pitch = get_head_pitch(pose_2d, w, h)

        is_drowsy = False
        event_name = None

        if ear < EAR_THRESH:
            ear_counter += 1
            if ear_counter >= EAR_CONSEC_FRAMES:
                is_drowsy = True
                event_name = "EYES_CLOSED"
                status = "CRITICAL: EYES CLOSED!"
        else:
            ear_counter = 0

        if mar > MAR_THRESH:
            mar_counter += 1
            if mar_counter >= MAR_CONSEC_FRAMES:
                event_name = "YAWN"
                status = "WARNING: YAWNING DETECTED"
                status_color = (0, 165, 255)
        else:
            mar_counter = 0

        if pitch < PITCH_NOD_THRESH and ear_counter > (EAR_CONSEC_FRAMES // 2):
            is_drowsy = True
            event_name = "HEAD_NOD"
            status = "CRITICAL: HEAD NODDING / SLEEP!"

        if is_drowsy:
            status_color = (0, 0, 255)
            curr = time.time()
            if curr - last_beep_time > 0.4:
                trigger_beep()
                last_beep_time = curr

        if event_name and (time.time() - last_log_time > 2.0):
            log_event(event_name, ear, mar, pitch)
            last_log_time = time.time()

        for pt in np.concatenate((coords[LEFT_EYE], coords[RIGHT_EYE], coords[MOUTH])):
            cv2.circle(frame, tuple(pt), 1, (0, 255, 0), -1)

    # HUD display
    cv2.rectangle(frame, (10, 10), (460, 140), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(frame, f"EAR: {ear:.2f} (Alarm < {EAR_THRESH})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR: {mar:.2f} (Alarm > {MAR_THRESH})", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Driver Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()