# Real-Time Driver Drowsiness & Fatigue Detection System 🚗😴

An end-to-end edge computer vision application that monitors driver attentiveness in real-time. Built with Python, OpenCV, and MediaPipe's Face Landmarker Tasks API, this system runs locally at 30+ FPS on standard CPU hardware to detect microsleeps, prolonged eye closure, yawning, and head nodding.

---

## 🌟 Key Features

- **Eye Closure Monitoring (EAR):** Differentiates between normal blinks and fatigue-induced microsleeps.
- **Yawn Detection (MAR):** Tracks mouth aspect ratio over consecutive temporal frames.
- **Head Pose Estimation (solvePnP):** Computes pitch angle via 3D facial projection to identify nodding and head drops.
- **Instant Audio-Visual Alerts:** Displays live status overlays and triggers audio alarms without blocking video processing.
- **Automated Incident Logging:** Persists time-stamped telemetry data to `fatigue_log.csv` for post-drive fatigue analysis.

---

## 🧠 Underlying Methodology & Metrics

### 1. Eye Aspect Ratio (EAR)
EAR calculates the ratio between the vertical distances of the eyelids and the horizontal width of the eye:

$$\text{EAR} = \frac{\Vert{}p_2 - p_6\Vert{} + \Vert{}p_3 - p_5\Vert{}}{2 \Vert{}p_1 - p_4\Vert{}}$$

- Normal blinking lasts approximately 100–300 ms (3–9 frames at 30 FPS).
- Prolonged closure is flagged when $\text{EAR} < 0.22$ continuously for $\ge 45$ consecutive frames (~1.5 seconds).

### 2. Mouth Aspect Ratio (MAR)
MAR computes vertical lip separation relative to horizontal mouth width:

$$\text{MAR} = \frac{\Vert{}m_2 - m_8\Vert{} + \Vert{}m_3 - m_7\Vert{} + \Vert{}m_4 - m_6\Vert{}}{2 \Vert{}m_1 - m_5\Vert{}}$$

- A yawn event is triggered when $\text{MAR} > 0.65$ persists for $\ge 40$ frames (~1.3 seconds).

### 3. Head Pose & Pitch Estimation
Using OpenCV's `cv2.solvePnP`, 2D facial landmarks (nose tip, chin, eye corners, mouth corners) are projected against a canonical 3D human head model. 
- Pitch angle is derived via Rodrigues transformation and Euler angle decomposition.
- Pitch angles below $-15.0^\circ$ accompanied by partial eye closure trigger a head-nodding sleep alert.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+ / 3.12 / 3.13
- **Computer Vision:** OpenCV (`opencv-python`)
- **Facial Landmark Tracking:** MediaPipe Tasks Vision API (`mediapipe`)
- **Scientific Computing:** NumPy, SciPy (Euclidean distance computation)
- **Audio Dispatch:** Pygame (sound synthesis)

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. A virtual environment is recommended.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/driver-drowsiness-detection.git](https://github.com/your-username/driver-drowsiness-detection.git)
cd driver-drowsiness-detection