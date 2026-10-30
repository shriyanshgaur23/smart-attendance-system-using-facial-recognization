# Smart Attendance System Using Facial Recognition

## Files
- `database.py` — SQLite setup: students table + attendance table
- `registration.py` — Run this first to enroll each student (captures face images, saves encoding)
- `attendance.py` — Run this to start a live attendance session
- `dataset/` — Auto-created folder where captured face images are stored
- `attendance.db` — Auto-created SQLite database (students + attendance records)

## Setup

1. **Install Python 3.8–3.10** (face_recognition/dlib can be finicky on very new Python versions).

2. **Install dependencies:**
   ```bash
   pip install opencv-python face_recognition numpy
   ```

   Note: `face_recognition` depends on `dlib`, which needs CMake and a C++ compiler.
   - **Windows:** install [CMake](https://cmake.org/download/) and Visual Studio Build Tools first, then `pip install dlib` (or use `pip install face_recognition` directly — it may pull a prebuilt wheel depending on your Python version).
   - **Linux:** `sudo apt install cmake` then `pip install dlib face_recognition`.
   - **Mac:** `brew install cmake` then `pip install dlib face_recognition`.

   If dlib installation keeps failing, an easier alternative is to swap `face_recognition` for OpenCV's built-in **LBPH Face Recognizer** (`cv2.face.LBPHFaceRecognizer_create()`) — lower accuracy but zero install headaches. Ask if you want that version of the code instead.

## How to Run

1. **Register students** (run once per student):
   ```bash
   python registration.py
   ```
   Enter roll number and name, look at the camera, it'll capture 20 images and save the face encoding.

2. **Start attendance session:**
   ```bash
   python attendance.py
   ```
   The webcam opens, recognizes registered faces, and automatically marks attendance (once per person per day). Press `q` to end the session — this also exports the day's attendance to a CSV file.

## How It Works (quick recap)
1. `registration.py` captures ~20 face images per student and converts each into a 128-dimension "face encoding" using a pretrained deep learning model (bundled inside `face_recognition`). The 20 encodings are averaged into one representative vector per student and stored in SQLite.
2. `attendance.py` continuously reads webcam frames, detects any faces present, encodes them the same way, and compares each against every stored encoding using Euclidean distance. A distance below `MATCH_THRESHOLD` (0.6) counts as a match.
3. On a match, it inserts `(roll_no, name, date, time)` into the `attendance` table — a `UNIQUE(roll_no, date)` constraint stops duplicate entries for the same day.

## Tuning
- `IMAGES_TO_CAPTURE` in `registration.py` — more images = better accuracy, slower enrollment.
- `MATCH_THRESHOLD` in `attendance.py` — lower it (e.g. 0.5) for stricter matching if you get false positives; raise it (e.g. 0.65) if genuine matches are being missed.

## For Your Report/Viva
- Mention the accuracy-vs-lighting tradeoff, and that Haar Cascade (used only for quick face capture during registration) is faster but less accurate than the CNN/HOG-based detector used during recognition.
- Known limitations to discuss: poor lighting, glasses/masks reducing accuracy, no liveness detection (a photo could theoretically fool it) — good material for your "Limitations" and "Future Scope" sections.
