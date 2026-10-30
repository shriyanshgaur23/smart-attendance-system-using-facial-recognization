"""
registration.py
Enrolls a new student into the system:
  1. Opens the webcam
  2. Captures several images of the student's face
  3. Stores the images and trains a recognizer model
  4. Saves the roll number + name into the SQLite database

Run this once per student before running attendance.py
"""

import os
import pickle
from pathlib import Path

import cv2
import numpy as np

try:
    import face_recognition
except ImportError:  # pragma: no cover - handled by OpenCV fallback
    face_recognition = None

from database import ( 
    add_student,
    get_dataset_dir,
    get_model_metadata_path,
    get_model_path,
    init_db,
)

DATASET_DIR = get_dataset_dir()
IMAGES_TO_CAPTURE = 20


def _get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def capture_face_images(roll_no, name):
    """Capture multiple face images for a student and return their folder path."""
    student_dir = os.path.join(DATASET_DIR, f"{roll_no}_{name}")
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam. Check camera connection/permissions.")

    face_cascade = _get_face_cascade()
    count = 0
    print(f"\nCapturing {IMAGES_TO_CAPTURE} images for {name} ({roll_no}).")
    print("Look at the camera and move your head slightly for variety.")
    print("Press 'q' to stop early.\n")

    while count < IMAGES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            img_path = os.path.join(student_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1
            break

        cv2.putText(frame, f"Captured: {count}/{IMAGES_TO_CAPTURE}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Registering Face - press q to quit early", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return student_dir


def generate_encoding(student_dir):
    """Compute a face encoding if face_recognition is available; otherwise raise a clear fallback error."""
    if face_recognition is None:
        raise ValueError("Fallback mode uses trained OpenCV LBPH models instead of face_recognition encodings.")

    encodings = []
    for filename in os.listdir(student_dir):
        img_path = os.path.join(student_dir, filename)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            continue
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if face_encodings:
            encodings.append(face_encodings[0])

    if not encodings:
        raise ValueError("No usable face encodings found. Try capturing again with better lighting.")

    avg_encoding = np.mean(encodings, axis=0)
    return avg_encoding


def train_opencv_model():
    """Train an LBPH matcher from the images stored for all enrolled students."""
    if not os.path.isdir(DATASET_DIR):
        return

    samples = []
    labels = []
    label_to_student = {}

    student_dirs = sorted(os.listdir(DATASET_DIR))
    for label_id, student_dir_name in enumerate(student_dirs):
        student_dir = os.path.join(DATASET_DIR, student_dir_name)
        if not os.path.isdir(student_dir):
            continue

        label_to_student[label_id] = student_dir_name
        face_cascade = _get_face_cascade()

        for filename in sorted(os.listdir(student_dir)):
            img_path = os.path.join(student_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))
            samples.append(face_roi)
            labels.append(label_id)

    if not samples:
        raise ValueError("No valid training samples were found in the dataset.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(labels, dtype=np.int32))
    recognizer.write(get_model_path())

    with open(get_model_metadata_path(), "wb") as f:
        pickle.dump(label_to_student, f)

    return label_to_student


def register_student():
    roll_no = input("Enter Roll Number: ").strip()
    name = input("Enter Name: ").strip()

    student_dir = capture_face_images(roll_no, name)

    if face_recognition is not None:
        print("Generating face encoding, please wait...")
        encoding = generate_encoding(student_dir)
        add_student(roll_no, name, encoding)
    else:
        # Store a lightweight placeholder for compatibility with the database schema.
        add_student(roll_no, name, np.zeros((128,), dtype=np.float32))
        train_opencv_model()

    print(f"\n✅ {name} ({roll_no}) registered successfully!\n")


if __name__ == "__main__":
    init_db()
    while True:
        register_student()
        again = input("Register another student? (y/n): ").strip().lower()
        if again != 'y':
            break
