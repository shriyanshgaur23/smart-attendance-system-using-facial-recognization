"""
attendance.py
Runs the live attendance session:
  1. Loads all registered students' face data from the database
  2. Opens the webcam feed
  3. Detects and recognizes faces in real time
  4. Marks attendance (once per student per day) in the database
  5. Displays a live overlay with names and a "Present" tag

Press 'q' to end the session and export today's attendance to CSV.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None

from database import (
    export_attendance_csv,
    get_all_students,
    get_model_metadata_path,
    get_model_path,
    init_db,
    mark_attendance,
)

MATCH_THRESHOLD = 0.6


def load_known_faces():
    students = get_all_students()
    roll_numbers = [s[0] for s in students]
    names = [s[1] for s in students]
    encodings = [s[2] for s in students]
    return roll_numbers, names, encodings


def _face_cascade():
    cascade_path = Path(__file__).resolve().with_name("haarcascade_frontalface_default.xml")
    if not cascade_path.is_file():
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load face cascade from {cascade_path}")
    return cascade


def _run_cv2_attendance():
    model_path = get_model_path()
    metadata_path = get_model_metadata_path()
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("No trained face model found. Run registration.py first for at least one student.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(metadata_path, "rb") as f:
        label_to_student = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    face_cascade = _face_cascade()
    marked_today = set()
    print("Starting attendance session. Press 'q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))
            label_id, confidence = recognizer.predict(face_roi)

            if label_id in label_to_student and confidence < 80:
                roll_no, name = label_to_student[label_id].split("_", 1) if "_" in label_to_student[label_id] else (label_to_student[label_id], label_to_student[label_id])
                if name.startswith("student"):
                    name = label_to_student[label_id]
                now = datetime.now()
                time_str = now.strftime("%H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")
                newly_marked = mark_attendance(roll_no, name, date_str, time_str)
                if newly_marked and roll_no not in marked_today:
                    marked_today.add(roll_no)
                    print(f"✅ Marked present: {name} ({roll_no}) at {time_str}")
                label_text = f"{name} - Present"
            else:
                label_text = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if "Present" in label_text else (0, 0, 255), 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Smart Attendance System - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = f"attendance_{today}.csv"
    export_attendance_csv(today, csv_path)
    print(f"\nSession ended. Attendance exported to {csv_path}")


def run_attendance():
    if face_recognition is None:
        _run_cv2_attendance()
        return

    roll_numbers, names, known_encodings = load_known_faces()

    if not known_encodings:
        print("No students registered yet. Run registration.py first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    marked_today = set()
    print("Starting attendance session. Press 'q' to stop.\n")
    process_this_frame = True
    face_locations = []
    face_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_labels = []

            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(distances) if len(distances) > 0 else None

                if best_match_index is not None and distances[best_match_index] < MATCH_THRESHOLD:
                    roll_no = roll_numbers[best_match_index]
                    name = names[best_match_index]

                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    newly_marked = mark_attendance(roll_no, name, date_str, time_str)
                    if newly_marked and roll_no not in marked_today:
                        marked_today.add(roll_no)
                        print(f"✅ Marked present: {name} ({roll_no}) at {time_str}")

                    label = f"{name} - Present"
                else:
                    label = "Unknown"

                face_labels.append(label)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), label in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if "Present" in label else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Smart Attendance System - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = f"attendance_{today}.csv"
    export_attendance_csv(today, csv_path)
    print(f"\nSession ended. Attendance exported to {csv_path}")


if __name__ == "__main__":
    init_db()
    run_attendance()
