import io
import pickle
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from database import (
    add_student,
    export_attendance_csv,
    get_all_students,
    get_attendance_for_date,
    get_dataset_dir,
    get_model_metadata_path,
    get_model_path,
    init_db,
    mark_attendance,
)

init_db()

DATASET_DIR = Path(get_dataset_dir())
CASCADE_FILE = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📸 Smart Attendance System")

def ensure_face_cascade():
    if not CASCADE_FILE.exists():
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, str(CASCADE_FILE))
    return str(CASCADE_FILE)


def face_cascade():
    cascade_path = ensure_face_cascade()
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load face cascade from {cascade_path}")
    return cascade


def detect_face_roi(gray_image):
    faces = face_cascade().detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return cv2.resize(gray_image[y:y + h, x:x + w], (200, 200))


def normalize_image(image):
    if image is None:
        return None

    if isinstance(image, np.ndarray):
        arr = image
    elif hasattr(image, "getvalue"):
        arr = np.array(Image.open(io.BytesIO(image.getvalue())).convert("RGB"))
    else:
        arr = np.array(Image.open(image).convert("RGB"))

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def save_student_image(roll_no, name, image):
    student_dir = DATASET_DIR / f"{roll_no}_{name}"
    student_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(student_dir.iterdir()))
    image_path = student_dir / f"{index}.jpg"
    rgb = normalize_image(image)
    if rgb is None:
        raise ValueError("No image data was provided.")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), bgr)
    return image_path


def train_recognizer():
    samples = []
    labels = []
    metadata = {}
    label_id = 0

    for student_dir in sorted(DATASET_DIR.iterdir()):
        if not student_dir.is_dir():
            continue

        metadata[label_id] = student_dir.name
        for img_file in sorted(student_dir.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            face = detect_face_roi(image)
            if face is not None:
                samples.append(face)
                labels.append(label_id)

        label_id += 1

    if not samples:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(labels, dtype=np.int32))
    recognizer.write(str(Path(get_model_path())))

    with open(get_model_metadata_path(), "wb") as f:
        pickle.dump(metadata, f)

    return recognizer, metadata


def recognize_image(image):
    model_path = Path(get_model_path())
    metadata_path = Path(get_model_metadata_path())

    if not model_path.exists() or not metadata_path.exists():
        return "Unknown", None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    rgb = normalize_image(image)
    if rgb is None:
        return "Unknown", None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    face = detect_face_roi(gray)
    if face is None:
        return "Unknown", None

    label_id, confidence = recognizer.predict(face)
    if label_id not in metadata or confidence > 80:
        return "Unknown", None

    student_name = metadata[label_id]
    if "_" in student_name:
        roll_no, name = student_name.split("_", 1)
    else:
        roll_no, name = student_name, student_name

    return name, roll_no


def register_student(roll_no, name, image):
    if not roll_no or not name or image is None:
        return False, "Please provide a roll number, name, and a face image."

    save_student_image(roll_no, name, image)
    add_student(roll_no, name, np.zeros((128,), dtype=np.float32))
    train_recognizer()
    return True, f"{name} ({roll_no}) registered successfully."


login_tab, attendance_tab, record_tab = st.tabs(["Register Student", "Take Attendance", "View Records"])

with login_tab:
    st.subheader("Register a New Student")
    st.write("Use your camera or upload a front-facing photo.")
    roll_no = st.text_input("Roll Number")
    name = st.text_input("Student Name")
    captured = st.camera_input("Capture registration photo")
    uploaded = st.file_uploader("Or upload a photo", type=["png", "jpg", "jpeg"])

    image = captured if captured is not None else uploaded
    if image is not None:
        st.image(image, caption="Preview", use_container_width=True)

    if st.button("Register Student"):
        if image is None:
            st.warning("Capture or upload a student photo first.")
        else:
            ok, message = register_student(roll_no, name, image)
            if ok:
                st.success(message)
            else:
                st.error(message)

with attendance_tab:
    st.subheader("Take Attendance")
    st.write("Capture a face photo and the system will try to match it to a registered student.")
    attendance_image = st.camera_input("Capture attendance photo")
    if attendance_image is not None:
        st.image(attendance_image, caption="Attendance photo", use_container_width=True)

    if st.button("Mark Attendance"):
        if attendance_image is None:
            st.warning("Capture a photo before marking attendance.")
        else:
            name, roll_no = recognize_image(attendance_image)
            if name == "Unknown":
                st.warning("No registered student matched this face.")
            else:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                marked = mark_attendance(roll_no, name, date_str, time_str)
                if marked:
                    export_attendance_csv(date_str, f"attendance_{date_str}.csv")
                    st.success(f"Attendance marked for {name} ({roll_no}) at {time_str}.")
                else:
                    st.info(f"{name} ({roll_no}) is already marked for today.")

with record_tab:
    st.subheader("View Attendance Records")
    selected_date = st.date_input("Select date", datetime.now())
    date_str = selected_date.strftime("%Y-%m-%d")
    records = get_attendance_for_date(date_str)

    if records:
        st.table([{"Roll No": r[0], "Name": r[1], "Time": r[2]} for r in records])
    else:
        st.warning(f"No attendance records found for {date_str}.")

    st.subheader("Registered Students")
    students = get_all_students()
    if students:
        st.table([{"Roll No": s[0], "Name": s[1]} for s in students])
    else:
        st.warning("No students registered yet.")