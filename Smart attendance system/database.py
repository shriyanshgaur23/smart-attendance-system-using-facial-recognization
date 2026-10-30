"""
database.py
Sets up and manages the SQLite database for the Smart Attendance System.
Two tables:
  1. students   -> stores roll_no, name, and pickled face encoding
  2. attendance -> stores roll_no, name, date, time
"""

import sqlite3
import pickle
from pathlib import Path

DB_NAME = "attendance.db"
BASE_DIR = Path(__file__).resolve().parent


def get_db_path():
    return str((BASE_DIR / DB_NAME).resolve())


def get_dataset_dir():
    dataset_dir = BASE_DIR / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    return str(dataset_dir.resolve())


def get_model_path():
    model_path = BASE_DIR / "face_model.yml"
    return str(model_path.resolve())


def get_model_metadata_path():
    metadata_path = BASE_DIR / "face_model_metadata.pkl"
    return str(metadata_path.resolve())


def get_connection():
    return sqlite3.connect(get_db_path())


def init_db():
    """Create tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            roll_no TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(roll_no, date)
        )
    """)

    conn.commit()
    conn.close()


def add_student(roll_no, name, encoding):
    """Store a student's face encoding (numpy array) as a BLOB."""
    conn = get_connection()
    cursor = conn.cursor()
    encoded_blob = pickle.dumps(encoding)
    cursor.execute(
        "INSERT OR REPLACE INTO students (roll_no, name, encoding) VALUES (?, ?, ?)",
        (roll_no, name, encoded_blob),
    )
    conn.commit()
    conn.close()


def get_all_students():
    """Return list of (roll_no, name, encoding) for all registered students."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT roll_no, name, encoding FROM students")
    rows = cursor.fetchall()
    conn.close()

    students = []
    for roll_no, name, blob in rows:
        encoding = pickle.loads(blob)
        students.append((roll_no, name, encoding))
    return students


def mark_attendance(roll_no, name, date_str, time_str):
    """
    Insert an attendance record. If the student is already marked
    present for that date, this silently does nothing (UNIQUE constraint).
    Returns True if a new record was inserted, False if already marked.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO attendance (roll_no, name, date, time) VALUES (?, ?, ?, ?)",
            (roll_no, name, date_str, time_str),
        )
        conn.commit()
        inserted = True
    except sqlite3.IntegrityError:
        # Already marked present today
        inserted = False
    conn.close()
    return inserted


def get_attendance_for_date(date_str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT roll_no, name, time FROM attendance WHERE date = ? ORDER BY time",
        (date_str,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def export_attendance_csv(date_str, filepath):
    import csv
    rows = get_attendance_for_date(date_str)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Roll No", "Name", "Time"])
        writer.writerows(rows)
    return filepath
