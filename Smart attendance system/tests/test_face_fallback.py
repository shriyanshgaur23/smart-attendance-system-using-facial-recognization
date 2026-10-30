import importlib.util
from pathlib import Path


def _module_from_path(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registration_and_attendance_import_without_face_recognition():
    root = Path(__file__).resolve().parents[1]
    registration = _module_from_path("registration_fallback_test", root / "registration.py")
    attendance = _module_from_path("attendance_fallback_test", root / "attendance.py")
    assert hasattr(registration, "generate_encoding")
    assert hasattr(attendance, "run_attendance")
