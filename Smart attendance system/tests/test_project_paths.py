import unittest
from pathlib import Path

from database import get_db_path, get_dataset_dir


class ProjectPathTests(unittest.TestCase):
    def test_database_path_is_explicit_and_absolute(self):
        db_path = Path(get_db_path())
        self.assertTrue(db_path.is_absolute())
        self.assertTrue(db_path.name == "attendance.db")

    def test_dataset_dir_is_explicit_and_absolute(self):
        dataset_dir = Path(get_dataset_dir())
        self.assertTrue(dataset_dir.is_absolute())
        self.assertTrue(dataset_dir.name == "dataset")


if __name__ == "__main__":
    unittest.main()
