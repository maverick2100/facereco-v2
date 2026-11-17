#!/usr/bin/env python3
"""Register cameras from config/cameras.yaml in the existing database. Creates no tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import load_settings  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import open_database  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def main() -> int:
    settings = load_settings()
    setup_logging(settings.logging_config, settings.log_level)
    db = open_database(settings.storage)
    n = CameraRepository(db).sync(settings.cameras)
    print(f"database : {db.description}")
    print(f"cameras  : {n} registered")
    print(f"persons  : {len(PersonRepository(db).list())}")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
