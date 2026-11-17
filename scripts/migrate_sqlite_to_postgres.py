#!/usr/bin/env python3
"""Copy everything from the SQLite database into the PostgreSQL database configured in .env.

    python scripts/migrate_sqlite_to_postgres.py                    # uses FR_DB_PATH as the source
    python scripts/migrate_sqlite_to_postgres.py --sqlite other.db
    python scripts/migrate_sqlite_to_postgres.py --truncate         # empty the PostgreSQL tables first

Row ids are kept; the id sequences are moved past the
copied rows afterwards. Run it while the system is stopped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import load_settings  # noqa: E402
from database.connection import Database, PostgresDatabase, open_database  # noqa: E402

# Parent tables first (foreign keys).
TABLES = ["cameras", "persons", "person_counts", "person_crossings"]
SERIAL_TABLES = ["persons", "person_counts", "person_crossings"]
CHUNK = 2000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite", help="source SQLite file (default FR_DB_PATH)")
    ap.add_argument("--truncate", action="store_true", help="delete existing PostgreSQL rows first")
    args = ap.parse_args()

    settings = load_settings()
    src_path = Path(args.sqlite) if args.sqlite else settings.storage.db_path
    if not src_path.exists():
        print(f"source not found: {src_path}")
        return 1
    if settings.storage.backend != "postgres":
        print("PostgreSQL is not configured: set FR_DATABASE_URL (or FR_PG_HOST ...) in .env first")
        return 1

    src = Database(src_path)
    dst = open_database(settings.storage)
    assert isinstance(dst, PostgresDatabase)

    if args.truncate:
        dst.execute("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")
    else:
        busy = [t for t in TABLES if dst.fetchone(f"SELECT COUNT(*) AS n FROM {t}")["n"]]
        if busy:
            print(f"PostgreSQL tables already have rows: {busy}. Re-run with --truncate to replace them.")
            return 1

    print(f"{src.description}  ->  {dst.description}")
    for table in TABLES:
        cols = [r["name"] for r in src.fetchall(f"PRAGMA table_info({table})")]
        col_sql, marks = ", ".join(cols), ", ".join("?" * len(cols))
        total, last_id = 0, None
        while True:
            # keyset pagination (cameras has a TEXT id)
            if last_id is None:
                rows = src.fetchall(f"SELECT {col_sql} FROM {table} ORDER BY id LIMIT {CHUNK}")
            else:
                rows = src.fetchall(f"SELECT {col_sql} FROM {table} WHERE id > ? ORDER BY id LIMIT {CHUNK}",
                                    (last_id,))
            if not rows:
                break
            dst.executemany(f"INSERT INTO {table}({col_sql}) VALUES ({marks})", [tuple(r) for r in rows])
            total += len(rows)
            last_id = rows[-1]["id"]
        print(f"  {table:<14} {total} rows")

    for table in SERIAL_TABLES:
        dst.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)")
    src.close()
    dst.close()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
