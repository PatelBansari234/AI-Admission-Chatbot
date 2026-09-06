"""One-time setup: apply schema.sql to the chatbot database.

    python -m backend.db.init_db

Safe to re-run - every CREATE TABLE/INDEX in schema.sql uses IF NOT EXISTS.
Assumes the database and app role already exist (a plain LOGIN role can't
create its own database) - see README for the one-time superuser step that
creates them.
"""

from pathlib import Path

import psycopg

from backend.config import DATABASE_URL, DB_NAME, DB_USER

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def ensure_database() -> None:
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5):
            pass
    except psycopg.OperationalError as exc:
        raise SystemExit(
            f"Could not connect to database '{DB_NAME}' as role '{DB_USER}': {exc}\n\n"
            "Create the role and database first, as a Postgres superuser:\n"
            f"  CREATE ROLE {DB_USER} WITH LOGIN PASSWORD '<pick a password>';\n"
            f"  CREATE DATABASE {DB_NAME} OWNER {DB_USER};\n"
            "Then set DB_PASSWORD (and DB_USER/DB_NAME if changed) in .env."
        )
    print(f"Connected to '{DB_NAME}' as '{DB_USER}'")


def apply_schema() -> None:
    """Splits schema.sql on bare semicolons - too naive to handle a
    semicolon inside a string literal or a DO block, but schema.sql doesn't
    use either. It DOES need full-line "--" comments stripped first, though:
    one previously broke this by containing a semicolon in its own text
    ("...UNIQUE; two colleges...") and got split mid-sentence into an
    unprefixed fragment Postgres tried to parse as SQL."""
    lines = SCHEMA_PATH.read_text(encoding="utf-8").splitlines()
    sql = "\n".join(line for line in lines if not line.strip().startswith("--"))
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        for stmt in statements:
            conn.execute(stmt)
    print(f"Applied schema ({len(statements)} statements)")


def main() -> None:
    ensure_database()
    apply_schema()


if __name__ == "__main__":
    main()
