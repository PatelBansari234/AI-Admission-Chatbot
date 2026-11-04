import os

# On Windows, ensure PostgreSQL libpq is discoverable by psycopg before any backend submodules load.
if os.name == "nt":
    for _pg_path in (
        r"C:\PostgreSQL17\pgsql\bin",
        r"C:\Program Files\PostgreSQL\17\bin",
        r"C:\Program Files\PostgreSQL\16\bin",
    ):
        if os.path.exists(_pg_path) and _pg_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + ";" + _pg_path
            break
