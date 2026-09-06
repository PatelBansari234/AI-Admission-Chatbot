"""PostgreSQL connection pool.

Rows come back as dicts (column name -> value) via dict_row, and every
connection is pinned to UTC so TIMESTAMPTZ columns read back consistently
regardless of the machine's local timezone - matching the UTC timestamps the
rest of the app already assumes.
"""

from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from backend.config import DATABASE_URL

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    conn.execute("SET TIME ZONE 'UTC'")
    conn.commit()  # otherwise the pool hands out a connection stuck mid-transaction


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
            configure=_configure,
            open=True,
        )
    return _pool


@contextmanager
def get_conn():
    with get_pool().connection() as conn:
        yield conn
