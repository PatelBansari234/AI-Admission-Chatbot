"""Database Connection Pool Diagnostic Script for SXCA Admission Chatbot."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from backend.config import DATABASE_URL
from backend.db.connection import get_pool, get_conn


def run_db_diagnostic():
    print("=" * 60)
    print("RUNNING DATABASE POOL DIAGNOSTIC")
    print("=" * 60)

    # 1. PostgreSQL reachable
    postgres_reachable = False
    try:
        conn_direct = psycopg.connect(DATABASE_URL, connect_timeout=5)
        conn_direct.close()
        postgres_reachable = True
        print("PostgreSQL reachable: YES")
    except Exception as e:
        print(f"PostgreSQL reachable: NO ({e})")

    if not postgres_reachable:
        print("Pool created: NO")
        print("Connection acquired: NO")
        print("Simple query: FAIL")
        print("Connection returned: NO")
        print("Pool healthy after release: NO")
        return

    # 2. Pool created
    pool_created = False
    try:
        pool = get_pool()
        pool_created = True
        print("Pool created: YES")
    except Exception as e:
        print(f"Pool created: NO ({e})")

    if not pool_created:
        print("Connection acquired: NO")
        print("Simple query: FAIL")
        print("Connection returned: NO")
        print("Pool healthy after release: NO")
        return

    # 3. Connection acquired & 4. Simple query & 5. Connection returned
    conn_acquired = False
    simple_query_pass = False
    conn_returned = False

    try:
        with get_conn() as conn:
            conn_acquired = True
            row = conn.execute("SELECT 1 AS result").fetchone()
            if row and row.get("result") == 1:
                simple_query_pass = True
        conn_returned = True
        print("Connection acquired: YES")
        print("Simple query: PASS" if simple_query_pass else "Simple query: FAIL")
        print("Connection returned: YES")
    except Exception as e:
        if not conn_acquired:
            print(f"Connection acquired: NO ({e})")
        print("Simple query: FAIL")
        if not conn_returned:
            print(f"Connection returned: NO ({e})")

    # 6. Pool healthy after release (acquire max_size connections and return them)
    pool_healthy = False
    try:
        max_size = pool.max_size
        conns = []
        # Acquire up to max_size connections
        for i in range(max_size):
            c = pool.getconn(timeout=5.0)
            conns.append(c)
        
        # Release all connections back to pool
        for c in conns:
            pool.putconn(c)
            
        # Try getting one more to verify pool is healthy
        with pool.connection(timeout=5.0) as conn:
            r = conn.execute("SELECT 1 AS res").fetchone()
            if r and r.get("res") == 1:
                pool_healthy = True

        print("Pool healthy after release: YES" if pool_healthy else "Pool healthy after release: NO")
    except Exception as e:
        print(f"Pool healthy after release: NO ({e})")

    print("=" * 60)


if __name__ == "__main__":
    run_db_diagnostic()
