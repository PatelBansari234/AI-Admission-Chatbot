import os
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Postgres bin is in PATH
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"

out_file = PROJECT_ROOT / "startup_debug.txt"

def main():
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("Importing main...\n")
        
        from backend.api.main import app
        import uvicorn

        with open(out_file, "a", encoding="utf-8") as f:
            f.write("Imported app successfully. Starting uvicorn...\n")
            
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except BaseException as e:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write("CRASH EXCEPTION:\n")
            f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
