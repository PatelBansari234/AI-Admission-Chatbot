import sys
import os
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


log_file = PROJECT_ROOT / "start_backend.log"

try:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Importing app...\n")
        
    from backend.api.main import app
    import uvicorn
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("Starting uvicorn.run...\n")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
except Exception as e:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("Exception caught:\n")
        f.write(traceback.format_exc())
except BaseException as be:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"BaseException: {be}\n")
        f.write(traceback.format_exc())
