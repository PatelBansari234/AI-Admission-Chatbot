import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"

import uvicorn
from backend.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
