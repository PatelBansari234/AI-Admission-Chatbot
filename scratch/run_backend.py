import traceback
from pathlib import Path

log_path = Path(__file__).resolve().parent.parent / "backend_crash.log"

try:
    import os
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"
    
    import backend.api.main
    import uvicorn
    
    uvicorn.run(backend.api.main.app, host="127.0.0.1", port=8000)
except BaseException as e:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(traceback.format_exc())
