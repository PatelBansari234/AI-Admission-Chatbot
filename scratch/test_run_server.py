import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import uvicorn
    from backend.api.main import app
    print("App imported successfully!")
    # test running server for 2 seconds
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    print("Server configured successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
