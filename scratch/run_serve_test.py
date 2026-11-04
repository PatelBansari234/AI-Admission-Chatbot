import asyncio
import sys
import os
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"

import uvicorn
from backend.api.main import app

async def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    try:
        print("Calling server.serve()...")
        await server.serve()
    except BaseException as e:
        print(f"SERVE ERROR: {type(e)}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as e:
        print(f"MAIN ERROR: {type(e)}: {e}")
        traceback.print_exc()
