import sys
import os
import traceback

with open("debug_trace.txt", "w") as f:
    f.write("Starting script\n")
    try:
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parent
        f.write(f"PROJECT_ROOT={PROJECT_ROOT}\n")
        sys.path.insert(0, str(PROJECT_ROOT))
        f.write("Importing uvicorn...\n")
        import uvicorn
        f.write("Importing backend.api.main...\n")
        import backend.api.main
        f.write("Imported backend.api.main!\n")
    except Exception as e:
        f.write("Exception caught:\n")
        f.write(traceback.format_exc())
    except BaseException as be:
        f.write(f"BaseException caught: {type(be)}: {be}\n")
        f.write(traceback.format_exc())
