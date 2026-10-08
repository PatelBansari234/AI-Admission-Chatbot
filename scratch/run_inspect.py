import traceback
try:
    with open("scratch/inspect_part4.py", "r", encoding="utf-8") as f:
        code = f.read()
    exec(code)
except Exception:
    with open("scratch/inspect_err.txt", "w", encoding="utf-8") as out:
        traceback.print_exc(file=out)
    traceback.print_exc()
