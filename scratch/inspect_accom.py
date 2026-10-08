import sys
import os

sys.path.insert(0, os.getcwd())

from backend.vectorstore import store

chunks = [
    c for c in store.get_all_chunks(college_id=1)
    if 'admissions-accommodation.html' in c['metadata'].get('source', '')
]
print(f"Total accommodation chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    safe_preview = c["text"][:160].encode('ascii', errors='backslashreplace').decode('ascii')
    print(f"\n--- Chunk {i+1} (length {len(c['text'])}) ---")
    print(safe_preview)
