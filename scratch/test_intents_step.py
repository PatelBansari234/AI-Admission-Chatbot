import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Step 1: Counter")
from collections import Counter
print("Step 2: numpy")
import numpy as np
print("Step 3: config")
from backend.config import INTENT_DISTANCE_THRESHOLD
print("Step 4: repo")
from backend.db import repo
print("Step 5: embedder")
from backend.embeddings.embedder import embed_documents
print("All steps passed!")
