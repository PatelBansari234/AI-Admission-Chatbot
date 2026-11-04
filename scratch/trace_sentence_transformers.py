import sys
import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def trace(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

trace("Importing torch...")
import torch
trace("Torch imported successfully")

trace("Importing scipy...")
import scipy
trace("Scipy imported successfully")

trace("Importing scipy.linalg...")
import scipy.linalg
trace("Scipy.linalg imported successfully")

trace("Importing sklearn...")
import sklearn
trace("Sklearn imported successfully")

trace("Importing sentence_transformers...")
import sentence_transformers
trace("SentenceTransformers imported successfully!")
