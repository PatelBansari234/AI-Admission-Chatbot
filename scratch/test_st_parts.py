import os
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def t(name, fn):
    t0 = time.time()
    print(f"Testing {name}...", end="", flush=True)
    fn()
    print(f" OK ({time.time() - t0:.2f}s)", flush=True)

t("sentence_transformers.backend", lambda: __import__("sentence_transformers.backend"))
t("sentence_transformers.base.sampler", lambda: __import__("sentence_transformers.base.sampler"))
t("sentence_transformers.cross_encoder", lambda: __import__("sentence_transformers.cross_encoder"))
t("sentence_transformers.sentence_transformer.datasets", lambda: __import__("sentence_transformers.sentence_transformer.datasets"))
t("sentence_transformers.sentence_transformer.model", lambda: __import__("sentence_transformers.sentence_transformer.model"))
t("sentence_transformers.sentence_transformer.trainer", lambda: __import__("sentence_transformers.sentence_transformer.trainer"))
t("sentence_transformers.sparse_encoder", lambda: __import__("sentence_transformers.sparse_encoder"))
t("sentence_transformers.util", lambda: __import__("sentence_transformers.util"))
