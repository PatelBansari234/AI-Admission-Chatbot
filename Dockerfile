FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    HF_HOME=/app/models/hf_cache \
    LLM_PROVIDER=gemini \
    GEMINI_MODEL=gemini-3.8-flash \
    LOW_MEMORY_MODE=true \
    EMBED_DEVICE=cpu \
    RERANK_DEVICE=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt /app/requirements-prod.txt
RUN pip install --upgrade pip && pip install -r /app/requirements-prod.txt

# Pre-download BAAI/bge-small-en-v1.5 safetensors & tokenizer (133 MB) during build
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-small-en-v1.5', cache_dir='/app/models/embeddings', allow_patterns=['*.json', '*.safetensors', '*.txt'])"

COPY backend /app/backend
COPY data /app/data

# Extract the pre-indexed ChromaDB seed archive into /app/vector_db during image build
RUN mkdir -p /app/vector_db /app/data/raw/pdfs /app/data/raw/websie /app/data/processed /app/models/hf_cache \
    && python -c "import tarfile, pathlib; p = pathlib.Path('/app/data/vector_db_seed.tar.gz'); tarfile.open(p, 'r:gz').extractall('/app/vector_db') if p.exists() else None" \
    && chmod -R 777 /app

EXPOSE 10000

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
