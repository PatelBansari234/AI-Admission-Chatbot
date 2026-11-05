FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/app/models/hf_cache \
    LLM_PROVIDER=gemini \
    EMBED_DEVICE=cpu \
    RERANK_DEVICE=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt /app/requirements-prod.txt
RUN pip install --upgrade pip && pip install -r /app/requirements-prod.txt

# Pre-download BGE embedding and reranker models during image build so container startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='/app/models/embeddings'); CrossEncoder('BAAI/bge-reranker-base')"

COPY backend /app/backend
COPY data /app/data
COPY vector_db /app/vector_db

RUN mkdir -p /app/data/raw/pdfs /app/data/raw/websie /app/data/processed /app/vector_db /app/models/hf_cache \
    && chmod -R 777 /app

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
