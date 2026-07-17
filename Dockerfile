FROM python:3.11-slim

# build-essential covers any transitive deps without prebuilt wheels for this
# platform; curl is needed for the container HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps in their own layer so code-only changes don't bust the cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm the sentence-transformers embedding model into the image so the
# first request after a cold start doesn't pay for the download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
