# AskDocs container: CPU only, runs as a non-root user.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy only what is needed to install the package (see .dockerignore).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# /data holds the SQLite index and the downloaded embedding model (a volume).
# /corpus is where your documents live (mounted from ./corpus; writable so the
# web page can upload documents).
RUN useradd --create-home --uid 10001 askdocs \
    && mkdir -p /data /corpus \
    && chown askdocs:askdocs /data /corpus

ENV ASKDOCS_CORPUS_DIR=/corpus \
    ASKDOCS_DB_PATH=/data/askdocs.sqlite \
    ASKDOCS_MODEL_CACHE_DIR=/data/models

USER askdocs
EXPOSE 8000

# The slim image has no curl, so the health check uses Python's standard library.
# start-period allows for the first-run model download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["askdocs", "serve", "--host", "0.0.0.0", "--port", "8000"]
