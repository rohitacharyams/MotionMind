# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------
# MotionMind Coach — FastAPI + Groq/LLM + Azure Speech + Gemini Live
#
# Image is intentionally lean (~400 MB). sentence-transformers / torch
# are NOT shipped; semantic_search.search() falls back to keyword
# matching when the optional dep isn't present.
#
# Build:
#   docker build -f coach/Dockerfile -t motionmind-coach .
# ---------------------------------------------------------------------

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GROQ_MODEL=llama-3.3-70b-versatile \
    GROQ_SEEDER_MODEL=llama-3.1-8b-instant \
    GROQ_WRITER_MODEL=llama-3.1-8b-instant \
    WEBSITES_PORT=8770

# curl is used by HEALTHCHECK; ca-certs for Groq + Azure TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first for layer caching.
COPY coach/requirements.txt /app/coach/requirements.txt
RUN pip install --upgrade pip wheel \
    && pip install -r /app/coach/requirements.txt

# Application sources (allowlist .dockerignore strips logs / .env / etc.).
COPY coach            /app/coach
COPY data/characters  /app/data/characters
COPY data/models      /app/data/models
COPY data/motion_db   /app/data/motion_db
# Backing-track music: AIST originals (per-clip) + 3 generic loops
# for CMU/no-music clips. ~125 MB total — acceptable for offline play.
COPY data/audio/aist_orig /app/data/audio/aist_orig
COPY data/audio/music     /app/data/audio/music

# Strip any secrets / scratch files that may have slipped past .dockerignore.
RUN rm -f /app/coach/.env /app/coach/.env.example \
          /app/coach/_meta_seed.log /app/coach/_batch_retarget.log \
    && find /app -name __pycache__ -type d -prune -exec rm -rf {} +

EXPOSE 8770

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8770/healthz || exit 1

CMD ["uvicorn", "coach.server:app", \
     "--host", "0.0.0.0", "--port", "8770", \
     "--log-level", "info", "--no-access-log"]
