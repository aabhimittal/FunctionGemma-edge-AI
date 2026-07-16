# Minimal serving image. The core library has no dependencies, so the image only
# needs the serving extras. Multi-stage keeps the runtime layer small.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the serving dependencies first for better layer caching.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy the application.
COPY functiongemma ./functiongemma
COPY serving ./serving
COPY data ./data
COPY configs ./configs

# Optional: a fitted calibrator can be baked in or mounted at runtime.
# COPY calibrator.json .

EXPOSE 8000

# Simple container healthcheck hitting the liveness endpoint.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
