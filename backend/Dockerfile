FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ROOT=/var/data/models

# GDAL/PROJ/GEOS runtime for rasterio, fiona, pyproj and shapely wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/app ./app
COPY backend/pytest.ini ./pytest.ini
COPY backend/tests ./tests
COPY frontend /frontend
COPY models /var/data/models
ENV SPATIALSHIFT_FRONTEND_ROOT=/frontend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
