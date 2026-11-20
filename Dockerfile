FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenMP runtime required by LightGBM and XGBoost on Debian/Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/models /app/data /app/logs

EXPOSE 8000 8501

# Default command starts the FastAPI inference service
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
