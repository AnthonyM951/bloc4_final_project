FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for psycopg2 & Spark
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl openjdk-21-jre-headless ffmpeg \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose Flask/Gunicorn
EXPOSE 5000

# Run with Gunicorn in prod
CMD ["python", "src/app.py"]
