FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.py /app/model.py

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-u", "model.py"]
