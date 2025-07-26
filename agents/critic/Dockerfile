# Dockerfile for Critic Agent
FROM python:3.11-slim

WORKDIR /app

COPY agents/critic/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY agents/critic/main.py .
COPY agents/critic/tools.py .

ENV PYTHONPATH="/app"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
