# Dockerfile for Chief Editor Agent
FROM python:3.11-slim

WORKDIR /app

COPY agents/chief_editor/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY agents/chief_editor/main.py .
COPY agents/chief_editor/tools.py .

ENV PYTHONPATH="/app"

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
