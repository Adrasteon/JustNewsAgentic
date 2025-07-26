
FROM python:3.11-slim

WORKDIR /app

COPY agents/scout/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY agents/scout/main.py .
COPY agents/scout/tools.py .

ENV PYTHONPATH="/app"

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
