
FROM python:3.11-slim

WORKDIR /app

COPY agents/synthesizer/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY agents/synthesizer/main.py .
COPY agents/synthesizer/tools.py .

ENV PYTHONPATH="/app"

EXPOSE 8005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
