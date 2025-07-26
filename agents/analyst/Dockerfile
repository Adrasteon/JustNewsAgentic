
FROM python:3.11-slim

WORKDIR /app

COPY agents/analyst/requirements_analyst.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Install additional GPU dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir transformers accelerate safetensors sentencepiece protobuf

COPY agents/analyst/main.py .
COPY agents/analyst/tools.py .

ENV PYTHONPATH="/app"

EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
