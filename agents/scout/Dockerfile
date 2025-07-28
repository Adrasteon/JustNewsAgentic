
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Python build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install setuptools and wheel first to replace distutils
RUN pip install --upgrade pip setuptools wheel

COPY agents/scout/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/scout/main.py .
COPY agents/scout/tools.py .

ENV PYTHONPATH="/app"

EXPOSE 8006
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
