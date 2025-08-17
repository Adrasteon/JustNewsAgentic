# Dockerfile for MCP Message Bus
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Python build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install setuptools and wheel first to replace distutils  
RUN pip install --upgrade pip setuptools wheel

# Copy the local requirements.txt first
COPY mcp_bus/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY mcp_bus/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
