# Dockerfile for MCP Message Bus
FROM python:3.11-slim

WORKDIR /app

# Copy the local requirements.txt first
COPY mcp_bus/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY mcp_bus/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
