FROM python:3.11-slim
WORKDIR /app
COPY agents/mcp_bus/requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY agents/mcp_bus/main.py .

ENV PYTHONPATH="/app"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
