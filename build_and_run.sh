#!/bin/bash

# Build and start all services

echo "Building and starting all services..."
docker-compose up --build

# List running containers
echo "Listing running containers..."
docker ps

# Print logs for all containers
echo "Printing logs for all containers..."
docker-compose logs --tail=100

# Health check for MCP Bus
echo "Checking health of MCP Bus..."
for i in {1..20}; do
  if curl -sf http://localhost:8000/health; then
    echo "MCP Bus is healthy."
    break
  else
    echo "Waiting for MCP Bus on port 8000..."
    sleep 2
  fi
  if [ $i -eq 20 ]; then
    echo "MCP Bus failed health check.";
    docker-compose logs mcp_bus
    exit 1
  fi
done

echo "System build and run complete."
