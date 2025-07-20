# JustNewsAgentic

This project implements the JustNews V3 system, an agentic, MCP-first news analysis ecosystem. It is designed as a collaborative group of specialized AI agents that work together to find, analyze, and synthesize news stories in a way that is clear, factually correct, and free of bias.

## Architecture

The system is built on a microservices architecture where each service is an independent AI agent. These agents communicate via a central **MCP (Model Context Protocol) Message Bus**. This allows for a flexible, scalable, and dynamic system where agents can delegate tasks and collaborate to achieve complex goals.

For full architectural details, see `docs/JustNews_Proposal_V3.md` and `docs/JustNews_Plan_V3.md`.


## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Build & Run

1. Build and start all services:
    ```bash
    docker-compose up --build
    ```

2. The following services will be started:
    - `mcp_bus`: Central message bus for agent communication
    - `chief_editor`: Orchestrates news workflows
    - `scout`: Discovers and crawls news sources
    - `fact_checker`: Validates news and verifies claims
    - `analyst`: Scores bias, sentiment, and extracts entities
    - `synthesizer`: Clusters and synthesizes articles
    - `critic`: Reviews synthesis and neutrality
    - `memory`: Unified data access (PostgreSQL, vector search, training logs)
    - `db`: PostgreSQL database

### Database Setup & Migrations


Database migrations are located in `agents/memory/db_migrations/`. To apply them manually (from the project root):

```bash
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/001_create_articles_table.sql
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/002_create_training_examples_table.sql
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/003_create_article_vectors_table.sql
```

### API Endpoints

Each agent exposes its tools as HTTP endpoints (see `main.py` in each agent folder). The MCP bus provides `/register`, `/call`, and `/agents` endpoints for agent discovery and tool invocation.

