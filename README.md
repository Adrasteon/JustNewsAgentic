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

    - `synthesizer`: Clusters and synthesizes articles using ML models (sentence-transformers for clustering, LLM for neutralization/aggregation, feedback logging for continual learning). Supports KMeans, BERTopic, and HDBSCAN clustering (set `SYNTHESIZER_CLUSTER_METHOD`).
    - `critic`: Reviews synthesis and neutrality using LLM-based critique, logs feedback for continual learning and retraining. Optional fact-checking pipeline for cross-referencing (enable with `CRITIC_USE_FACTCHECK=1`).
    - `memory`: Unified data access (PostgreSQL, vector search with pgvector, semantic retrieval with embeddings, feedback logging, and retrieval usage tracking for learning-to-rank). Key tool interfaces are mirrored in `tools.py` for clarity and local testing.
    - `db`: PostgreSQL database

### Database Setup & Migrations


Database migrations are located in `agents/memory/db_migrations/`. To apply them manually (from the project root):

```bash
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/001_create_articles_table.sql
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/002_create_training_examples_table.sql
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/003_create_article_vectors_table.sql
```


### ML, Feedback Loops & Continual Learning

All major agents implement machine learning-based feedback loops for continual improvement:

- **Synthesizer**: Uses sentence-transformers for clustering (KMeans, BERTopic, HDBSCAN), LLM for neutralization/aggregation, and logs feedback from Critic/Chief Editor for continual learning. Clustering method is set via `SYNTHESIZER_CLUSTER_METHOD`.
- **Critic**: Uses LLM for critique, logs all feedback and editorial outcomes for retraining and adaptation. Optional fact-checking pipeline can be enabled with `CRITIC_USE_FACTCHECK=1`.
- **Memory**: Provides semantic retrieval with embeddings and vector search, logs all retrievals and downstream outcomes for future learning-to-rank and model improvement. Tool interfaces are mirrored in `tools.py` for clarity and local testing.

**Feedback Logging:**
- All agents log feedback to agent-specific files (e.g., `./feedback_synthesizer.log`) and/or the database.
- Feedback includes tool usage, outcomes, errors, and user/editorial input.
- Standardized logging format: UTC timestamp, event, details (as JSON/dict).

**Retraining & Continual Learning:**
- Feedback logs are used for both online and scheduled retraining.
- Retraining procedures are documented in each agent's code and can be automated via scripts or CI/CD.
- See `action_plan.md` and agent `tools.py` for retraining hooks and feedback usage.

### API Endpoints

Each agent exposes its tools as HTTP endpoints (see `main.py` in each agent folder). The MCP bus provides `/register`, `/call`, and `/agents` endpoints for agent discovery and tool invocation.

