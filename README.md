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

### Dual Functionality: Standalone Execution & MCP Bus Integration

Each agent can be started independently while maintaining the ability to communicate with other agents via the MCP Bus. Follow the instructions below for standalone execution and MCP Bus integration:

#### Standalone Execution
Each agent can operate independently without relying on other agents or services. See the "Standalone Execution" section above for detailed instructions.

#### MCP Bus Integration
When the MCP Bus is available, agents will automatically register their tools and use the bus for inter-agent communication. This allows for dynamic collaboration between agents.

**Key Features:**
- Agents register their tools with the MCP Bus upon startup.
- If the MCP Bus is unavailable, agents will continue to operate independently.
- Error handling ensures smooth operation in standalone mode.

**Starting the MCP Bus:**
1. Navigate to the `mcp_bus` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the MCP Bus:
    ```bash
    uvicorn main:app --reload --port 8000
    ```

**Agent Registration:**
Agents will automatically attempt to register their tools with the MCP Bus at `http://localhost:8000`. Ensure the MCP Bus is running before starting agents for full functionality.

### Standalone Execution

Each agent can be started independently without relying on other agents or services. Follow the instructions below for standalone execution:

#### Chief Editor Agent
1. Navigate to the `agents/chief_editor` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8001
    ```

#### Scout Agent
1. Navigate to the `agents/scout` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8002
    ```

#### Fact-Checker Agent
1. Navigate to the `agents/fact_checker` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8003
    ```

#### Analyst Agent
1. Navigate to the `agents/analyst` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8004
    ```

#### Synthesizer Agent
1. Navigate to the `agents/synthesizer` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8005
    ```

#### Critic Agent
1. Navigate to the `agents/critic` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8006
    ```

#### Memory Agent
1. Navigate to the `agents/memory` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8007
    ```