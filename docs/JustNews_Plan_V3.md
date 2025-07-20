# JustNews V3: Agentic Implementation Plan

This document provides the engineering plan for building the JustNews V3 system as outlined in the `JustNews_Proposal_V3.md`. It details the agent-based architecture, the MCP tool specifications for each agent, and a file-by-file project structure.

## 1. Guiding Principles

The V3 architecture is guided by the following principles:

-   **Agent-First:** The primary unit of software is the **Agent**, a self-contained process hosting an LLM and a set of tools.
-   **MCP-Native:** Communication between agents is exclusively handled via a central **MCP Message Bus**. Agents expose their capabilities as tools on this bus.
-   **Dynamic Orchestration:** The workflow is not a fixed pipeline but a dynamic collaboration between agents, directed by a high-level **Chief Editor Agent**.
-   **Unified Memory:** All persistent data (articles, metadata, training examples) is managed by a dedicated **Memory Agent**, which provides data access to other agents via MCP tools.

## 2. Technology Stack

The technology stack remains largely the same as V2, with the key addition of an MCP-enabled message bus.

| Component | Technology | Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Industry standard for AI/ML. |
| **Agent Framework** | FastAPI | Ideal for creating the API endpoints for each agent. |
| **Containerization** | Docker & Docker Compose | Essential for managing the multi-agent architecture. |
| **MCP Message Bus** | RabbitMQ (with MCP layer) | Enables reliable, asynchronous, tool-based communication. |
| **Core Database** | PostgreSQL 16+ / `pgvector` | Managed exclusively by the Memory Agent. |
| **Graph Database** | Neo4j | Managed exclusively by the Memory Agent. |
| **AI/ML Libraries** | Hugging Face `transformers` | Provides access to open-source models. |
| **CI/CD** | GitHub Actions | For automated testing and deployment. |

## 3. Agent & Tool Specification

This section defines the core tools each agent will expose on the MCP bus.

#### 1. Chief Editor Agent
-   **LLM:** `Llama-3-70B-Instruct`
-   **Purpose:** High-level orchestration and final editorial control.
-   **MCP Tools Exposed:**
    -   `request_story_brief(topic: str, scope: str)`: Initiates the workflow for a new story.
    -   `publish_story(story_id: str)`: Approves and publishes a final story.

#### 2. Scout Agent
-   **LLM:** `Llama-3-8B-Instruct`
-   **Purpose:** Web discovery and content retrieval.
-   **MCP Tools Exposed:**
    -   `discover_sources(query: str) -> list[str]`: Searches the web for relevant URLs.
    -   `crawl_url(url: str, extraction_prompt: str | None) -> str`: Fetches and extracts content from a single URL.
    -   `deep_crawl_site(domain: str, keywords: list[str]) -> list[str]`: Performs a deep crawl on a specific website.

#### 3. Fact-Checker Agent
-   **LLM:** `Mistral-7B-Instruct-v0.2`
-   **Purpose:** Initial validation and claim verification.
-   **MCP Tools Exposed:**
    -   `validate_is_news(content: str) -> bool`: Checks if content is a news article.
    -   `verify_claims(claims: list[str], sources: list[str]) -> dict`: Verifies specific claims against provided source material.

#### 4. Analyst Agent
-   **LLM:** `Mistral-7B-Instruct-v0.2`
-   **Purpose:** Content analysis and scoring.
-   **MCP Tools Exposed:**
    -   `score_bias(text: str) -> float`
    -   `score_sentiment(text: str) -> float`
    -   `identify_entities(text: str) -> list[str]`

#### 5. Synthesizer Agent
-   **LLM:** `Llama-3-70B-Instruct` (fine-tuned for aggregation)
-   **Purpose:** Clustering, neutralization, and synthesis.
-   **MCP Tools Exposed:**
    -   `cluster_articles(article_ids: list[str]) -> list[list[str]]`
    -   `neutralize_text(text: str) -> str`
    -   `aggregate_cluster(article_ids: list[str]) -> str`

#### 6. Critic Agent
-   **LLM:** `Llama-3-8B-Instruct`
-   **Purpose:** Quality control and training data generation.
-   **MCP Tools Exposed:**
    -   `critique_synthesis(summary: str, source_ids: list[str]) -> str`: Checks for hallucinations and omissions.
    -   `critique_neutrality(original_text: str, neutralized_text: str) -> str`: Checks for factual drift after neutralization.

#### 7. Memory Agent
-   **LLM:** None (Acts as a tool provider)
-   **Purpose:** Unified data access layer.
-   **MCP Tools Exposed:**
    -   `save_article(content: str, metadata: dict) -> str`: Saves an article, returns its ID.
    -   `get_article(article_id: str) -> dict`: Retrieves an article.
    -   `vector_search_articles(query: str, top_k: int) -> list[dict]`: Finds similar articles.
    -   `log_training_example(task: str, input: dict, output: dict, critique: str)`: Logs a new training example.

## 4. Project Structure (Agent-Based)

```
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── docker-compose.yml      # Defines and configures all AGENTS for local dev
├── JustNews_Proposal_V3.md
├── JustNews_Plan_V3.md
└── agents/
    ├── chief_editor/
    │   ├── Dockerfile
    │   ├── main.py           # Hosts the agent, connects to MCP bus
    │   ├── tools.py          # Implements the agent's MCP tools
    │   ├── models/           # Llama-3-70B-Instruct
    │   └── requirements.txt
    ├── scout/
    │   ├── Dockerfile
    │   ├── main.py
    │   ├── tools.py
    │   ├── models/           # Llama-3-8B-Instruct
    │   └── requirements.txt
    ├── fact_checker/
    │   ├── ... (structure repeats for each agent)
    ├── analyst/
    │   ├── ...
    ├── synthesizer/
    │   ├── ...
    ├── critic/
    │   ├── ...
    └── memory/
        ├── Dockerfile
        ├── main.py
        ├── tools.py          # Implements data access tools, connects to DBs
        ├── db_migrations/
        └── requirements.txt
```

This structure organizes the system logically around its core intelligent components, making it easier to develop, test, and scale each agent independently.
