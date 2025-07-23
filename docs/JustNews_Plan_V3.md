## ML & Feedback Loop Implementation (2025-07-20)

The following improvements have been implemented as of July 2025:

- **Synthesizer Agent**: Uses sentence-transformers for clustering, LLM for neutralization/aggregation, and logs feedback for continual learning.
- **Critic Agent**: Uses LLM for critique, logs all feedback and editorial outcomes for retraining and adaptation.
- **Memory Agent**: Provides semantic retrieval with embeddings and vector search, logs all retrievals and downstream outcomes for future learning-to-rank and model improvement.

All feedback is logged to agent-specific files and/or the database, supporting both online and scheduled retraining. See `CHANGELOG.md` and `JustNews_Proposal_V3.md` for technical details.
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
-   **LLM:** `Llama-3-70B-Instruct` (fine-tuned for aggregation and summarization)
-   **Purpose:** Clustering, neutralization, and synthesis using ML models.
-   **ML Implementation:**
    -   **Clustering:** Use sentence-transformers or similar embedding models to generate article embeddings. Apply clustering algorithms (e.g., KMeans, BERTopic, HDBSCAN) to group related articles. Store cluster assignments for traceability.
    -   **Neutralization:** Use the LLM (fine-tuned for style transfer) to rewrite text in a neutral tone. Optionally, use a classifier to score neutrality before and after transformation.
    -   **Aggregation:** Use the LLM (fine-tuned for summarization) to generate cluster summaries. Optionally, use RAG (retrieval-augmented generation) to ground summaries in source content.
-   **Feedback Loop Implementation:**
    -   Log feedback from Critic and Chief Editor on cluster quality and summary neutrality.
    -   Use this feedback as supervised data to fine-tune clustering, neutralization, and summarization models.
    -   Automate periodic retraining or enable online learning for continuous improvement.
-   **MCP Tools Exposed:**
    -   `cluster_articles(article_ids: list[str]) -> list[list[str]]` (uses embedding models and clustering)
    -   `neutralize_text(text: str) -> str` (uses LLM for style transfer/neutralization)
    -   `aggregate_cluster(article_ids: list[str]) -> str` (uses LLM for summarization)



#### 6. Critic Agent
-   **LLM:** `Llama-3-8B-Instruct` (fine-tuned for critique and review)
-   **Purpose:** Automated review and feedback using ML models.
-   **ML Implementation:**
    -   **Critique Synthesis:** Use the LLM (fine-tuned for critique) to evaluate summaries for hallucinations, omissions, and factual consistency. Optionally, use a fact-checking pipeline to cross-reference with source articles.
    -   **Critique Neutrality:** Use the LLM to compare original and neutralized text, flagging factual drift or loss of nuance. Optionally, use a classifier to score neutrality and factuality.
-   **Feedback Loop Implementation:**
    -   Store all critiques and editorial outcomes as labeled data.
    -   Use this data to fine-tune the LLM for more accurate and context-aware critiques.
    -   Enable continual learning or scheduled retraining to adapt to evolving editorial standards.
-   **MCP Tools Exposed:**
    -   `critique_synthesis(summary: str, source_ids: list[str]) -> str` (uses LLM to check for hallucinations and omissions)
    -   `critique_neutrality(original_text: str, neutralized_text: str) -> str` (uses LLM to check for factual drift after neutralization)



#### 7. Memory Agent
-   **LLM:** None (Acts as a tool provider)
-   **Purpose:** Unified data access layer with semantic retrieval.
-   **ML Implementation:**
    -   **Semantic Retrieval:** Use vector search (pgvector, FAISS, or similar) to enable semantic search over articles and metadata. Generate embeddings using a pre-trained or fine-tuned model (e.g., sentence-transformers).
    -   **Retrieval-Augmented Generation (RAG):** Support other agents by providing relevant context passages for generative tasks.
-   **Feedback Loop Implementation:**
    -   Track which retrievals are used in successful fact-checks, syntheses, or critiques.
    -   Use this feedback to improve retrieval ranking (e.g., via learning-to-rank or reinforcement learning).
    -   Log all retrievals and their downstream outcomes for future model training.
-   **MCP Tools Exposed:**
    -   `save_article(content: str, metadata: dict) -> str`: Saves an article, returns its ID.
    -   `get_article(article_id: str) -> dict`: Retrieves an article.
    -   `vector_search_articles(query: str, top_k: int) -> list[dict]`: Finds similar articles (uses vector search/embeddings).
    -   `log_training_example(task: str, input: dict, output: dict, critique: str)`: Logs a new training example for feedback and model improvement.

#### 8. MCP Bus (Orchestration)
-   **LLM:** None (Orchestration logic)
-   **Purpose:** Task routing and agent collaboration.
-   **ML Implementation:**
    -   **Smart Orchestration:** Use reinforcement learning (RL) or multi-agent coordination models to optimize task routing, agent collaboration, and workflow efficiency. RL agents can learn optimal policies for delegating tasks based on agent performance, workload, and outcomes.
-   **Feedback Loop Implementation:**
    -   Monitor workflow outcomes and agent performance metrics (e.g., task completion time, success rate, quality scores).
    -   Use these metrics as rewards/signals for RL or coordination models to adapt orchestration strategies over time.
    -   Log all orchestration decisions and their outcomes for future model training and system improvement.

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

## Updated Functionality: Standalone Execution

Each agent in the JustNews V3 system can now operate independently as a standalone unit while maintaining the ability to communicate via the MCP Bus. This enhancement ensures:

- **Standalone Execution**: Agents can start and function without relying on other agents or services.
- **MCP Bus Integration**: Agents register their tools with the MCP Bus if available, enabling dynamic collaboration.
- **Error Handling**: Robust mechanisms ensure smooth operation in standalone mode if the MCP Bus is unavailable.

### Key Adjustments

- **Agent Independence**: Each agent's `main.py` includes conditional logic for MCP Bus registration and standalone operation.
- **Dual Functionality**: Agents retain their original structure and functionality while supporting standalone execution.
- **Documentation Updates**: Instructions for standalone execution and MCP Bus integration are provided in the `README.md`.

### Benefits

- Enhanced flexibility for development and testing.
- Improved resilience in scenarios where the MCP Bus is temporarily unavailable.
- Simplified deployment for individual agent services.
