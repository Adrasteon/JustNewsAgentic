# JustNews: Project & Development Plan

This document provides a comprehensive project plan for building the JustNews system as outlined in the `JustNews_Proposal_V2.md`. It details the recommended technology stack, architecture, and a file-by-file breakdown of the project structure, adhering to principles of modularity, robustness, and industry best practices.

## 1. Guiding Principles & Best Practices

The development of JustNews will adhere to the following principles:

-   **Modularity:** The system will be built as a set of independent, containerized microservices. This isolates concerns, simplifies development and testing, and allows for individual components to be scaled or updated without impacting the entire system.
-   **Asynchronicity:** Services will communicate asynchronously via a message queue. This decouples components and creates a resilient, non-blocking architecture that can handle high-volume data flow and long-running tasks efficiently.
-   **Robustness & Reliability:** Each service will have its own error handling, logging, and health checks. A staging environment and a CI/CD pipeline will be used to ensure code quality and prevent regressions.
-   **Transparency & Reproducibility:** All infrastructure will be defined as code (IaC) using Docker. All models, dependencies, and configurations will be explicitly versioned to ensure the entire system is reproducible.
-   **Microsoft Developer Best Practices (Adapted for Python/OSS):**
    -   **Clean Architecture:** Services are self-contained and expose clear API boundaries.
    -   **Strong Typing:** Python type hints will be used throughout the codebase for clarity and static analysis.
    -   **Comprehensive Testing:** A multi-layered testing strategy (unit, integration, E2E) will be implemented.
    -   **Thorough Documentation:** Every module and service will have a `README.md` explaining its purpose, and code will be well-documented.

## 2. Technology Stack

| Component                 | Technology                                       | Justification                                                                                                                            |
| ------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Language**              | Python 3.11+                                     | The industry standard for AI/ML development with an unparalleled ecosystem of libraries (Hugging Face, PyTorch).                         |
| **Service API Framework** | FastAPI                                          | High-performance, asynchronous, and includes automatic data validation and API documentation, making it ideal for building microservice APIs. |
| **Containerization**      | Docker & Docker Compose                          | Essential for creating reproducible environments, simplifying deployment, and managing the multi-service architecture locally.             |
| **Inter-Service Comms**   | RabbitMQ                                         | A mature, robust message broker that enables reliable, asynchronous communication between services, ensuring decoupling and scalability.      |
| **Core Database**         | PostgreSQL 16+                                   | A powerful, reliable relational database for storing core article data and metadata.                                                     |
| **Vector Store**          | `pgvector` (Postgres Extension)                  | Provides efficient similarity search directly within the main database, simplifying the stack for the clustering/matching tasks.           |
| **Graph Database**        | Neo4j                                            | Purpose-built for managing and querying the complex, evolving relationships between stories, making the timeline feature fast and robust. |
| **AI/ML Libraries**       | Hugging Face `transformers`, `sentence-transformers` | Provides easy access to state-of-the-art open-source models and tools for fine-tuning and inference.                                   |
| **CI/CD**                 | GitHub Actions                                   | Tightly integrated with the source code repository for automating testing, linting, and build processes.                                 |

## 3. High-Level Architecture

The system is a set of microservices communicating via a central RabbitMQ message queue.

1.  The **Orchestrator** initiates tasks by publishing messages (e.g., `crawl_request`).
2.  The **Requester** service consumes this message, generates crawl parameters, and publishes them.
3.  The **Executor (Crawl4AI)** consumes the parameters, crawls the web, and places the raw content onto a queue.
4.  The **Validator** consumes the raw content, validates it, and if successful, places the validated article on another queue.
5.  This pattern continues through **Enrichment**, **Synthesis**, and **Publishing**.
6.  The **Critic** services listen for outputs from the main pipeline and run their evaluations asynchronously, posting training data to a dedicated queue or database table.

This event-driven architecture ensures that a failure in one service (e.g., the `TextNeutralizer`) does not bring down the entire pipeline.

## 4. Project Structure & File Breakdown

```
.
├── .github/
│   └── workflows/
│       └── - [ ] ci.yml            # GitHub Actions: runs linting, testing, and docker builds on push/PR
├── .gitignore
├── docker-compose.yml              # Defines and configures all services for local development
├── JustNews_Plan.md                # This project plan
├── JustNews_Proposal_V2.md         # The project proposal
└── services/
    ├── orchestrator/
    │   ├── - [ ] Dockerfile        # Defines the container for the Orchestrator service
    │   ├── - [ ] main.py           # Main entrypoint, starts the service and initiates pipeline tasks
    │   ├── - [ ] pipeline_manager.py # Logic for managing the sequence of tasks in the workflow
    │   ├── - [ ] config.py         # Service configuration (e.g., task schedules, queue names)
    │   ├── - [ ] requirements.txt  # Python dependencies
    │   └── - [ ] README.md         # Service-specific documentation
    ├── requester/
    │   ├── - [ ] Dockerfile
    │   ├── - [ ] main.py           # FastAPI app, exposes an endpoint for generating crawl strategies
    │   ├── - [ ] strategist.py     # Contains the Requester LLM logic for creating CrawlParams
    │   ├── - [ ] models/           # Directory to store the fine-tuned Requester LLM
    │   │   └── - [ ] Llama-3-8B-Instruct/
    │   ├── - [ ] config.py
    │   ├── - [ ] requirements.txt
    │   └── - [ ] README.md
    ├── validator/
    │   ├── - [ ] Dockerfile
    │   ├── - [ ] main.py           # Consumes raw articles, validates them, and publishes valid ones
    │   ├── - [ ] validation.py     # Contains the Validator LLM logic
    │   ├── - [ ] models/
    │   │   └── - [ ] Mistral-7B-Instruct-v0.2/
    │   ├── - [ ] config.py
    │   ├── - [ ] requirements.txt
    │   └── - [ ] README.md
    ├── enrichment/
    │   ├── - [ ] Dockerfile
    │   ├── - [ ] main.py           # Consumes validated articles, enriches them, and saves to DB
    │   ├── - [ ] analysis.py       # Contains the Enrichment LLM logic for scoring bias, etc.
    │   ├── - [ ] db_writer.py      # Handles writing enriched articles to PostgreSQL
    │   ├── - [ ] models/
    │   │   └── - [ ] Mistral-7B-Instruct-v0.2/
    │   ├── - [ ] config.py
    │   ├── - [ ] requirements.txt
    │   └── - [ ] README.md
    ├── synthesizer/
    │   ├── - [ ] Dockerfile
    │   ├── - [ ] main.py           # Consumes clustering tasks, runs the full synthesis pipeline
    │   ├── - [ ] clustering.py     # Story Clustering logic (using SBERT)
    │   ├── - [ ] neutralization.py # Text Neutralizer LLM logic
    │   ├── - [ ] aggregation.py    # Aggregator LLM logic
    │   ├── - [ ] db_interface.py   # Reads from/writes to PostgreSQL and Neo4j
    │   ├── - [ ] models/
    │   │   ├── - [ ] all-mpnet-base-v2/ # SBERT model for embeddings
    │   │   ├── - [ ] Llama-3-8B-Instruct_Neutralizer/ # Fine-tuned Neutralizer
    │   │   └── - [ ] Llama-3-70B-Instruct_Aggregator/ # Fine-tuned Aggregator
    │   ├── - [ ] config.py
    │   ├── - [ ] requirements.txt
    │   └── - [ ] README.md
    ├── publisher/
    │   ├── - [ ] Dockerfile
    │   ├── - [ ] main.py           # Consumes new synthesized articles and triggers publication
    │   ├── - [ ] story_matcher.py  # Story Matching LLM logic (using Graph DB)
    │   ├── - [ ] site_generator.py # Generates static HTML files from templates
    │   ├── - [ ] templates/        # HTML templates for the static site
    │   │   └── - [ ] article.html
    │   ├── - [ ] config.py
    │   ├── - [ ] requirements.txt
    │   └── - [ ] README.md
    └── critics/
        ├── - [ ] Dockerfile        # A single Docker image for all critic models
        ├── - [ ] main.py           # Consumes outputs from the pipeline and routes to the correct critic
        ├── - [ ] coherence_critic.py # Logic for the Cluster Coherence Critic
        ├── - [ ] factual_critic.py # Logic for the Factual Consistency Critic
        ├── - [ ] synthesis_critic.py # Logic for the Synthesis Critic
        ├── - [ ] training_data_writer.py # Writes generated training data to a dedicated DB table
        ├── - [ ] models/           # Directory for all critic models
        │   └── - [ ] Llama-3-8B-Instruct/
        ├── - [ ] config.py
        ├── - [ ] requirements.txt
        └── - [ ] README.md
```

## 5. Model Inventory

| Task                            | Service       | Model(s)                               | Synopsis                                                                    |
| ------------------------------- | ------------- | -------------------------------------- | --------------------------------------------------------------------------- |
| Crawling Strategy Generation    | `requester`   | `Llama-3-8B-Instruct`                  | Generates optimal `CrawlParams` based on high-level goals.                  |
| Initial Content Validation      | `validator`   | `Mistral-7B-Instruct-v0.2`             | Classifies content as "news article" or "other".                            |
| Bias & Sentiment Analysis       | `enrichment`  | `Mistral-7B-Instruct-v0.2`             | Scores validated articles on bias, sentiment, and other linguistic metrics. |
| Article Embedding               | `synthesizer` | `all-mpnet-base-v2` (SBERT)            | Creates numerical vector embeddings of articles for similarity comparison.  |
| Text Neutralization             | `synthesizer` | `Llama-3-8B-Instruct` (fine-tuned)     | Rewrites article text to remove subjective and biased language.             |
| Story Aggregation               | `synthesizer` | `Llama-3-70B-Instruct` (fine-tuned)    | Synthesizes a single, de-biased article from a cluster of sources.          |
| Cluster Coherence Evaluation    | `critics`     | `Llama-3-8B-Instruct`                  | Scores how well a cluster of articles covers a single event.                |
| Factual Consistency Evaluation  | `critics`     | `Llama-3-8B-Instruct`                  | Checks if neutralized text alters or omits facts from the original.         |
| Synthesis Quality Evaluation    | `critics`     | `Llama-3-8B-Instruct`                  | Checks for hallucinations or omissions in the final synthesized article.    |
