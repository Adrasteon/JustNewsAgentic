# JustNews V3: An Agentic, MCP-First News Analysis Ecosystem

## 1. Executive Summary

This proposal (V3) reframes the JustNews system from a linear pipeline of microservices into a dynamic, collaborative ecosystem of specialized AI agents. The central architectural principle is the **Model Context Protocol (MCP)**, which allows these agents to communicate, delegate tasks, and access data sources as if they were native tools.

This agentic, MCP-first approach fundamentally simplifies the system's design while dramatically increasing its flexibility, intelligence, and capacity for emergent, self-improving behavior. It moves beyond a pre-defined workflow and towards a goal-oriented system where agents collaborate to produce clear, factually-grounded, and bias-neutral news analysis.

## 2. The Core Shift: From Pipeline to Agent Ecosystem

The V2 proposal described a system of services connected by message queues. V3 replaces this with a central **MCP Message Bus** (implemented over a technology like RabbitMQ or NATS) and a collection of **Agents**.

Each "service" from the V2 plan is now an **Agent**: a self-contained unit hosting an LLM and a specific set of skills. Crucially, each agent exposes its capabilities (e.g., `crawl_website`, `analyze_bias`, `cluster_articles`) as tools on the MCP bus. This means any agent can discover and invoke the capabilities of any other agent.

| V2 Component | V3 Agent Counterpart | Key MCP-Exposed Tools |
| :--- | :--- | :--- |
| Orchestrator | **Chief Editor Agent** | `request_story_brief`, `publish_story` |
| Requester | **Scout Agent** | `discover_sources`, `crawl_url`, `deep_crawl_site` |
| Validator | **Fact-Checker Agent** | `validate_is_news`, `verify_claims` |
| Enrichment | **Analyst Agent** | `score_bias`, `score_sentiment`, `identify_entities` |
| Synthesizer | **Synthesizer Agent** | `cluster_articles`, `neutralize_text`, `aggregate_cluster` |
| Publisher | **Librarian Agent** | `find_related_story`, `update_story_timeline` |
| Critic | **Critic Agent** | `critique_neutrality`, `critique_synthesis` |
| Data Layer | **Memory Agent** | `save_article`, `find_article_by_id`, `vector_search_articles`, `log_training_example` |

## 3. The Agentic Workflow: A Dynamic Conversation

The workflow is no longer a rigid, linear sequence. It's a dynamic, goal-driven conversation between agents, orchestrated by the **Chief Editor**.

**Example Workflow: "Cover the latest on the Artemis program"**

1.  **Initiation:** A human operator (or a schedule) gives the **Chief Editor Agent** a high-level brief: `"Generate a story about the latest developments in NASA's Artemis program."`

2.  **Delegation & Discovery:**
    *   The **Chief Editor** broadcasts a task: `"I need sources for a story on the 'Artemis program'."`
    *   The **Scout Agent** picks this up. It uses its `discover_sources` tool (which might internally use a web search tool like the one I have) to find relevant sites (e.g., nasa.gov, space.com). It then uses its `crawl_url` tool to fetch the raw content.

3.  **Collaborative Processing:**
    *   As the **Scout Agent** finds articles, it publishes them to the bus. For each article, it calls the **Fact-Checker Agent**: `"@FactChecker, is this a valid news article?"`
    *   The **Fact-Checker** validates the content and, if it passes, calls the **Analyst Agent**: `"@Analyst, score this article for bias."`
    *   Simultaneously, the **Fact-Checker** calls the **Memory Agent**: `"@Memory, save this validated article and its bias score."`

4.  **Synthesis & Review:**
    *   The **Chief Editor**, monitoring the bus, sees enough high-quality, low-bias articles have been collected. It calls the **Synthesizer Agent**: `"@Synthesizer, cluster these articles and produce a neutral summary."`
    *   The **Synthesizer** performs its internal clustering, neutralization, and aggregation, then presents the draft summary to the **Chief Editor**.

    *   The **Chief Editor** then invokes the **Critic Agent**: `"@Critic, review this summary for factual consistency against the source articles."` The Critic has access to the Memory agent to retrieve the original sources. The Critic agent uses an LLM for critique and logs all feedback for continual learning and retraining.

5.  **Learning & Archiving:**
    *   The **Critic's** feedback is used by the **Chief Editor** to decide whether to request a revision from the **Synthesizer** or approve the story.
    *   This feedback loop (summary + critique) is automatically logged by the **Memory Agent** as a new training example. The Memory agent now supports semantic retrieval with embeddings, vector search, and logs all retrievals and outcomes for future learning-to-rank and model improvement.
    *   Once approved, the **Chief Editor** calls the **Librarian Agent** to link the new story to any existing timelines about the Artemis program and publish the final output.

## 4. Key Advantages of the V3 Architecture

*   **Reduced Complexity:** The central logic is now embedded in the agents' reasoning, not in complex, hard-coded pipeline managers and message queue routing. Adding a new capability is as simple as adding a new agent with new tools to the bus.
*   **Greater Flexibility & Emergent Behavior:** The system can adapt to unforeseen situations. If the **Scout Agent** finds a source in a new language, it could learn to invoke a (hypothetical) **Translator Agent** before passing the content to the **Fact-Checker**. This behavior doesn't need to be pre-programmed.
*   **Active, Continuous Learning:** The "virtuous cycle" is no longer a batched, offline process. It's continuous and online. The **Critic Agent** provides real-time feedback, and other agents can be fine-tuned on the fly based on this stream of high-quality training data.
*   **Enhanced Accuracy & Nuance:** By turning data access into a tool (`@Memory, find...`), agents like the **Critic** can actively pull the specific information they need to make more accurate judgments, rather than relying on data pushed to them. The **Fact-Checker** could use the **Memory Agent** to see if a new article contradicts previously stored facts.
*   **Massive Scalability:** The agent-based model is inherently parallelizable. You can run multiple instances of any agent to handle increased load, and the MCP bus will route tasks accordingly.

## 5. Next Steps

The next step is to create a new `JustNews_Plan_V3.md` that reflects this agentic architecture. This will involve defining the specific tools for each agent, the core message schemas for the MCP bus, and a revised project structure that organizes the code by agents rather than by pipeline stages.

This V3 proposal represents a paradigm shift. It moves JustNews from a well-engineered but rigid data processing system to a truly intelligent, adaptable, and self-improving news analysis ecosystem, ready to grow into a substantial and trustworthy information source.

## ML & Feedback Loop Implementation (2025-07-20)

- **Synthesizer Agent**: Now uses sentence-transformers for clustering, LLM for neutralization/aggregation, and logs feedback for continual learning.
- **Critic Agent**: Now uses LLM for critique, logs all feedback and editorial outcomes for retraining and adaptation.
- **Memory Agent**: Now provides semantic retrieval with embeddings and vector search, logs all retrievals and downstream outcomes for future learning-to-rank and model improvement.

All feedback is logged to agent-specific files and/or the database, supporting both online and scheduled retraining. See `CHANGELOG.md` and `JustNews_Plan_V3.md` for technical details.

## Proposal Update: Standalone Functionality

To enhance the flexibility and resilience of the JustNews V3 system, each agent now supports standalone execution. This update ensures:

- **Independent Operation**: Agents can start and function independently without relying on other agents or services.
- **MCP Bus Integration**: Agents register their tools with the MCP Bus upon startup, enabling inter-agent communication.
- **Fallback Mechanisms**: Agents operate in standalone mode if the MCP Bus is unavailable, ensuring uninterrupted functionality.

### Implementation Details

- **Agent Code Updates**: Conditional logic added to `main.py` files for MCP Bus registration and standalone operation.
- **Documentation Enhancements**: Standalone execution instructions included in the `README.md`.
- **Error Handling**: Robust mechanisms to handle MCP Bus unavailability.

### Advantages

- Increased flexibility for development and deployment.
- Improved system resilience and fault tolerance.
- Simplified testing and debugging for individual agents.
