# JustNews: A Proposal for a Self-Improving, De-Biased News Aggregation System - V2

## 1. Overview

This document outlines a conceptual architecture for an advanced, multi-phase system named "JustNews." The system is designed to autonomously crawl the web for news, validate the content, analyze it for bias and sentiment, synthesize a neutral "source of truth" for each news story, and publish the results on a public-facing website that tracks how stories evolve over time.

This revised proposal (V2) incorporates a detailed analysis of potential risks and includes advanced strategies for ensuring transparency, performance, and long-term reliability.

The system is built around a core principle of a "virtuous cycle": the data acquisition and synthesis strategy is continuously refined based on automated critique and quality assessment, leading to ever-increasing efficiency and accuracy.

## 2. Core Components

The system is composed of several key modules and components, managed by a central Orchestrator.

1.  **Orchestrator:** The central Python application that manages the entire pipeline, calling each component in sequence and handling the flow of data. It is responsible for model versioning and dependency management.
2.  **Requester LLM ("The Strategist"):** A local, fine-tunable LLM responsible for generating optimal web crawling strategies.
3.  **Crawl4AI Docker/MCP Instance ("The Executor"):** The workhorse of the system that executes crawl commands.
4.  **Validator LLM ("The Initial Filter"):** An efficient LLM that performs the initial check to filter out non-news content.
5.  **Enrichment LLM ("The Analyst"):** A specialized model that scores articles on metrics like sentiment, bias, and persuasiveness.
6.  **Data Persistence Layer (Hybrid Model):**
    *   **PostgreSQL Database ("The Archive"):** The primary store for core article text, metadata, and analysis scores.
    *   **Vector Store (e.g., `pgvector` extension or dedicated DB):** Stores text embeddings from SBERT models for efficient similarity search.
    *   **Graph Database (e.g., Neo4j, ArangoDB) ("The Weaver"):** Models the complex relationships between articles, clusters, and evolving stories to power the timeline feature efficiently.
7.  **Synthesizer Module:**
    *   **Story Clustering LLM:** Groups articles by event, underpinned by the Vector Store.
    *   **Text Neutralizer LLM:** Rewrites articles to remove bias.
    *   **Aggregator LLM:** Synthesizes a single, de-biased article from a cluster.
8.  **Publisher Module:**
    *   **Story Matching LLM:** Links new stories to existing timelines, powered by the Graph Database.
    *   **Static Site Generator:** Creates the final HTML webpages.
9.  **Critic LLMs:** A suite of specialized LLMs for evaluating the operational LLMs to generate training data.

## 3. The End-to-End Workflow

The workflow remains as described previously, operating in a continuous loop of Acquisition, Enrichment, Synthesis, and Publishing. The key refinements are in the learning process and operational safeguards.

## 4. The Automated Learning Loops: Core Principle of Self-Critique

The "Perform -> Critique -> Refine" pattern is central to the system's evolution. The following sections detail the refined training loops for each component.

### 4.1. Training the Story Clustering LLM
*   **Perform & Critique:** As described in V1, using a **"Cluster Coherence Critic" LLM**.
*   **Refine:** The generated positive and negative examples are used to fine-tune the embedding model or a small classification head on top of it.

### 4.2. Training the Text Neutralizer LLM
*   **Perform:** The **Text Neutralizer LLM** rewrites an article.
*   **Critique (Enhanced):**
    1.  **Bias Reduction:** The **Enrichment LLM** confirms a significant reduction in the bias score.
    2.  **Factual Consistency:** The **"Factual Consistency Critic" LLM** is explicitly trained to check for not only contradictions but also the **omission of critical context or qualifying language**.
*   **Refine:** Only pairs that pass this enhanced critique become training data.

### 4.3. Training the Aggregator LLM
*   **Perform & Critique:** As described in V1, using a **"Synthesis Critic" LLM** to check for hallucinations and omissions.
*   **Refine:** Golden examples are used to fine-tune the **Aggregator LLM**.

## 5. Advanced Considerations & Risk Mitigation

This section addresses potential challenges to ensure the system is robust, transparent, and scalable.

### 5.1. Ensuring Genuine Neutrality and Transparency
*   **The "Bias of the Critic" Problem:** The system's concept of "neutrality" is defined by its initial critic models.
*   **Mitigation:** The public-facing website will feature a dedicated **"Methodology"** section. This page will not be an afterthought; it is a core feature. It will detail the definitions of bias used, the architecture of the critic models, and provide open access to the anonymized training data used for the critics themselves. This radical transparency allows users to audit the system's definition of neutrality.

### 5.2. Maintaining Information Fidelity
*   **Problem 1: "Over-Neutralization":** An aggressive neutralizer might strip important nuance or context.
*   **Mitigation:** The **Factual Consistency Critic** is enhanced (as per section 4.2) to be sensitive to changes in nuance, not just bald facts.
*   **Problem 2: "Echo Chamber Reinforcement":** The Requester LLM might learn to only crawl from a narrow set of sources that pass validation easily.
*   **Mitigation:** The Orchestrator will track a **"Source Diversity Score."** If this score drops, the Requester LLM will be penalized for suggesting sources from over-represented domains and rewarded for exploring new, more diverse domains, creating a constant pressure for exploration.

### 5.3. System Performance and Scalability
*   **Problem 1: Computational Cost:** The critique phase for every article is a significant bottleneck.
*   **Mitigation:**
    *   **Asynchronous, Batched Critiques:** The critique process will run as a separate, lower-priority task. The main pipeline is not blocked waiting for critiques.
    *   **Selective Critiquing:** The system will implement a sampling mechanism. For instance, it will critique 100% of articles from new sources but only a 10% random sample from established, high-quality sources.
*   **Problem 2: Database Complexity:** Querying story timelines can become slow.
*   **Mitigation:** The adoption of a **hybrid data persistence layer** (see section 2.6) directly addresses this. A graph database is purpose-built for efficiently querying the complex relationships in a story timeline.

### 5.4. Model Governance and Reliability
*   **Problem 1: "Dependency Hell":** A change in one model can negatively impact another.
*   **Mitigation:**
    *   **Immutable Model Versioning:** Every fine-tuned model is saved as a new, versioned artifact (e.g., `TextNeutralizer-v1.1.0`).
    *   **Staging Environment:** Before a new model version is promoted to the production pipeline, it runs in a "shadow" staging environment. Its performance is compared against the current production model on a live data sample. Promotion is not automatic; it requires passing a battery of regression tests.
*   **Problem 2: Training Data Quality Degradation:** A flawed critic could poison the training data pool.
*   **Mitigation:** A **minimal Human-in-the-Loop (HITL) review dashboard** will be implemented. This is a crucial safety check. It will surface a small, random sample (e.g., <1%) of generated training examples for human spot-checking, allowing for the early detection of systematic errors in the critic models without sacrificing automation.

---

## Appendix A: Recommended Open-Source Models

To ensure full control and transparency, the JustNews project will be built exclusively with local, open-source models.

### 1. For General Purpose & Critic Tasks
*(Requester, Text Neutralizer, Aggregator, All Critic LLMs)*
*   **Models:** Llama 3 Series (e.g., `Llama-3-8B-Instruct`, `Llama-3-70B-Instruct`)
*   **Source:** Meta / Hugging Face Hub.

### 2. For Specialized Analysis & Classification
*(Enrichment LLM, Validator LLM)*
*   **Models:** Mistral Series (e.g., `Mistral-7B-Instruct-v0.2`)
*   **Source:** Mistral AI / Hugging Face Hub.

### 3. For Embedding & Semantic Similarity
*(Underpinning the Story Clustering & Story Matching LLMs)*
*   **Models:** Sentence-BERT (SBERT) models (e.g., `all-mpnet-base-v2`)
*   **Source:** UKPLab / Hugging Face Hub.
