# JustNews: A Proposal for a Self-Improving, De-Biased News Aggregation System

## 1. Overview

This document outlines a conceptual architecture for an advanced, multi-phase system named "JustNews." The system is designed to autonomously crawl the web for news, validate the content, analyze it for bias and sentiment, synthesize a neutral "source of truth" for each news story, and publish the results on a public-facing website that tracks how stories evolve over time.

The system is built around a core principle of a "virtuous cycle": the data acquisition and synthesis strategy is continuously refined based on automated critique and quality assessment, leading to ever-increasing efficiency and accuracy.

## 2. Core Components

The system is composed of several key modules and components, managed by a central Orchestrator.

1.  **Orchestrator:** The central Python application that manages the entire pipeline, calling each component in sequence and handling the flow of data.
2.  **Requester LLM ("The Strategist"):** A local, fine-tunable Large Language Model (LLM) responsible for generating optimal web crawling strategies based on high-level goals and feedback from previous runs.
3.  **Crawl4AI Docker/MCP Instance ("The Executor"):** The workhorse of the system. It receives detailed crawl commands from the Orchestrator and executes them, fetching web content efficiently.
4.  **Validator LLM ("The Critic"):** An LLM tasked with a single, critical job: determining if a piece of crawled content is a valid news article, filtering out opinion pieces, forum discussions, and other non-news content.
5.  **Enrichment LLM ("The Analyst"):** An LLM (or a suite of specialized models) that performs detailed linguistic analysis on validated articles, scoring them on metrics like sentiment, bias, and persuasiveness.
6.  **PostgreSQL Database ("The Archive"):** A robust relational database that serves as the system's long-term memory. It stores all crawled articles, their metadata, analysis scores, and the relationships between evolving stories.
7.  **Synthesizer Module:** A collection of LLMs that work together to process the collected data:
    *   **Story Clustering LLM:** Groups individual articles from various sources by the unique real-world event they cover.
    *   **Text Neutralizer LLM:** Rewrites article text to remove subjective, emotional, and biased language while preserving factual information.
    *   **Aggregator LLM:** Synthesizes a single, de-biased, and comprehensive news article from a cluster of "cleaned" source articles, using their bias scores to weigh the reliability of the information.
8.  **Publisher Module:** The final output stage of the system:
    *   **Story Matching LLM:** Intelligently links new stories to existing ones in the database to build historical timelines.
    *   **Static Site Generator:** Creates the final HTML webpages for publication.
9.  **Critic LLMs:** A suite of specialized LLMs whose sole purpose is to evaluate the output of the operational LLMs, providing the quantitative and qualitative feedback needed for automated fine-tuning.

## 3. The End-to-End Workflow

The system operates in a continuous, multi-phase loop.

### Phase 1: Acquisition & Validation (Daily Operation)
*   **Step 1: Strategy & Request:** The Orchestrator provides a high-level goal (e.g., "Find today's technology news") to the **Requester LLM**. The LLM generates a detailed `CrawlParams` JSON object.
*   **Step 2: Execution:** The Orchestrator sends the `CrawlParams` to the **Crawl4AI** instance.
*   **Step 3: Validation:** The Orchestrator passes the content to the **Validator LLM**. Invalid content is discarded and used as a negative feedback signal. Valid content proceeds.

### Phase 2: Enrichment & Storage (Daily Operation)
*   **Step 4: Analysis:** The Orchestrator sends the validated article to the **Enrichment LLM** for scoring.
*   **Step 5: Archiving:** The Orchestrator saves the article, metadata, and scores into the **PostgreSQL Database**.

### Phase 3: Synthesis & Aggregation (Processing the Day's Articles)
*   **Step 6: Retrieval & Clustering:** The Orchestrator retrieves new articles and uses the **Story Clustering LLM** to group them.
*   **Step 7: Neutralize & Synthesize:** For each cluster, the **Text Neutralizer LLM** cleans each article, and the **Aggregator LLM** synthesizes a single, de-biased summary.

### Phase 4: Publishing & Story Evolution (The Final Output Stage)
*   **Step 8: Story Matching:** The **Story Matching LLM** links the new synthesized article to an existing `story_id` or assigns a new one.
*   **Step 9: Webpage Generation:** The **Publisher Module** generates the final HTML page, including the synthesized article, source links with bias scores, and a story timeline.
*   **Step 10: Deployment:** The generated files are deployed to a web server.

## 4. The Automated Learning Loops: Core Principle of Self-Critique

The system's intelligence comes from its ability to learn through a "Perform -> Critique -> Refine" pattern. The success and failure signals from the **Validation** phase are fed back to the **Requester LLM**, teaching it to generate better crawling strategies. This core principle is extended to the synthesis LLMs to create a fully self-improving system.

### 4.1. Training the Story Clustering LLM
*   **Perform:** The **Story Clustering LLM** groups articles into clusters.
*   **Critique:** A dedicated **"Cluster Coherence Critic" LLM** assesses each cluster. It receives the titles and summaries of all articles in a cluster and returns a coherence score (0.0 to 1.0) answering: "Do all of these articles describe the same core event?"
*   **Refine:**
    *   **High-Coherence Clusters (>0.95):** These are saved as high-quality positive training examples.
    *   **Low-Coherence Clusters (<0.5):** These are flagged as failures. The system automatically generates negative training examples by creating pairs of unrelated articles from the failed cluster.
    *   The accumulated examples are periodically used to fine-tune the **Story Clustering LLM**.

### 4.2. Training the Text Neutralizer LLM
*   **Perform:** The **Text Neutralizer LLM** rewrites a biased article.
*   **Critique (Two-Part):**
    1.  **Bias Reduction:** The **Enrichment LLM** scores both the original and neutralized text. A successful run must show a significant reduction in the bias score.
    2.  **Factual Consistency:** A **"Factual Consistency Critic" LLM** compares the original and neutralized texts to answer: "Does the neutralized text contradict, omit, or invent any factual claims present in the original?"
*   **Refine:** If a neutralization attempt reduces bias *and* passes the factual consistency check, the `(original_article, neutralized_text)` pair is stored as a high-quality training example to fine-tune the **Text Neutralizer LLM**.

### 4.3. Training the Aggregator LLM
*   **Perform:** The **Aggregator LLM** synthesizes a single article from a cluster of neutralized sources.
*   **Critique:** A **"Synthesis Critic" LLM** evaluates the output against the source articles. It checks for:
    1.  **Hallucination:** Is there any information in the summary that is not supported by the sources?
    2.  **Omission:** Are any critical facts, present in the majority of sources, missing from the summary?
*   **Refine:** A synthesized article that passes both checks is stored as a golden `(source_cluster_texts, synthesized_article)` training example. These examples are used to fine-tune the **Aggregator LLM** for accuracy and faithfulness.

---

## Appendix A: Recommended Open-Source Models

To ensure full control, transparency, and avoid potential bias from third-party systems, the JustNews project will be built exclusively with local, open-source models. Corporate or cloud-based LLMs will not be used.

### 1. For General Purpose & Critic Tasks
*(Requester, Text Neutralizer, Aggregator, All Critic LLMs)*

*   **Models:** Llama 3 Series (e.g., `Llama-3-8B-Instruct`, `Llama-3-70B-Instruct`)
*   **Why:** These models offer state-of-the-art performance in instruction-following, reasoning, and text generation. Their quality makes them ideal for the core synthesis tasks and for the nuanced work of critiquing other models' outputs. They can be fine-tuned effectively with the data generated from the learning loops.
*   **Source:** Meta / Hugging Face Hub.

### 2. For Specialized Analysis & Classification
*(Enrichment LLM, Validator LLM)*

*   **Models:** Mistral Series (e.g., `Mistral-7B-Instruct-v0.2`, `Mixtral-8x7B`)
*   **Why:** These models provide exceptional performance for their size, making them highly efficient for focused tasks like classification (e.g., "is this a news article?") and scoring (e.g., calculating a bias metric). Their speed is advantageous for the high-volume enrichment phase.
*   **Source:** Mistral AI / Hugging Face Hub.

### 3. For Embedding & Semantic Similarity
*(Underpinning the Story Clustering & Story Matching LLMs)*

*   **Models:** Sentence-BERT (SBERT) models (e.g., `all-mpnet-base-v2`)
*   **Why:** These are not general-purpose LLMs but are highly optimized for a specific, critical task: creating numerical representations (embeddings) of text. They are the industry standard for calculating semantic similarity, which is perfect for determining if two articles are about the same topic. Using a specialized model here is far more efficient than using a full LLM.
*   **Source:** UKPLab / Hugging Face Hub.