# JustNews V4 Agent Endpoint Reference & Pipeline Flowchart

## Overview
This document details the endpoints for each agent in the JustNews V4 system, explains their use, highlights overlaps/duplication, and shows how endpoints complement each other. A flowchart visualizes agent interactions and pipeline usage.

---

## Agent Endpoint Table

| Agent         | Endpoint(s)                       | Purpose/Use |
|---------------|-----------------------------------|-------------|
| **Scout**     | `/search_related_articles`, `/discover_sources`, `/crawl_url`, `/deep_crawl_site`, `/enhanced_deep_crawl_site`, `/intelligent_source_discovery`, `/intelligent_content_crawl`, `/intelligent_batch_analysis`, `/enhanced_newsreader_crawl`, `/production_crawl_ultra_fast`, `/get_production_crawler_info`, `/health`, `/ready`, `/log_feedback` | Source discovery, crawling, batch analysis, feedback, health |
| **Analyst**   | `/identify_entities`, `/analyze_text_statistics`, `/extract_key_metrics`, `/analyze_content_trends`, `/health`, `/ready`, `/log_feedback` | Entity/statistics extraction, trend analysis, feedback, health |
| **Fact Checker** | `/verify_facts`, `/validate_sources`, `/validate_is_news`, `/validate_is_news_gpu`, `/verify_claims_gpu`, `/health`, `/ready` | Fact/claim verification, source validation, health |
| **Synthesizer** | `/cluster_articles`, `/neutralize_text`, `/aggregate_cluster`, `/synthesize_news_articles_gpu`, `/get_synthesizer_performance`, `/health`, `/ready`, `/log_feedback` | Clustering, synthesis, neutralization, performance, health |
| **Critic**    | `/critique_synthesis`, `/critique_neutrality`, `/critique_content_gpu`, `/get_critic_performance`, `/health`, `/ready`, `/log_feedback` | Synthesis critique, neutrality, performance, health |
| **Chief Editor** | `/request_story_brief`, `/publish_story`, `/health`, `/ready` | Story orchestration, publishing, health |
| **Memory**    | `/save_article`, `/vector_search_articles`, `/log_training_example`, `/get_embedding_model`, `/log_feedback`, `/health`, `/ready` | Storage, vector search, training, feedback, health |
| **Reasoning** | `/add_fact`, `/validate_fact`, `/contradiction_check`, `/explain_decision`, `/health`, `/ready` | Symbolic logic, fact validation, contradiction, explainability, health |
| **Dashboard** | `/get_status`, `/send_command`, `/receive_logs`, `/health` | Agent status, control, logging, health |
| **Balancer**  | `/web_search_balance`, `/register`, `/balance`, `/health`, `/status`, `/resource_status` | Batch delegation, dual analysis, registration, resource/status, health |

---

## Endpoint Overlap & Duplication
- **Health/Ready Endpoints:** All agents expose `/health` and `/ready` for monitoring and orchestration.
- **Feedback/Logging:** Many agents have `/log_feedback` for audit and transparency.
- **Batch/Analysis:** Scout, Analyst, Synthesizer, Critic, and Balancer all support batch analysis endpoints, sometimes with different names.
- **Source Discovery/Crawling:** Scout and Newsreader overlap in content extraction/crawling, but Scout is more advanced for source intelligence.
- **Fact Validation:** Fact Checker and Reasoning both validate facts, but Reasoning uses symbolic logic while Fact Checker uses ML models.

## Endpoint Complementarity
- **Scout → Balancer:** Scout's source discovery feeds into Balancer's dual analysis pipeline.
- **Analyst → Fact Checker:** Analyst extracts entities/statistics, Fact Checker verifies claims.
- **Synthesizer → Critic:** Synthesizer aggregates and neutralizes, Critic reviews for quality and neutrality.
- **Chief Editor → Memory:** Chief Editor orchestrates stories, Memory stores and enables vector search.
- **Dashboard:** Orchestrates, monitors, and controls all agents via status and command endpoints.
- **Reasoning:** Provides explainability and contradiction checks for outputs from other agents.

---

## Pipeline Flowchart

Below is a flowchart showing how agents and their endpoints interact in typical news analysis pipelines. The flowchart is provided as a JPEG image for easy reference.

![JustNews V4 Agent Pipeline Flowchart](markdown_docs/agent_documentation/AGENT_PIPELINE_FLOWCHART.jpeg)

---

## Notes
- For endpoint details, see each agent's documentation in `markdown_docs/agent_documentation/`.
- The flowchart illustrates both sequential and parallel agent interactions, with feedback loops for continuous learning and audit.
- Overlapping endpoints are color-coded in the flowchart for clarity.

---

**Document generated: August 15, 2025**
