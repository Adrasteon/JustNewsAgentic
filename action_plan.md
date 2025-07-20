# Action Plan: Agent Improvements for JustNews V3

This action plan details the steps required to address errors, omissions, stubs, and areas for improvement identified in the current codebase for each agent. The goal is to move all agents toward robust, production-ready, ML/LLM-powered, feedback-driven implementations as described in the V3 Plan.

---

## 1. Chief Editor Agent
- **Replace stubs with real MCP bus integration:**
  - Implement actual HTTP calls to the MCP bus for `request_story_brief` and `publish_story`.
  - Handle responses and errors from the bus.
- **Add feedback logging:**
  - Log all editorial decisions and workflow outcomes for continual learning.
- **Enhance orchestration logic:**
  - Support dynamic task delegation and monitoring of agent responses.
- **Testing:**
  - Add unit and integration tests for orchestration flows.

---

## 2. Scout Agent
- **Replace stubs with real implementations:**
  - Integrate a real web search API (e.g., Google/Bing Custom Search, SerpAPI) for `discover_sources`.
  - Implement robust web crawling and content extraction for `crawl_url` and `deep_crawl_site`.
- **Add error handling and feedback logging:**
  - Log failed searches/crawls and user feedback for continual improvement.
- **Support extraction prompts:**
  - Allow custom extraction prompts to guide content extraction.
- **Testing:**
  - Add tests for search, crawl, and extraction logic.

---

## 3. Fact-Checker Agent
- **Replace rule-based logic with ML/LLM:**
  - Use an LLM or claim verification model for `validate_is_news` and `verify_claims`.
  - Integrate with external fact-checking APIs if available.
- **Add feedback logging:**
  - Log fact-check outcomes and user/editor feedback for retraining.
- **Testing:**
  - Add tests for claim validation and verification.

---

## 4. Analyst Agent
- **Replace rule-based logic with ML/LLM:**
  - Use LLM or ML models for `score_bias`, `score_sentiment`, and `identify_entities`.
  - Integrate with NER and sentiment analysis libraries (spaCy, transformers, etc.).
- **Add feedback logging:**
  - Log analysis results and feedback for model improvement.
- **Testing:**
  - Add tests for bias, sentiment, and entity recognition.

---

## 5. Synthesizer Agent
- **Enhance clustering and aggregation:**
  - Add error handling and validation for clustering and LLM calls.
  - Support additional clustering algorithms (BERTopic, HDBSCAN).
- **Ensure feedback loop is used in retraining:**
  - Automate periodic retraining using logged feedback.
- **Testing:**
  - Add tests for clustering, neutralization, and aggregation.

---

## 6. Critic Agent
- **Enhance critique logic:**
  - Add error handling for LLM pipeline.
  - Integrate optional fact-checking pipeline for cross-referencing.
- **Ensure feedback loop is used in retraining:**
  - Automate periodic retraining using logged feedback.
- **Testing:**
  - Add tests for critique synthesis and neutrality.

---

## 7. Memory Agent
- **Clarify tool interface:**
  - Move or mirror key tool interfaces from `main.py` to `tools.py` for clarity and maintainability.
- **Enhance error handling:**
  - Add robust error handling for DB and embedding/model calls.
- **Ensure feedback loop is used for learning-to-rank:**
  - Use logged retrievals and outcomes to improve ranking models.
- **Testing:**
  - Add tests for semantic retrieval, vector search, and feedback logging.

---

## 8. General/All Agents
- **Documentation:**
  - Update docstrings and README sections for all new/changed logic.
- **Feedback Loop:**
  - Standardize feedback logging format and location across agents.
  - Document retraining and continual learning procedures.
- **CI/CD:**
  - Add/expand tests to cover new ML/LLM logic and feedback mechanisms.

---

## 9. Timeline & Milestones
1. **Week 1:** Replace stubs/rule-based logic in Scout, Fact-Checker, Analyst. Add feedback logging to all agents.
2. **Week 2:** Implement real MCP bus integration for Chief Editor. Enhance orchestration and error handling.
3. **Week 3:** Expand clustering/aggregation in Synthesizer. Add fact-checking pipeline to Critic. Move tool interfaces in Memory.
4. **Week 4:** Standardize feedback loop, automate retraining, finalize documentation, and expand tests.

---

## 10. Success Criteria
- All agents use ML/LLM-based logic for their core tools.
- All feedback is logged and used for continual learning.
- All stubs and rule-based placeholders are replaced.
- Documentation and tests are up to date.

---

*For details, see the latest `CHANGELOG.md`, `JustNews_Plan_V3.md`, and `JustNews_Proposal_V3.md`.*
