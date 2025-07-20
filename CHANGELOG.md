# Changelog

All notable changes to this project will be documented in this file.



## [0.2.2] - 2025-07-19

### Added
- Database migrations for `training_examples` and `article_vectors` tables in Memory Agent.
- Expanded README with service, database, and migration instructions.

### Improved
- Enhanced Chief Editor Agent: Implemented robust orchestration logic for `request_story_brief` and `publish_story` tools, including workflow stubs and improved logging.

## [0.2.0] - YYYY-MM-DD

### Added
- Implemented the MCP Message Bus.
- Implemented the Memory Agent with PostgreSQL integration.
- Implemented the Scout Agent with web search and crawling capabilities.

### Added
- Initial project scaffolding for all agents as per JustNews_Plan_V3.
- Creation of `JustNews_Proposal_V3.md` and `JustNews_Plan_V3.md`.
- Basic `README.md` and `CHANGELOG.md`.
## [0.3.0] - 2025-07-20

### Added
- Refactored Synthesizer agent to use sentence-transformers for clustering, LLM for neutralization/aggregation, and feedback logging for continual learning.
- Refactored Critic agent to use LLM for critique, feedback logging, and support for continual learning and retraining.
- Refactored Memory agent to implement semantic retrieval with embeddings (sentence-transformers), vector search (pgvector), feedback logging, and retrieval usage tracking for future learning-to-rank.

### Improved
- All agents now support ML-based feedback loops as described in JustNews_Plan_V3.md.
- Documentation and code comments updated to clarify feedback loop and continual learning mechanisms.
