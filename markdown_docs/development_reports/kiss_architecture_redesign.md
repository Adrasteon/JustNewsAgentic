---
title: KISS Architecture Redesign — JustNews V4 Simplified
description: Maximum Nucleoid utilization with minimal complexity
status: Design document (migrated from kiss_architecture_redesign.py on 2025‑08‑10)
---

## Proposed Simplified Architecture 🎯

### 3-Agent Architecture (Down from 8 agents)

#### 1. Content Agent (Combines Scout + Memory)
```python
class ContentAgent:
    """Simple web crawling and content extraction"""
    
    FUNCTIONS = [
        "crawl_websites",           # Crawl4AI for content extraction
        "extract_basic_metadata",   # Simple text processing  
        "store_articles",           # File-based storage
        "retrieve_articles"         # Simple search
    ]
    
    MODELS = None  # NO NEURAL MODELS NEEDED
    MEMORY = "50MB"  # Just for crawling
```

#### 2. Reasoning Agent (Nucleoid-Powered Logic Engine)
```python
class ReasoningAgent:
    """ALL validation logic using comprehensive Nucleoid rules"""
    
    FUNCTIONS = [
        "validate_news_quality",     # Rules-based quality scoring
        "assess_factual_content",    # Logic-based fact assessment
        "detect_bias_patterns",      # Rule-based bias detection  
        "evaluate_source_credibility", # Domain reputation rules
        "orchestrate_decisions",     # Multi-criteria decision logic
        "explain_reasoning"          # Clear reasoning chains
    ]
    
    RULES_ENGINE = "Nucleoid with 500+ news domain rules"
    MODELS = None  # PURE SYMBOLIC LOGIC
    MEMORY = "100MB"  # Rule storage and execution
```

#### 3. Processing Agent (Essential Neural Tasks Only)
```python
class ProcessingAgent:
    """Only truly necessary neural processing"""
    
    FUNCTIONS = [
        "extract_entities",          # spaCy NER (essential)
        "semantic_search",           # SentenceTransformers (essential)
        "summarize_content"          # Extractive summarization (simple)
    ]
    
    MODELS = ["spaCy-sm", "SentenceTransformers-mini"]  # Minimal models
    MEMORY = "1.5GB"  # Lightweight models only
```

### Nucleoid Rules Library (Replaces 5+ Neural Models)

```javascript
// === QUALITY ASSESSMENT RULES === (Replaces 8GB Scout Model)
QUALITY_RULES = [
    "if (word_count > 300 && paragraph_count > 3) then structure_score = 0.8",
    "if (sources_cited > 2 && quotes_count > 1) then credibility_bonus = 0.2", 
    "if (spelling_errors / word_count > 0.02) then quality_penalty = 0.3",
    "if (clickbait_words > 3 || excessive_caps > 5) then clickbait_flag = true",
    "if (clickbait_flag == false && structure_score > 0.7) then quality_tier = 'high'"
]

// === FACT VERIFICATION RULES === (Replaces DistilBERT)
FACT_RULES = [
    "if (speculation_words > 3 && fact_statements == 0) then factual_score = 0.2",
    "if (numbers_with_sources > 2) then factual_bonus = 0.3",
    "if (contains_absolute_claims == true && evidence_provided == false) then skepticism_flag = true",
    "if (factual_score > 0.7 && skepticism_flag == false) then fact_rating = 'reliable'"
]

// === BIAS DETECTION RULES === (Replaces RoBERTa)  
BIAS_RULES = [
    "if (emotional_adjectives / total_adjectives > 0.4) then emotional_bias = 0.7",
    "if (one_sided_sources == true && opposing_views == 0) then perspective_bias = 0.6",
    "if (loaded_language_count > 5) then rhetorical_bias = 0.5",
    "if (emotional_bias + perspective_bias + rhetorical_bias > 1.5) then bias_level = 'high'"
]

// === SOURCE CREDIBILITY RULES === (Replaces Complex Neural Assessment)
CREDIBILITY_RULES = [
    "if (domain_age > 1095 && fact_check_record > 100) then established_source = true",
    "if (established_source == true && error_corrections < 5) then tier_1_source = true", 
    "if (domain in reuters_list || domain in ap_list) then tier_1_source = true",
    "if (tier_1_source == true) then credibility_score = 0.9",
    "if (unknown_domain == true && social_media_only == true) then credibility_score = 0.3"
]

// === ORCHESTRATION RULES === (Replaces Chief Editor Neural Model)
ORCHESTRATION_RULES = [
    "if (quality_tier == 'high' && fact_rating == 'reliable' && bias_level != 'high') then auto_approve = true",
    "if (credibility_score < 0.5) then require_human_review = true",
    "if (breaking_news == true && sources_count < 2) then hold_for_confirmation = true",
    "if (controversy_detected == true) then escalate_to_senior_editor = true"
]
```

### Simplification Benefits 📉

#### Resource Reduction
```
BEFORE (Current):
- Agents: 8 complex agents
- GPU Memory: 18GB+ 
- Models: 15+ neural models
- Lines of Code: ~8,000 lines
- Complexity: High
- Explainability: Low (black box neural models)

AFTER (Simplified):  
- Agents: 3 focused agents
- GPU Memory: 1.5GB (91% reduction)
- Models: 2 lightweight models  
- Lines of Code: ~2,000 lines (75% reduction)
- Complexity: Low
- Explainability: High (clear rule chains)
```

#### Performance Improvement
```
Neural Model Inference: 50-200ms per article
Nucleoid Rules: 0.1-1ms per article (100x faster)

Memory Usage: 18GB → 1.5GB (91% reduction)
Startup Time: 3-5 minutes → 10-30 seconds
```

#### Maintainability Improvement
```
Current: Complex neural models, hard to debug, black box decisions
Simplified: Clear rules, easy to modify, transparent logic
```

### Implementation Phases 🗓️

1) Consolidate Duplicate Functions
- Merge entity extraction into single Processing Agent
- Eliminate duplicate DialoGPT models
- Centralize MCP bus communication

2) Neural → Rules Migration
- Replace Scout's 8GB model with quality rules
- Replace Fact Checker's DistilBERT with logic rules  
- Replace bias detection with pattern rules

3) Agent Consolidation
- Merge Scout + Memory → Content Agent
- Merge Analyst + Critic + Synthesizer + Chief Editor → Enhanced Reasoning Agent
- Keep Processing Agent for essential neural tasks

4) Advanced Nucleoid Features
- Implement complex orchestration rules
- Add temporal reasoning capabilities  
- Create dynamic rule learning system

### Risk Mitigation ⚠️

- Rules might be less accurate than neural models → For initial filtering, explicit rules are often more reliable; neural models can be kept for edge tasks in Processing Agent.
- Loss of nuance → Use Processing Agent for entity and semantic tasks; keep logic for decisions.
- Time to author rules → Rules are faster to iterate than retraining models and are transparent.
