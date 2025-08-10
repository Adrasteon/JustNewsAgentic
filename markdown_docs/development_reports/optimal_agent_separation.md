---
title: Optimal Agent Separation Architecture
description: Clear functional boundaries leveraging each agent's strengths
status: Design document (migrated from optimal_agent_separation.py on 2025‑08‑10)
---

## Overview

Defines optimal responsibilities between a neural Fact Checker and a symbolic Reasoning Agent, and how they integrate in a pipeline.

## Fact Checker — Neural Pattern Recognition & Assessment

```python
class FactCheckerOptimalScope:
    """
    Focus: Neural network-based assessment and pattern recognition
    Strength: Large-scale training data, probabilistic scoring
    """
    
    CORE_FUNCTIONS = [
        "neural_fact_plausibility_assessment",    # DistilBERT factual/questionable classification
        "source_credibility_scoring",             # RoBERTa-based domain reliability assessment  
        "semantic_evidence_retrieval",            # SentenceTransformers similarity search
        "automated_claim_extraction",             # spaCy NER for verifiable claims identification
    ]
```

## Reasoning Agent — Logic Validation & Rule-Based Analysis

```python
class ReasoningAgentOptimalScope:
    """
    Focus: Symbolic logic, rule-based validation, and explainable reasoning
    Strength: Precise logical operations, rule consistency, explainability
    """
    
    CORE_FUNCTIONS = [
        "logical_consistency_validation",         # Nucleoid contradiction detection
        "rule_based_fact_checking",              # Domain-specific news rules
        "multi_agent_consensus_orchestration",   # Coordinate between agents
        "temporal_reasoning_validation",          # Time-based logic rules
        "explainable_reasoning_chains",          # Clear decision explanations
    ]
```

## Integrated Pipeline — Optimal Cooperation

```python
class OptimalNewsValidationPipeline:
    """
    Stage 1: Fact Checker (Neural Assessment)
    Stage 2: Reasoning Agent (Logic Validation) 
    Stage 3: Integrated Decision
    """
    
    def comprehensive_validation(self, article_text: str, metadata: dict):
        # Stage 1: Neural Assessment
        neural_results = fact_checker.process_article(article_text, metadata["source_url"])
        
        # Stage 2: Logic Validation  
        logic_results = reasoning_agent.validate_with_logic(neural_results, metadata)
        
        # Stage 3: Integrated Decision
        final_decision = {
            "overall_confidence": self._calculate_consensus_confidence(neural_results, logic_results),
            "verification_status": self._determine_verification_status(logic_results),
            "explanation": logic_results["explainability"]["reasoning_steps"],
            "neural_assessment": neural_results["neural_assessment"],
            "logical_validation": logic_results["logical_validation"],
            "processing_summary": {
                "fact_checker_confidence": neural_results["processing_metadata"]["confidence"],
                "reasoning_validation": logic_results["orchestration_decision"]["consensus_confidence"],
                "final_recommendation": logic_results["orchestration_decision"]["recommended_action"]
            }
        }
        
        return final_decision
```

## Rationale

- Eliminates overlap: neural pattern recognition vs. deterministic logic validation
- Maximizes Nucleoid potential for explainable rules and orchestration
- Provides clear integration points and responsibility boundaries
