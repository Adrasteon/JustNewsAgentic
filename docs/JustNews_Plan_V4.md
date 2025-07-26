# JustNews V4: Migration and Implementation Plan

## Overview

This document provides the detailed engineering plan for migrating JustNewsAgentic from the current V3 architecture to the new V4 Hybrid Architecture. The migration is designed to be zero-downtime, risk-mitigated, and value-delivering from the first phase.

## 1. Migration Strategy

### Core Principles
- **Zero Downtime**: System remains operational throughout migration
- **Risk Mitigation**: Each phase can be rolled back independently
- **Immediate Value**: Benefits realized from first phase completion
- **Data Preservation**: All existing feedback logs and training data retained
- **Backward Compatibility**: Existing APIs and interfaces maintained

### Three-Phase Approach

```
Phase 1: Foundation Migration (Weeks 1-2)
├─ Replace corrupted local models with Docker Model Runner
├─ Implement OpenAI-compatible API integration
├─ Enhanced feedback collection infrastructure
└─ Validate system performance and reliability

Phase 2: Training Pipeline Development (Weeks 3-6)
├─ Build custom model training infrastructure
├─ Implement feedback-to-training-data pipeline
├─ Create A/B testing framework
└─ Train first generation custom models

Phase 3: Progressive Model Replacement (Months 2-6)
├─ Deploy custom models with fallback mechanisms
├─ Gradual replacement based on performance metrics
├─ Achieve complete AI independence
└─ Optimize and specialize for news analysis
```

## 2. Phase 1: Foundation Migration (Weeks 1-2)

### Objectives
- Eliminate current model corruption issues
- Establish reliable Docker Model Runner foundation
- Enhance feedback collection for future training
- Validate performance meets or exceeds current system

### Technical Tasks

#### 2.1 Docker Model Runner Setup

**Prerequisites Check:**
```powershell
# Verify Docker Desktop version (4.41+ required for Windows GPU)
docker --version

# Check GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Enable Docker Model Runner
# Docker Desktop → Settings → Features in Development → Enable "Docker Model Runner"
```

**Model Runner Configuration:**
```yaml
# docker-compose.override.yml additions
version: '3.8'
services:
  # Docker Model Runner service
  model-runner:
    image: docker/model-runner:latest
    environment:
      - ENABLE_GPU=true
      - TCP_PORT=12434
      - HOST_ACCESS=true
    ports:
      - "12434:12434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Updated analyst service
  analyst:
    environment:
      # New hybrid configuration
      - INFERENCE_MODE=docker_model_runner
      - MODEL_ENDPOINT=http://model-runner.docker.internal/engines/llama.cpp/v1/
      - PRIMARY_MODEL=ai/mistral:7b-instruct-v0.3
      - FALLBACK_MODEL=ai/llama3.2:7b-instruct
      - BACKUP_MODEL=ai/gemma3:7b-instruct
      
      # Enhanced feedback collection
      - FEEDBACK_COLLECTION=enhanced
      - FEEDBACK_DATABASE=enabled
      - PERFORMANCE_METRICS=enabled
      
      # Legacy support during migration
      - LEGACY_MODEL_FALLBACK=enabled
      - MIGRATION_MODE=phase1
    depends_on:
      - model-runner
      - db
```

#### 2.2 Enhanced Agent Tools

**Updated `agents/analyst/tools.py`:**

```python
# New hybrid inference client
class HybridInferenceClient:
    """Manages inference across Docker Model Runner and custom models."""
    
    def __init__(self):
        self.docker_endpoint = os.environ.get("MODEL_ENDPOINT", 
            "http://model-runner.docker.internal/engines/llama.cpp/v1/")
        self.primary_model = os.environ.get("PRIMARY_MODEL", "ai/mistral:7b-instruct-v0.3")
        self.fallback_model = os.environ.get("FALLBACK_MODEL", "ai/llama3.2:7b-instruct")
        self.backup_model = os.environ.get("BACKUP_MODEL", "ai/gemma3:7b-instruct")
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url=self.docker_endpoint,
            api_key="not-needed"  # Docker Model Runner doesn't require API key
        )
    
    def generate_with_fallback(self, prompt: str, model_preference: str = None) -> str:
        """Generate text with automatic fallback to backup models."""
        models_to_try = [
            model_preference or self.primary_model,
            self.fallback_model,
            self.backup_model
        ]
        
        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.1
                )
                
                # Log successful inference
                self.log_inference_success(model, prompt[:100])
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                self.log_inference_failure(model, str(e))
                continue
        
        raise RuntimeError("All models failed for inference")
    
    def log_inference_success(self, model: str, prompt_preview: str):
        """Log successful inference for performance tracking."""
        log_feedback("inference_success", {
            "model": model,
            "prompt_preview": prompt_preview,
            "timestamp": datetime.utcnow().isoformat(),
            "inference_type": "docker_model_runner"
        })
    
    def log_inference_failure(self, model: str, error: str):
        """Log failed inference for debugging."""
        log_feedback("inference_failure", {
            "model": model,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "inference_type": "docker_model_runner"
        })

# Global hybrid client
_hybrid_client = None

def get_hybrid_client():
    """Get or create hybrid inference client."""
    global _hybrid_client
    if _hybrid_client is None:
        _hybrid_client = HybridInferenceClient()
    return _hybrid_client
```

**Enhanced Feedback Collection:**

```python
def log_feedback_enhanced(event: str, details: dict, performance_metrics: dict = None):
    """Enhanced feedback logging with performance metrics and training preparation."""
    
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "details": details,
        "performance_metrics": performance_metrics or {},
        "training_metadata": {
            "suitable_for_training": is_suitable_for_training(event, details),
            "quality_score": calculate_quality_score(details),
            "data_category": categorize_training_data(event, details)
        }
    }
    
    # Log to file (existing functionality)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    
    # Log to database (new functionality)
    if os.environ.get("FEEDBACK_DATABASE") == "enabled":
        store_feedback_in_database(feedback_entry)

def is_suitable_for_training(event: str, details: dict) -> bool:
    """Determine if this feedback is suitable for training data."""
    training_events = ["score_bias", "score_sentiment", "identify_entities"]
    has_valid_output = details.get("score") is not None or details.get("entities")
    return event in training_events and has_valid_output

def calculate_quality_score(details: dict) -> float:
    """Calculate quality score for training data prioritization."""
    score = 1.0
    
    # Reduce score for errors
    if "error" in details:
        score *= 0.1
    
    # Increase score for successful complex operations
    if details.get("entities") and len(details["entities"]) > 2:
        score *= 1.2
    
    # Increase score for clear bias/sentiment indicators
    if details.get("score") and abs(details["score"]) > 0.7:
        score *= 1.1
    
    return min(score, 1.0)

def categorize_training_data(event: str, details: dict) -> str:
    """Categorize feedback for specialized model training."""
    categories = {
        "score_bias": "bias_detection",
        "score_sentiment": "sentiment_analysis", 
        "identify_entities": "entity_recognition"
    }
    return categories.get(event, "general")
```

#### 2.3 Updated Analysis Functions

**Bias Scoring with Docker Model Runner:**

```python
def score_bias(text: str) -> float:
    """Score political bias using Docker Model Runner with enhanced feedback."""
    
    start_time = time.time()
    client = get_hybrid_client()
    
    prompt = f"""Analyze the political bias in this news text. Respond with only a number between -1.0 (left bias) and +1.0 (right bias), where 0.0 is neutral.

Text: {text[:2000]}

Bias score:"""
    
    try:
        result = client.generate_with_fallback(prompt, model_preference="ai/mistral:7b-instruct-v0.3")
        inference_time = time.time() - start_time
        
        # Extract score from response
        score_match = re.search(r'-?\d*\.?\d+', result.strip())
        if score_match:
            score = float(score_match.group())
            score = max(-1.0, min(1.0, score))  # Clamp to valid range
        else:
            logger.warning(f"Could not parse bias score from: {result}")
            score = 0.0
        
        # Enhanced feedback logging
        performance_metrics = {
            "inference_time": inference_time,
            "model_used": client.primary_model,
            "text_length": len(text),
            "confidence": abs(score)  # Higher absolute values indicate higher confidence
        }
        
        log_feedback_enhanced("score_bias", {
            "text": text[:200], 
            "score": score, 
            "raw_response": result,
            "method": "docker_model_runner"
        }, performance_metrics)
        
        return score
        
    except Exception as e:
        error_time = time.time() - start_time
        log_feedback_enhanced("score_bias_error", {
            "text": text[:200], 
            "error": str(e),
            "method": "docker_model_runner",
            "error_time": error_time
        })
        logger.error(f"Error in score_bias: {e}")
        return 0.0  # Neutral fallback
```

#### 2.4 Database Schema Updates

**New Training Data Tables:**

```sql
-- Enhanced training examples table
CREATE TABLE IF NOT EXISTS training_examples_v4 (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Source information
    agent_name VARCHAR(50) NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    
    -- Input/Output data
    input_text TEXT NOT NULL,
    expected_output JSONB NOT NULL,
    actual_output JSONB,
    
    -- Quality metrics
    quality_score FLOAT DEFAULT 0.0,
    training_priority INTEGER DEFAULT 0,
    
    -- Performance data
    inference_time FLOAT,
    model_used VARCHAR(100),
    
    -- Training metadata
    suitable_for_training BOOLEAN DEFAULT FALSE,
    used_in_training BOOLEAN DEFAULT FALSE,
    training_iteration INTEGER
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    agent_name VARCHAR(50) NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    model_used VARCHAR(100),
    
    inference_time FLOAT,
    success_rate FLOAT,
    error_rate FLOAT,
    
    -- Performance benchmarks
    accuracy_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    
    -- Metadata
    sample_size INTEGER,
    benchmark_version VARCHAR(20)
);

-- Model evolution tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50), -- 'docker_runner' or 'custom'
    version VARCHAR(50) NOT NULL,
    
    -- Performance metrics
    benchmark_score FLOAT,
    training_examples_count INTEGER,
    
    -- Deployment status
    status VARCHAR(20) DEFAULT 'training', -- 'training', 'testing', 'deployed', 'deprecated'
    deployment_percentage FLOAT DEFAULT 0,
    
    -- Configuration
    model_config JSONB,
    training_config JSONB
);
```

### Phase 1 Validation Tasks

#### 2.5 Testing and Validation

**Performance Benchmarking:**

```python
# tests/test_phase1_migration.py
import pytest
import time
from agents.analyst.tools import score_bias, score_sentiment, identify_entities

class TestPhase1Migration:
    """Test suite for Phase 1 Docker Model Runner migration."""
    
    def test_docker_model_runner_connectivity(self):
        """Verify Docker Model Runner is accessible."""
        from agents.analyst.tools import get_hybrid_client
        
        client = get_hybrid_client()
        response = client.generate_with_fallback("Hello, are you working?")
        
        assert response is not None
        assert len(response) > 0
        assert "error" not in response.lower()
    
    def test_bias_scoring_performance(self):
        """Test bias scoring performance meets requirements."""
        test_text = """The new tax policy announced yesterday has sparked heated debate among politicians. Progressive lawmakers argue it unfairly benefits the wealthy, while conservative representatives claim it will stimulate economic growth."""
        
        start_time = time.time()
        score = score_bias(test_text)
        inference_time = time.time() - start_time
        
        # Performance requirements
        assert inference_time < 5.0  # Under 5 seconds
        assert -1.0 <= score <= 1.0   # Valid range
        
        # Should detect some bias in politically charged text
        assert abs(score) > 0.1
    
    def test_fallback_mechanism(self):
        """Test that fallback models work when primary fails."""
        from agents.analyst.tools import get_hybrid_client
        
        client = get_hybrid_client()
        
        # Simulate primary model failure by using non-existent model
        original_primary = client.primary_model
        client.primary_model = "ai/nonexistent:model"
        
        try:
            response = client.generate_with_fallback("Test fallback mechanism")
            assert response is not None
            assert len(response) > 0
        finally:
            client.primary_model = original_primary
    
    def test_feedback_collection_enhancement(self):
        """Verify enhanced feedback collection is working."""
        from agents.analyst.tools import log_feedback_enhanced
        
        test_details = {
            "text": "Test news article",
            "score": 0.7,
            "model_used": "ai/mistral:7b-instruct-v0.3"
        }
        
        test_metrics = {
            "inference_time": 1.5,
            "confidence": 0.7
        }
        
        # Should not raise exceptions
        log_feedback_enhanced("score_bias", test_details, test_metrics)
        
        # Verify feedback log file exists and has new entry
        import os
        assert os.path.exists(FEEDBACK_LOG)
```

**Migration Validation Script:**

```bash
#!/bin/bash
# validate_phase1.sh

echo "=== Phase 1 Migration Validation ==="

# 1. Check Docker Model Runner status
echo "Checking Docker Model Runner..."
docker model status || exit 1

# 2. Pull required models
echo "Pulling required models..."
docker model pull ai/mistral:7b-instruct-v0.3
docker model pull ai/llama3.2:7b-instruct
docker model pull ai/gemma3:7b-instruct

# 3. Test model connectivity
echo "Testing model connectivity..."
docker model run ai/mistral:7b-instruct-v0.3 "Hello, respond with 'OK' if you're working."

# 4. Start services
echo "Starting JustNews services..."
docker-compose up -d

# 5. Wait for services to be ready
echo "Waiting for services to initialize..."
sleep 30

# 6. Run integration tests
echo "Running integration tests..."
python -m pytest tests/test_phase1_migration.py -v

# 7. Performance benchmark
echo "Running performance benchmarks..."
python tests/benchmark_phase1.py

echo "=== Phase 1 Validation Complete ==="
```

### Phase 1 Success Criteria

- ✅ All Docker Model Runner models accessible and responsive
- ✅ Zero model corruption incidents
- ✅ Average inference time < 3 seconds
- ✅ 99%+ inference success rate with fallback
- ✅ Enhanced feedback collection operational
- ✅ All existing functionality preserved
- ✅ Performance meets or exceeds V3 baseline

---

## 3. Phase 2: Training Pipeline Development (Weeks 3-6)

### Objectives
- Build custom model training infrastructure from collected feedback
- Implement automated training data preparation pipeline
- Create A/B testing framework for model comparison
- Train first generation custom models

### Technical Tasks

#### 3.1 Training Infrastructure Setup

**Training Service Architecture:**

```yaml
# docker-compose.training.yml
version: '3.8'
services:
  # Training coordinator service
  training-coordinator:
    build: ./training/coordinator
    environment:
      - TRAINING_DATA_SOURCE=database
      - MODEL_REGISTRY_URL=http://model-registry:5000
      - TRAINING_QUEUE=redis://redis:6379
    depends_on:
      - redis
      - model-registry
      - db
    volumes:
      - ./training/models:/models
      - ./training/data:/training_data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Model registry for version control
  model-registry:
    image: registry:2
    ports:
      - "5000:5000"
    environment:
      - REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY=/var/lib/registry
    volumes:
      - ./training/registry:/var/lib/registry

  # Training data processor
  data-processor:
    build: ./training/data_processor
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/justnews
      - FEEDBACK_LOG_PATH=/feedback_logs
    volumes:
      - ./feedback_logs:/feedback_logs
      - ./training/data:/processed_data
    depends_on:
      - db

  # Redis for training queue
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

#### 3.2 Training Data Pipeline

**Training Data Processor:**

```python
# training/data_processor/processor.py
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import sqlite3
import psycopg2

class TrainingDataProcessor:
    """Processes feedback logs into training datasets."""
    
    def __init__(self, db_url: str, feedback_log_path: str):
        self.db_url = db_url
        self.feedback_log_path = feedback_log_path
        
    def extract_training_examples(self, days_back: int = 7) -> Dict[str, List]:
        """Extract training examples from recent feedback logs."""
        
        # Get cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        training_data = {
            "bias_detection": [],
            "sentiment_analysis": [],
            "entity_recognition": []
        }
        
        # Process database feedback
        db_examples = self._extract_from_database(cutoff_date)
        
        # Process file-based feedback
        file_examples = self._extract_from_files(cutoff_date)
        
        # Combine and categorize
        all_examples = db_examples + file_examples
        
        for example in all_examples:
            category = example.get("training_metadata", {}).get("data_category", "general")
            if category in training_data:
                training_data[category].append(example)
        
        return training_data
    
    def _extract_from_database(self, cutoff_date: datetime) -> List[Dict]:
        """Extract training examples from database."""
        conn = psycopg2.connect(self.db_url)
        
        query = """
        SELECT * FROM training_examples_v4 
        WHERE created_at >= %s 
        AND suitable_for_training = TRUE 
        AND quality_score > 0.5
        ORDER BY quality_score DESC, created_at DESC
        """
        
        df = pd.read_sql(query, conn, params=[cutoff_date])
        conn.close()
        
        return df.to_dict('records')
    
    def _extract_from_files(self, cutoff_date: datetime) -> List[Dict]:
        """Extract training examples from feedback log files."""
        examples = []
        
        try:
            with open(self.feedback_log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_date = datetime.fromisoformat(entry.get("timestamp", ""))
                        
                        if entry_date >= cutoff_date:
                            if entry.get("training_metadata", {}).get("suitable_for_training", False):
                                examples.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except FileNotFoundError:
            pass
        
        return examples
    
    def prepare_bias_training_data(self, examples: List[Dict]) -> Tuple[List[str], List[float]]:
        """Prepare bias detection training data."""
        texts = []
        scores = []
        
        for example in examples:
            details = example.get("details", {})
            if "text" in details and "score" in details:
                texts.append(details["text"])
                scores.append(float(details["score"]))
        
        return texts, scores
    
    def prepare_sentiment_training_data(self, examples: List[Dict]) -> Tuple[List[str], List[float]]:
        """Prepare sentiment analysis training data."""
        texts = []
        scores = []
        
        for example in examples:
            details = example.get("details", {})
            if "text" in details and "score" in details:
                texts.append(details["text"])
                scores.append(float(details["score"]))
        
        return texts, scores
    
    def prepare_entity_training_data(self, examples: List[Dict]) -> Tuple[List[str], List[List[Dict]]]:
        """Prepare entity recognition training data."""
        texts = []
        entities = []
        
        for example in examples:
            details = example.get("details", {})
            if "text" in details and "entities" in details:
                texts.append(details["text"])
                entities.append(details["entities"])
        
        return texts, entities
```

#### 3.3 Custom Model Training

**Fine-tuning Pipeline:**

```python
# training/coordinator/trainer.py
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import os

class NewsAnalysisTrainer:
    """Custom model trainer for news analysis tasks."""
    
    def __init__(self, base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Weights & Biases for experiment tracking
        wandb.init(project="justnews-v4-training")
    
    def train_bias_detector(self, texts: List[str], scores: List[float], 
                          output_dir: str = "./models/bias_detector_v1"):
        """Train a custom bias detection model."""
        
        # Prepare tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=1,  # Regression for bias score
            problem_type="regression"
        )
        
        # Prepare dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
        
        dataset = Dataset.from_dict({"text": texts, "labels": scores})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split dataset
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb"
        )
        
        # Custom metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.reshape(-1)
            
            # Calculate regression metrics
            mse = torch.nn.functional.mse_loss(torch.tensor(predictions), torch.tensor(labels))
            mae = torch.nn.functional.l1_loss(torch.tensor(predictions), torch.tensor(labels))
            
            return {
                "mse": mse.item(),
                "mae": mae.item(),
                "rmse": torch.sqrt(mse).item()
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Log final metrics
        eval_results = trainer.evaluate()
        wandb.log({"final_eval": eval_results})
        
        return output_dir
    
    def train_sentiment_analyzer(self, texts: List[str], scores: List[float],
                                output_dir: str = "./models/sentiment_analyzer_v1"):
        """Train a custom sentiment analysis model."""
        # Similar to bias detector but optimized for sentiment
        # Implementation details similar to train_bias_detector
        pass
    
    def train_entity_recognizer(self, texts: List[str], entities: List[List[Dict]],
                               output_dir: str = "./models/entity_recognizer_v1"):
        """Train a custom entity recognition model."""
        # Implementation for NER training using token classification
        pass
```

#### 3.4 A/B Testing Framework

**Model Comparison System:**

```python
# training/ab_testing/comparator.py
import random
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelPerformance:
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_inference_time: float
    error_rate: float
    user_satisfaction: float

class ABTestingFramework:
    """A/B testing framework for comparing Docker models vs custom models."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.test_configs = {}
        
    def create_ab_test(self, test_name: str, model_a: str, model_b: str, 
                      traffic_split: float = 0.5, duration_days: int = 7):
        """Create a new A/B test configuration."""
        
        test_config = {
            "test_name": test_name,
            "model_a": model_a,  # e.g., "docker:ai/mistral:7b-instruct-v0.3"
            "model_b": model_b,  # e.g., "custom:bias_detector_v1"
            "traffic_split": traffic_split,
            "start_date": datetime.utcnow(),
            "duration_days": duration_days,
            "active": True,
            "results": {
                "model_a": {"requests": 0, "successes": 0, "total_time": 0.0},
                "model_b": {"requests": 0, "successes": 0, "total_time": 0.0}
            }
        }
        
        self.test_configs[test_name] = test_config
        self._save_test_config(test_config)
        
    def route_request(self, test_name: str, request_data: Dict) -> str:
        """Route request to appropriate model based on A/B test configuration."""
        
        if test_name not in self.test_configs:
            raise ValueError(f"Test {test_name} not found")
        
        config = self.test_configs[test_name]
        
        if not config["active"]:
            return config["model_a"]  # Default to model A if test inactive
        
        # Route based on traffic split
        if random.random() < config["traffic_split"]:
            return config["model_a"]
        else:
            return config["model_b"]
    
    def record_result(self, test_name: str, model_id: str, success: bool, 
                     inference_time: float, user_feedback: Dict = None):
        """Record the result of a model inference for A/B testing."""
        
        if test_name not in self.test_configs:
            return
        
        config = self.test_configs[test_name]
        model_key = "model_a" if model_id == config["model_a"] else "model_b"
        
        config["results"][model_key]["requests"] += 1
        if success:
            config["results"][model_key]["successes"] += 1
        config["results"][model_key]["total_time"] += inference_time
        
        # Store detailed result in database
        self._store_ab_result(test_name, model_id, success, inference_time, user_feedback)
        
    def analyze_test(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results and determine winner."""
        
        if test_name not in self.test_configs:
            raise ValueError(f"Test {test_name} not found")
        
        config = self.test_configs[test_name]
        results = config["results"]
        
        # Calculate metrics for each model
        model_a_metrics = self._calculate_metrics(results["model_a"])
        model_b_metrics = self._calculate_metrics(results["model_b"])
        
        # Determine statistical significance (simplified)
        significance = self._calculate_significance(model_a_metrics, model_b_metrics)
        
        # Determine winner
        winner = self._determine_winner(model_a_metrics, model_b_metrics, significance)
        
        analysis = {
            "test_name": test_name,
            "model_a": {
                "id": config["model_a"],
                "metrics": model_a_metrics
            },
            "model_b": {
                "id": config["model_b"], 
                "metrics": model_b_metrics
            },
            "statistical_significance": significance,
            "winner": winner,
            "recommendation": self._generate_recommendation(winner, significance)
        }
        
        return analysis
    
    def _calculate_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate performance metrics from raw results."""
        requests = results["requests"]
        successes = results["successes"]
        total_time = results["total_time"]
        
        if requests == 0:
            return {"success_rate": 0.0, "avg_inference_time": 0.0, "error_rate": 1.0}
        
        return {
            "success_rate": successes / requests,
            "avg_inference_time": total_time / requests,
            "error_rate": (requests - successes) / requests
        }
    
    def _determine_winner(self, metrics_a: Dict, metrics_b: Dict, significance: float) -> str:
        """Determine the winning model based on multiple criteria."""
        
        # Weighted scoring system
        score_a = (
            metrics_a["success_rate"] * 0.4 +
            (1 / max(metrics_a["avg_inference_time"], 0.1)) * 0.3 +
            (1 - metrics_a["error_rate"]) * 0.3
        )
        
        score_b = (
            metrics_b["success_rate"] * 0.4 +
            (1 / max(metrics_b["avg_inference_time"], 0.1)) * 0.3 +
            (1 - metrics_b["error_rate"]) * 0.3
        )
        
        if significance < 0.05:  # Statistically significant
            return "model_a" if score_a > score_b else "model_b"
        else:
            return "inconclusive"
```

### Phase 2 Success Criteria

- ✅ Training pipeline processes feedback into quality training data
- ✅ First custom models achieve parity with Docker Model Runner
- ✅ A/B testing framework operational and collecting metrics
- ✅ Automated model training and evaluation pipeline
- ✅ Model registry and version control system functional

---

## 4. Phase 3: Progressive Model Replacement (Months 2-6)

### Objectives
- Deploy custom models with reliable fallback mechanisms
- Gradually replace Docker models based on performance metrics
- Achieve complete AI independence
- Optimize models for news-specific analysis tasks

### Technical Tasks

#### 4.1 Custom Model Deployment

**Hybrid Model Manager:**

```python
# agents/common/hybrid_model_manager.py
import os
import json
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests

class HybridModelManager:
    """Manages deployment and fallback between custom and Docker models."""
    
    def __init__(self):
        self.custom_models = {}
        self.model_configs = self._load_model_configs()
        self.deployment_strategy = self._load_deployment_strategy()
        
    def _load_model_configs(self) -> Dict:
        """Load model configuration from database or config file."""
        # Query model_versions table for active models
        return {
            "bias_detection": {
                "custom_model_path": "./models/bias_detector_v2",
                "docker_fallback": "ai/mistral:7b-instruct-v0.3",
                "deployment_percentage": 50,  # Start with 50% traffic
                "performance_threshold": {"accuracy": 0.85, "inference_time": 2.0}
            },
            "sentiment_analysis": {
                "custom_model_path": "./models/sentiment_analyzer_v1",
                "docker_fallback": "ai/llama3.2:7b-instruct",
                "deployment_percentage": 25,  # More conservative rollout
                "performance_threshold": {"accuracy": 0.80, "inference_time": 2.5}
            }
        }
    
    def get_model_for_task(self, task: str, request_id: str = None) -> Dict[str, Any]:
        """Get appropriate model for task based on deployment strategy."""
        
        if task not in self.model_configs:
            raise ValueError(f"Unknown task: {task}")
        
        config = self.model_configs[task]
        
        # Determine if we should use custom model based on deployment percentage
        use_custom = self._should_use_custom_model(task, request_id)
        
        if use_custom and self._is_custom_model_available(task):
            return {
                "type": "custom",
                "model_path": config["custom_model_path"],
                "fallback": config["docker_fallback"]
            }
        else:
            return {
                "type": "docker",
                "model_name": config["docker_fallback"]
            }
    
    def _should_use_custom_model(self, task: str, request_id: str) -> bool:
        """Determine if custom model should be used based on deployment percentage."""
        import hashlib
        
        config = self.model_configs[task]
        deployment_pct = config.get("deployment_percentage", 0)
        
        if deployment_pct == 0:
            return False
        elif deployment_pct == 100:
            return True
        else:
            # Consistent routing based on request_id hash
            if request_id:
                hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
                return (hash_val % 100) < deployment_pct
            else:
                # Random routing if no request_id
                import random
                return random.random() < (deployment_pct / 100.0)
    
    def load_custom_model(self, task: str) -> bool:
        """Load custom model into memory."""
        
        if task not in self.model_configs:
            return False
        
        config = self.model_configs[task]
        model_path = config["custom_model_path"]
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            self.custom_models[task] = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "loaded_at": datetime.utcnow()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom model for {task}: {e}")
            return False
    
    def inference_with_fallback(self, task: str, text: str, request_id: str = None) -> Dict[str, Any]:
        """Perform inference with automatic fallback to Docker model."""
        
        model_info = self.get_model_for_task(task, request_id)
        
        try:
            if model_info["type"] == "custom":
                result = self._custom_inference(task, text)
                
                # Log successful custom inference
                self._log_inference_result(task, "custom", True, result.get("inference_time", 0))
                
                return result
                
            else:
                result = self._docker_inference(model_info["model_name"], text)
                
                # Log Docker inference
                self._log_inference_result(task, "docker", True, result.get("inference_time", 0))
                
                return result
                
        except Exception as e:
            logger.warning(f"Primary inference failed for {task}: {e}")
            
            # Fallback to Docker model
            try:
                fallback_model = model_info.get("fallback") or self.model_configs[task]["docker_fallback"]
                result = self._docker_inference(fallback_model, text)
                
                # Log fallback inference
                self._log_inference_result(task, "docker_fallback", True, result.get("inference_time", 0))
                
                return result
                
            except Exception as fallback_error:
                logger.error(f"Fallback inference also failed for {task}: {fallback_error}")
                
                # Log complete failure
                self._log_inference_result(task, "failed", False, 0)
                
                raise RuntimeError(f"Both primary and fallback inference failed for {task}")
```

#### 4.2 Performance Monitoring and Auto-scaling

**Model Performance Monitor:**

```python
# training/monitoring/performance_monitor.py
import time
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

class ModelPerformanceMonitor:
    """Monitors model performance and triggers scaling decisions."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    def collect_performance_metrics(self, hours_back: int = 24) -> Dict[str, Dict]:
        """Collect performance metrics for all models."""
        
        conn = psycopg2.connect(self.db_url)
        
        query = """
        SELECT 
            tool_name as task,
            model_used,
            AVG(inference_time) as avg_inference_time,
            COUNT(*) as request_count,
            SUM(CASE WHEN success_rate > 0.95 THEN 1 ELSE 0 END) / COUNT(*) as success_rate,
            AVG(accuracy_score) as avg_accuracy
        FROM performance_metrics 
        WHERE timestamp >= %s
        GROUP BY tool_name, model_used
        """
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        df = pd.read_sql(query, conn, params=[cutoff_time])
        conn.close()
        
        # Organize by task
        metrics_by_task = {}
        for _, row in df.iterrows():
            task = row['task']
            model = row['model_used']
            
            if task not in metrics_by_task:
                metrics_by_task[task] = {}
            
            metrics_by_task[task][model] = {
                "avg_inference_time": row['avg_inference_time'],
                "request_count": row['request_count'],
                "success_rate": row['success_rate'],
                "avg_accuracy": row['avg_accuracy']
            }
        
        return metrics_by_task
    
    def evaluate_deployment_readiness(self, task: str, custom_model: str) -> Dict[str, Any]:
        """Evaluate if custom model is ready for increased deployment."""
        
        metrics = self.collect_performance_metrics()
        
        if task not in metrics or custom_model not in metrics[task]:
            return {"ready": False, "reason": "Insufficient data"}
        
        custom_metrics = metrics[task][custom_model]
        
        # Find Docker model metrics for comparison
        docker_metrics = None
        for model_name, model_metrics in metrics[task].items():
            if model_name.startswith("ai/"):  # Docker Model Runner models
                docker_metrics = model_metrics
                break
        
        if not docker_metrics:
            return {"ready": False, "reason": "No Docker model baseline"}
        
        # Evaluation criteria
        evaluation = {
            "performance_better": custom_metrics["avg_accuracy"] > docker_metrics["avg_accuracy"],
            "speed_acceptable": custom_metrics["avg_inference_time"] < 3.0,
            "reliability_good": custom_metrics["success_rate"] > 0.95,
            "sufficient_volume": custom_metrics["request_count"] > 100
        }
        
        ready = all(evaluation.values())
        
        return {
            "ready": ready,
            "evaluation": evaluation,
            "custom_metrics": custom_metrics,
            "docker_metrics": docker_metrics
        }
    
    def recommend_deployment_percentage(self, task: str) -> int:
        """Recommend deployment percentage for custom model."""
        
        readiness = self.evaluate_deployment_readiness(task, f"custom_{task}_model")
        
        if not readiness["ready"]:
            return 0
        
        current_percentage = self._get_current_deployment_percentage(task)
        
        # Conservative scaling strategy
        if current_percentage == 0:
            return 10  # Start with 10%
        elif current_percentage < 50:
            return min(current_percentage + 20, 50)  # Increase by 20% up to 50%
        elif current_percentage < 90:
            return min(current_percentage + 10, 90)  # More conservative after 50%
        else:
            return 100  # Full deployment
    
    def auto_scale_deployment(self):
        """Automatically scale deployment based on performance."""
        
        tasks = ["bias_detection", "sentiment_analysis", "entity_recognition"]
        
        for task in tasks:
            try:
                recommended_pct = self.recommend_deployment_percentage(task)
                current_pct = self._get_current_deployment_percentage(task)
                
                if recommended_pct != current_pct:
                    self._update_deployment_percentage(task, recommended_pct)
                    
                    logger.info(f"Updated {task} deployment: {current_pct}% -> {recommended_pct}%")
                    
            except Exception as e:
                logger.error(f"Auto-scaling failed for {task}: {e}")
```

#### 4.3 Complete Independence Validation

**Independence Assessment Tool:**

```python
# scripts/assess_independence.py
import requests
import json
from datetime import datetime, timedelta
import psycopg2

class IndependenceAssessment:
    """Assess progress toward complete AI independence."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    def assess_current_state(self) -> Dict[str, Any]:
        """Assess current state of AI independence."""
        
        # Check custom model deployment percentages
        deployment_status = self._check_deployment_status()
        
        # Check external dependencies
        external_deps = self._check_external_dependencies()
        
        # Check performance parity
        performance_status = self._check_performance_parity()
        
        # Calculate overall independence score
        independence_score = self._calculate_independence_score(
            deployment_status, external_deps, performance_status
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "independence_score": independence_score,
            "deployment_status": deployment_status,
            "external_dependencies": external_deps,
            "performance_status": performance_status,
            "recommendations": self._generate_recommendations(
                deployment_status, external_deps, performance_status
            )
        }
    
    def _check_deployment_status(self) -> Dict[str, int]:
        """Check deployment percentages for all tasks."""
        conn = psycopg2.connect(self.db_url)
        
        query = """
        SELECT model_name, deployment_percentage 
        FROM model_versions 
        WHERE status = 'deployed' AND model_type = 'custom'
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        deployment_status = {}
        for model_name, deployment_pct in results:
            task = model_name.split('_')[0]  # Extract task from model name
            deployment_status[task] = deployment_pct
        
        return deployment_status
    
    def _check_external_dependencies(self) -> Dict[str, bool]:
        """Check for remaining external dependencies."""
        
        # Check if Docker Model Runner is still being used
        docker_usage = self._check_docker_model_usage()
        
        # Check for external API calls
        external_api_usage = self._check_external_api_usage()
        
        return {
            "docker_model_runner_active": docker_usage > 0,
            "external_api_calls": external_api_usage > 0,
            "docker_usage_percentage": docker_usage,
            "external_api_percentage": external_api_usage
        }
    
    def _check_performance_parity(self) -> Dict[str, Dict[str, float]]:
        """Check if custom models meet performance requirements."""
        
        conn = psycopg2.connect(self.db_url)
        
        # Get recent performance metrics
        query = """
        SELECT 
            tool_name,
            model_used,
            AVG(accuracy_score) as avg_accuracy,
            AVG(inference_time) as avg_inference_time
        FROM performance_metrics 
        WHERE timestamp >= %s
        GROUP BY tool_name, model_used
        """
        
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        cursor = conn.cursor()
        cursor.execute(query, [cutoff_time])
        results = cursor.fetchall()
        conn.close()
        
        performance_status = {}
        for tool, model, accuracy, inference_time in results:
            if tool not in performance_status:
                performance_status[tool] = {}
            
            model_type = "custom" if not model.startswith("ai/") else "docker"
            performance_status[tool][model_type] = {
                "accuracy": accuracy,
                "inference_time": inference_time
            }
        
        return performance_status
    
    def _calculate_independence_score(self, deployment_status: Dict, 
                                    external_deps: Dict, 
                                    performance_status: Dict) -> float:
        """Calculate overall independence score (0-100)."""
        
        # Deployment score (40% of total)
        avg_deployment = sum(deployment_status.values()) / len(deployment_status) if deployment_status else 0
        deployment_score = avg_deployment * 0.4
        
        # External dependency score (40% of total)
        docker_penalty = external_deps.get("docker_usage_percentage", 0) * 0.3
        api_penalty = external_deps.get("external_api_percentage", 0) * 0.1
        dependency_score = max(0, 40 - docker_penalty - api_penalty)
        
        # Performance score (20% of total)
        performance_score = 0
        if performance_status:
            for task, models in performance_status.items():
                if "custom" in models and "docker" in models:
                    custom_perf = models["custom"]["accuracy"]
                    docker_perf = models["docker"]["accuracy"] 
                    if custom_perf >= docker_perf:
                        performance_score += 20 / len(performance_status)
        
        total_score = deployment_score + dependency_score + performance_score
        return min(100, max(0, total_score))
    
    def generate_independence_report(self) -> str:
        """Generate a comprehensive independence assessment report."""
        
        assessment = self.assess_current_state()
        
        report = f"""
# JustNews V4 AI Independence Assessment Report
Generated: {assessment['timestamp']}

## Overall Independence Score: {assessment['independence_score']:.1f}/100

### Deployment Status
"""
        
        for task, percentage in assessment['deployment_status'].items():
            status = "✅" if percentage == 100 else "🔄" if percentage > 0 else "❌"
            report += f"- {task}: {percentage}% custom deployment {status}\n"
        
        report += f"""
### External Dependencies
- Docker Model Runner Usage: {'❌' if assessment['external_dependencies']['docker_model_runner_active'] else '✅'}
- External API Calls: {'❌' if assessment['external_dependencies']['external_api_calls'] else '✅'}

### Performance Status
"""
        
        for task, models in assessment['performance_status'].items():
            if 'custom' in models and 'docker' in models:
                custom_acc = models['custom']['accuracy']
                docker_acc = models['docker']['accuracy']
                comparison = "✅" if custom_acc >= docker_acc else "❌"
                report += f"- {task}: Custom {custom_acc:.3f} vs Docker {docker_acc:.3f} {comparison}\n"
        
        report += f"""
### Recommendations
"""
        for rec in assessment['recommendations']:
            report += f"- {rec}\n"
        
        return report

# Usage example
if __name__ == "__main__":
    assessor = IndependenceAssessment("postgresql://user:password@localhost:5432/justnews")
    report = assessor.generate_independence_report()
    print(report)
    
    # Save report
    with open(f"independence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", "w") as f:
        f.write(report)
```

### Phase 3 Success Criteria

- ✅ Custom models achieve >90% deployment for all core tasks
- ✅ Performance metrics show custom models outperform Docker models by >10%
- ✅ Zero external API dependencies
- ✅ Complete AI independence score >95/100
- ✅ News-specific optimizations demonstrate clear competitive advantages
- ✅ System handles 10x current load with custom models

---

## 5. Migration Timeline and Milestones

### Detailed Timeline

```
Phase 1: Foundation Migration (Weeks 1-2)
├─ Week 1
│  ├─ Day 1-2: Docker Model Runner setup and configuration
│  ├─ Day 3-4: Agent tools integration with hybrid inference
│  ├─ Day 5-7: Enhanced feedback collection implementation
└─ Week 2
   ├─ Day 8-10: Testing and validation
   ├─ Day 11-12: Performance benchmarking
   ├─ Day 13-14: Production deployment and monitoring

Phase 2: Training Pipeline (Weeks 3-6)
├─ Week 3
│  ├─ Training infrastructure setup
│  ├─ Data processing pipeline implementation
│  ├─ Model registry and version control
└─ Week 4-6
   ├─ A/B testing framework development
   ├─ First custom model training
   ├─ Performance validation and iteration

Phase 3: Progressive Replacement (Months 2-6)
├─ Month 2
│  ├─ 10% custom model deployment
│  ├─ Performance monitoring setup
│  ├─ Automated scaling implementation
├─ Month 3-4
│  ├─ 50% custom model deployment
│  ├─ News-specific optimizations
│  ├─ Advanced feature development
└─ Month 5-6
   ├─ 90-100% custom model deployment
   ├─ Complete independence validation
   ├─ Performance optimization and specialization
```

### Risk Mitigation Checkpoints

#### Week 1 Checkpoint
- ✅ Docker Model Runner responding to all test queries
- ✅ All existing functionality preserved
- ✅ No performance degradation observed
- **Go/No-Go Decision**: Proceed to Week 2 or rollback

#### Week 2 Checkpoint
- ✅ Enhanced feedback collection operational
- ✅ Performance meets baseline requirements
- ✅ System stability maintained for 48+ hours
- **Go/No-Go Decision**: Proceed to Phase 2 or stabilize Phase 1

#### Month 1 Checkpoint
- ✅ Training pipeline producing quality models
- ✅ A/B testing showing positive results
- ✅ Infrastructure scaling as expected
- **Go/No-Go Decision**: Proceed to deployment or iterate on training

#### Month 3 Checkpoint
- ✅ Custom models showing performance parity
- ✅ 50% deployment achieved without issues
- ✅ Cost reduction targets being met
- **Go/No-Go Decision**: Proceed to full deployment or optimize further

### Rollback Procedures

Each phase includes specific rollback procedures:

#### Phase 1 Rollback
```bash
# Revert to V3 configuration
git checkout v3-stable
docker-compose down
docker-compose -f docker-compose.v3.yml up -d
```

#### Phase 2 Rollback
```bash
# Disable training pipeline, keep Docker Model Runner
export TRAINING_MODE=disabled
docker-compose restart analyst critic synthesizer
```

#### Phase 3 Rollback
```python
# Reduce custom model deployment to 0%
deployment_manager.set_deployment_percentage("all_tasks", 0)
# System automatically falls back to Docker Model Runner
```

---

## 6. Success Metrics and KPIs

### Phase 1 KPIs
- **Reliability**: 99.9%+ uptime, zero model corruption incidents
- **Performance**: <3 second average inference time
- **Functionality**: 100% feature parity with V3
- **Data Collection**: 100% feedback capture rate

### Phase 2 KPIs
- **Training Efficiency**: <24 hour model iteration cycles
- **Model Quality**: Custom models achieve 95%+ parity with Docker models
- **Automation**: 90%+ automated training pipeline processes
- **A/B Testing**: Statistical significance achieved in <1 week tests

### Phase 3 KPIs
- **Independence**: 95%+ independence score
- **Performance**: Custom models outperform Docker models by 10%+
- **Cost**: 80%+ reduction in AI infrastructure costs
- **Specialization**: News-specific features not available in general models

### Business Impact Metrics
- **Cost Savings**: Target $120k+/year in eliminated API costs
- **Performance Improvement**: 20%+ improvement in news analysis accuracy
- **Competitive Advantage**: Proprietary AI capabilities
- **Operational Efficiency**: 50%+ reduction in model-related debugging time

---

## 7. Resource Requirements

### Infrastructure Requirements

#### Phase 1
- **Compute**: Existing GPU resources + Docker Model Runner
- **Storage**: +50GB for Docker model cache
- **Network**: Enhanced logging infrastructure
- **Personnel**: 1 senior engineer, 0.5 DevOps engineer

#### Phase 2
- **Compute**: Additional GPU for training (recommend RTX 4090 or A6000)
- **Storage**: +500GB for training data and model versions
- **Network**: Model registry and training data pipelines
- **Personnel**: +1 ML engineer, +0.5 data engineer

#### Phase 3
- **Compute**: Production-grade inference infrastructure
- **Storage**: +1TB for complete model independence
- **Network**: High-throughput model serving
- **Personnel**: +0.5 ML engineer for optimization

### Budget Estimates

#### One-time Costs
- **Phase 1**: $5,000 (infrastructure scaling)
- **Phase 2**: $15,000 (training hardware + cloud resources)
- **Phase 3**: $10,000 (production optimization)
- **Total**: $30,000 one-time investment

#### Ongoing Savings
- **Year 1**: $60,000 saved (50% reduction in AI costs)
- **Year 2+**: $120,000+ saved annually (complete independence)
- **ROI**: 4x return on investment within 12 months

---

## 8. Conclusion

The JustNews V4 migration plan provides a comprehensive roadmap from the current state to complete AI independence while maintaining operational excellence throughout the transition.

### Key Success Factors
1. **Phased Approach**: Risk-mitigated progression with clear rollback options
2. **Zero Downtime**: Continuous operation throughout migration
3. **Performance Focus**: Maintaining and improving system performance
4. **Data-Driven Decisions**: Metrics-based progression through phases

### Expected Outcomes
- **Immediate**: Elimination of model corruption issues
- **Short-term**: 50% cost reduction, improved reliability
- **Long-term**: Complete AI independence, superior performance, competitive advantage

This plan transforms JustNewsAgentic from a consumer of AI services to a producer of specialized AI capabilities, ensuring long-term sustainability and competitive differentiation in the news analysis market.

---

*This document serves as the definitive implementation guide for JustNews V4 migration. All phases should be executed with careful attention to success criteria and rollback procedures.*
