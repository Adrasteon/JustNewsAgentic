
"""
Tools for the Analyst Agent.
"""

import logging
import os
import json
from datetime import datetime
import re

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyst.tools")

# Import ML dependencies
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.pipelines import pipeline
    import torch
    from safetensors.torch import load_file
    HAS_TRANSFORMERS = True
    # Type annotations for when available
    ModelType = AutoModelForCausalLM
    TokenizerType = AutoTokenizer
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    torch = None
    ModelType = None
    TokenizerType = None

# Model configuration - using Mistral 7B Instruct v0.3 for GPU performance
MODEL_NAME = os.environ.get("ANALYST_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
FEEDBACK_LOG = "feedback_analyst.log"

# Global variables for lazy loading
_model = None
_tokenizer = None

def get_dialog_model():
    """Loads and returns the dialog model and tokenizer with GPU support and offline capability."""
    global _model, _tokenizer
    
    if not HAS_TRANSFORMERS:
        raise ImportError("Transformers library not available. Please install with: pip install transformers torch")
    
    if _model is None or _tokenizer is None:
        # Check for local model first
        from pathlib import Path
        # Configure model paths - support both local development and Docker
        local_model_path = Path(os.getenv('LOCAL_MODEL_PATH', Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')))
        
        model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.3"
        
        # Detect device
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        try:
            # If we have local files with config, try offline loading
            if local_model_path.exists() and (local_model_path / "config.json").exists():
                logger.info(f"Found local model files at {local_model_path}")
                logger.info(f"Attempting fully offline model loading")
                
                # Use local files only
                _tokenizer = AutoTokenizer.from_pretrained(
                    str(local_model_path),
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # Load model with GPU optimization
                model_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True,
                }
                
                if device == "cuda" and torch:
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "attn_implementation": "eager"
                    })
            
            if local_model_path.exists():
                logger.info("Attempting fully offline model loading")
                
                # Use local files only
                _tokenizer = AutoTokenizer.from_pretrained(
                    str(local_model_path),
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # CRITICAL: Disable chat template to prevent role validation errors
                logger.info("Disabling tokenizer chat template to prevent role validation errors")
                if hasattr(_tokenizer, 'chat_template'):
                    _tokenizer.chat_template = None
                # Also try alternative attributes that might contain the template
                if hasattr(_tokenizer, 'default_chat_template'):
                    _tokenizer.default_chat_template = None
                # Force use of padding token as EOS if needed
                if _tokenizer.pad_token is None:
                    _tokenizer.pad_token = _tokenizer.eos_token
                
                # Load model with GPU optimization
                model_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True,
                }
                
                if device == "cuda" and torch:
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "attn_implementation": "eager"
                    })
                
                _model = AutoModelForCausalLM.from_pretrained(
                    str(local_model_path),
                    **model_kwargs
                )
                
                logger.info(f"Successfully loaded model offline from {local_model_path}")
                
            else:
                # Fallback to HuggingFace Hub approach
                logger.info(f"Loading model from HuggingFace: {model_name_or_path}")
                
                # Check for HF token
                hf_token = os.environ.get("HUGGINGFACE_TOKEN")
                if not hf_token:
                    logger.warning("HUGGINGFACE_TOKEN not found. You may not be able to access gated models.")
                
                # Load tokenizer from HuggingFace Hub
                _tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    token=hf_token,
                    trust_remote_code=True
                )
                
                # Load model with GPU optimization
                model_kwargs = {
                    "trust_remote_code": True,
                }
                
                if hf_token:
                    model_kwargs["token"] = hf_token
                
                if device == "cuda" and torch:
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "attn_implementation": "eager"
                    })
                
                _model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            
            # Set pad token if not exists
            if hasattr(_tokenizer, 'pad_token') and _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            logger.info(f"Successfully loaded model on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    return _model, _tokenizer

def log_feedback(event: str, details: dict):
    """Logs feedback to a file."""
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def score_bias(text: str) -> float:
    """Scores the bias of a given text using the pre-loaded model."""
    logger.info(f"Scoring bias for text: '{text[:50]}...' ")
    try:
        model, tokenizer = get_dialog_model()
        
        # Simple prompt format - bypass chat template for now
        prompt = f"Analyze the political bias in this news text. Rate from 0.0 (left-leaning) to 1.0 (right-leaning), 0.5 is neutral.\n\nText: {text}\n\nBias score:"
        
        logger.info(f"Using prompt: '{prompt[:100]}...'")
        
        # Try very simple tokenization - avoid any special handling
        logger.info("Tokenizing with encode instead of tokenizer call...")
        
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1000)
            logger.info(f"Successfully tokenized input, shape: {input_ids.shape}")
        except Exception as tokenizer_error:
            logger.error(f"Tokenization error: {tokenizer_error}")
            raise tokenizer_error
        
        if torch and torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        logger.info("Starting generation...")
        if torch:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
        else:
            outputs = model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        logger.info("Generation completed, decoding...")
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the score from the generated text
        generated_part = result[len(prompt):].strip()
        logger.info(f"Generated response: '{generated_part}'")
        
        # Try to parse a number from the response
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0\.0', generated_part)
        if numbers:
            score = float(numbers[0])
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
        else:
            # Try to extract any decimal number
            all_numbers = re.findall(r'\d+\.?\d*', generated_part)
            if all_numbers:
                try:
                    score = float(all_numbers[0])
                    if score > 1.0:
                        score = score / 10.0  # Assume they meant 0.X
                    score = max(0.0, min(1.0, score))
                except:
                    score = 0.5  # Default to neutral
            else:
                score = 0.5  # Default to neutral if no valid score found
        
        log_feedback("score_bias", {"text": text[:100], "score": score, "raw": result, "generated": generated_part})
        return score
    except Exception as e:
        logger.error(f"Error in score_bias: {e}")
        log_feedback("score_bias_error", {"text": text[:100], "error": str(e)})
        return 0.5

def score_sentiment(text: str) -> float:
    """Scores the sentiment of a given text using the pre-loaded model."""
    logger.info(f"Scoring sentiment for text: '{text[:50]}...' ")
    try:
        model, tokenizer = get_dialog_model()
        
        # Simple prompt format - bypass chat template for now
        prompt = f"Analyze the emotional sentiment in this news text. Rate from 0.0 (very negative) to 1.0 (very positive), 0.5 is neutral.\n\nText: {text}\n\nSentiment score:"
        
        # Tokenize and generate with GPU optimization  
        inputs = tokenizer(prompt, return_tensors="pt", truncate=True, max_length=1000)
        if torch and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        if torch:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the score from the generated text
        generated_part = result[len(prompt):].strip()
        logger.info(f"Generated response: '{generated_part}'")
        
        # Try to parse a number from the response
        numbers = re.findall(r'0\.\d+|1\.0|0\.0', generated_part)
        if numbers:
            score = float(numbers[0])
            score = max(0.0, min(1.0, score))
        else:
            # Try to extract any decimal number
            all_numbers = re.findall(r'\d+\.?\d*', generated_part)
            if all_numbers:
                try:
                    score = float(all_numbers[0])
                    if score > 1.0:
                        score = score / 10.0  # Assume they meant 0.X
                    score = max(0.0, min(1.0, score))
                except:
                    score = 0.5  # Default to neutral
            else:
                score = 0.5  # Default to neutral if no valid score found
        
        log_feedback("score_sentiment", {"text": text[:100], "score": score, "raw": result, "generated": generated_part})
        return score
    except Exception as e:
        logger.error(f"Error in score_sentiment: {e}")
        log_feedback("score_sentiment_error", {"text": text[:100], "error": str(e)})
        return 0.5

def identify_entities(text: str) -> list:
    """Identifies named entities in the given text using the pre-loaded model."""
    logger.info(f"Identifying entities for text: '{text[:50]}...' ")
    try:
        model, tokenizer = get_dialog_model()
        
        # Simple prompt format - bypass chat template for now
        prompt = f"Extract all named entities (people, organizations, locations) from this news text. List them separated by commas.\n\nText: {text}\n\nEntities:"
        
        # Tokenize and generate with GPU optimization
        inputs = tokenizer(prompt, return_tensors="pt", truncate=True, max_length=1000)
        if torch and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        if torch:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the entities from the generated text
        generated_part = result[len(prompt):].strip()
        logger.info(f"Generated response: '{generated_part}'")
        
        # Split by commas and clean up
        entities = []
        if generated_part:
            for item in generated_part.split(','):
                item = item.strip()
                # Remove common prefixes and clean up
                item = re.sub(r'^[•\-\d\.\s]+', '', item).strip()
                if item and len(item) > 1 and not item.lower().startswith(('here', 'the', 'entities', 'and')):
                    entities.append(item)
        
        log_feedback("identify_entities", {"text": text[:100], "entities": entities, "raw": result, "generated": generated_part})
        return entities[:10]  # Limit to 10 entities
    except Exception as e:
        logger.error(f"Error in identify_entities: {e}")
        log_feedback("identify_entities_error", {"text": text[:100], "error": str(e)})
        return []
