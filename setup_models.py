#!/usr/bin/env python3
"""
Local Model Setup Script for JustNewsAgentic
Downloads and caches gated models locally for offline use.
"""

import os
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login, snapshot_download
except ImportError:
    print("❌ Please install transformers and huggingface_hub first:")
    print("pip install transformers huggingface_hub accelerate torch")
    sys.exit(1)

def setup_models():
    """Download and cache models locally"""
    
    # Get HF token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face token: ").strip()
    
    if not hf_token:
        print("❌ Hugging Face token is required for gated models")
        return False
    
    # Login to HF
    try:
        login(token=hf_token)
        print("✅ Logged into Hugging Face")
    except Exception as e:
        print(f"❌ Failed to login to Hugging Face: {e}")
        return False
    
    # Download Mistral 7B Instruct v0.3 (optimized download)
    print(f"\n📥 Downloading Mistral-7B-Instruct-v0.3 (essential files only)...")
    mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
    mistral_models_path.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
            allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], 
            local_dir=mistral_models_path,
            token=hf_token
        )
        print(f"✅ Mistral-7B-Instruct-v0.3 downloaded to: {mistral_models_path}")
    except Exception as e:
        print(f"❌ Failed to download Mistral-7B-Instruct-v0.3: {e}")
        return False
    
    # Alternative: Download via transformers (full model, larger but more compatible)
    print(f"\n📥 Downloading backup models via transformers...")
    
    # Models to download via transformers
    models = [
        "microsoft/DialoGPT-large",            # Fact Checker
        # Add other models as needed
    ]
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
    print(f"📁 Models will be cached to: {cache_dir}")
    
    for model_name in models:
        print(f"\n📥 Downloading {model_name}...")
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            print(f"✅ Tokenizer for {model_name} downloaded")
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            print(f"✅ Model {model_name} downloaded")
            
            # Free memory
            del model
            del tokenizer
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            continue
    
    print("\n🎉 Model setup complete!")
    print("Models are now cached locally and ready for offline use.")
    return True

if __name__ == "__main__":
    print("🚀 JustNewsAgentic Model Setup")
    print("===============================")
    
    setup_models()
