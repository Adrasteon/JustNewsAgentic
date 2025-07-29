"""
GPU-Accelerated Scout Agent Engine with LLaMA-3-8B Intelligence
Production-ready content pre-filtering and classification system
"""

import os
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class GPUScoutInferenceEngine:
    """
    Production-ready Scout agent with LLaMA-3-8B intelligence for content pre-filtering
    
    Features:
    - News vs non-news classification
    - Content quality assessment  
    - Bias detection and flagging
    - Source credibility scoring
    - Batch processing for efficiency
    - GPU acceleration with FP16 precision
    """
    
    def __init__(self, model_path: str = "microsoft/DialoGPT-medium"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.classification_pipeline = None
        
        # Initialize models
        self._initialize_models()
        
        # Content classification prompts
        self.classification_prompts = {
            "news_detection": """
Analyze the following content and determine if it is legitimate news content.

Content: {content}

Respond with JSON format:
{
    "is_news": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "content_type": "news|opinion|forum|advertisement|social_media|other"
}
""",
            "quality_assessment": """
Assess the quality of this news content on multiple dimensions.

Content: {content}

Respond with JSON format:
{
    "overall_quality": 0.0-1.0,
    "factual_content": 0.0-1.0,
    "source_credibility": 0.0-1.0,
    "writing_quality": 0.0-1.0,
    "completeness": 0.0-1.0,
    "reasoning": "detailed explanation"
}
""",
            "bias_detection": """
Analyze this content for political, ideological, or commercial bias.

Content: {content}

Respond with JSON format:
{
    "bias_score": 0.0-1.0,
    "bias_type": "political|commercial|ideological|none",
    "bias_direction": "left|right|center|pro|anti|neutral",
    "bias_indicators": ["list of specific bias indicators found"],
    "confidence": 0.0-1.0
}
"""
        }
    
    def _initialize_models(self):
        """Initialize LLaMA-3-8B model with GPU optimization"""
        try:
            logger.info(f"🔄 Loading Scout LLaMA-3-8B model on {self.device}...")
            
            # Quantization config for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with GPU optimization
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                    # Removed flash_attention_2 requirement
                )
                
                # Create pipeline for batch processing
                self.classification_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    # Remove device parameter since model is already loaded with device_map="auto"
                    batch_size=4,  # Optimized for 8GB memory allocation
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9
                )
                
                logger.info("✅ GPU-accelerated Scout model loaded successfully")
            else:
                # CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                
                self.classification_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.1
                )
                
                logger.warning("⚠️ Using CPU fallback for Scout model")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Scout model: {e}")
            raise
    
    def classify_news_content(self, content: str, url: str = None) -> Dict:
        """
        Classify content as news vs non-news with confidence scoring
        """
        try:
            prompt = self.classification_prompts["news_detection"].format(content=content[:2000])
            
            # Generate classification
            response = self.classification_pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1
            )
            
            # Extract JSON response
            generated_text = response[0]['generated_text'][len(prompt):].strip()
            classification = self._extract_json_response(generated_text)
            
            # Add metadata
            classification["url"] = url
            classification["timestamp"] = datetime.utcnow().isoformat()
            classification["content_length"] = len(content)
            
            return classification
            
        except Exception as e:
            logger.error(f"❌ News classification failed: {e}")
            return {
                "is_news": False,
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}",
                "content_type": "error",
                "url": url,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def assess_content_quality(self, content: str, url: str = None) -> Dict:
        """
        Assess content quality across multiple dimensions
        """
        try:
            prompt = self.classification_prompts["quality_assessment"].format(content=content[:2000])
            
            response = self.classification_pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1
            )
            
            generated_text = response[0]['generated_text'][len(prompt):].strip()
            quality_assessment = self._extract_json_response(generated_text)
            
            # Add metadata
            quality_assessment["url"] = url
            quality_assessment["timestamp"] = datetime.utcnow().isoformat()
            quality_assessment["content_length"] = len(content)
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            return {
                "overall_quality": 0.0,
                "factual_content": 0.0,
                "source_credibility": 0.0,
                "writing_quality": 0.0,
                "completeness": 0.0,
                "reasoning": f"Assessment error: {str(e)}",
                "url": url,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def detect_bias(self, content: str, url: str = None) -> Dict:
        """
        Detect and analyze bias in content
        """
        try:
            prompt = self.classification_prompts["bias_detection"].format(content=content[:2000])
            
            response = self.classification_pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1
            )
            
            generated_text = response[0]['generated_text'][len(prompt):].strip()
            bias_analysis = self._extract_json_response(generated_text)
            
            # Add metadata
            bias_analysis["url"] = url
            bias_analysis["timestamp"] = datetime.utcnow().isoformat()
            bias_analysis["content_length"] = len(content)
            
            return bias_analysis
            
        except Exception as e:
            logger.error(f"❌ Bias detection failed: {e}")
            return {
                "bias_score": 0.5,
                "bias_type": "unknown",
                "bias_direction": "unknown",
                "bias_indicators": [],
                "confidence": 0.0,
                "url": url,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def comprehensive_content_analysis(self, content: str, url: str = None) -> Dict:
        """
        Perform comprehensive analysis combining all Scout capabilities
        """
        try:
            logger.info(f"🔍 Performing comprehensive analysis for: {url}")
            
            # Run all analyses
            news_classification = self.classify_news_content(content, url)
            quality_assessment = self.assess_content_quality(content, url)
            bias_analysis = self.detect_bias(content, url)
            
            # Calculate overall Scout score
            overall_score = self._calculate_scout_score(
                news_classification, quality_assessment, bias_analysis
            )
            
            comprehensive_analysis = {
                "scout_score": overall_score,
                "news_classification": news_classification,
                "quality_assessment": quality_assessment,
                "bias_analysis": bias_analysis,
                "recommendation": self._generate_recommendation(
                    news_classification, quality_assessment, bias_analysis, overall_score
                ),
                "url": url,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Analysis complete. Scout Score: {overall_score:.2f}")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {e}")
            return {
                "scout_score": 0.0,
                "error": str(e),
                "url": url,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def batch_analyze_content(self, content_list: List[Tuple[str, str]]) -> List[Dict]:
        """
        Batch process multiple content items for efficiency
        """
        logger.info(f"🔄 Starting batch analysis of {len(content_list)} items")
        
        results = []
        batch_size = 4  # Optimized for 8GB memory allocation
        
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]
            
            for content, url in batch:
                try:
                    analysis = self.comprehensive_content_analysis(content, url)
                    results.append(analysis)
                except Exception as e:
                    logger.error(f"❌ Batch analysis failed for {url}: {e}")
                    results.append({
                        "scout_score": 0.0,
                        "error": str(e),
                        "url": url,
                        "analysis_timestamp": datetime.utcnow().isoformat()
                    })
        
        logger.info(f"✅ Batch analysis complete. {len(results)} items processed")
        return results
    
    def _extract_json_response(self, text: str) -> Dict:
        """Extract JSON from model response"""
        try:
            # Find JSON content between { and }
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback parsing
                return {"error": "No valid JSON found in response", "raw_response": text}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {"error": f"JSON parsing failed: {str(e)}", "raw_response": text}
    
    def _calculate_scout_score(self, news_class: Dict, quality: Dict, bias: Dict) -> float:
        """Calculate overall Scout score for content filtering"""
        try:
            # Base news confidence
            news_score = news_class.get("confidence", 0.0) if news_class.get("is_news", False) else 0.0
            
            # Quality weighted score
            quality_score = quality.get("overall_quality", 0.0)
            
            # Bias penalty (high bias reduces score)
            bias_penalty = 1.0 - bias.get("bias_score", 0.5)
            
            # Combined Scout score
            scout_score = (news_score * 0.4 + quality_score * 0.4 + bias_penalty * 0.2)
            
            return min(max(scout_score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Scout score calculation error: {e}")
            return 0.0
    
    def _generate_recommendation(self, news_class: Dict, quality: Dict, bias: Dict, scout_score: float) -> str:
        """Generate recommendation for downstream processing"""
        if scout_score >= 0.8:
            return "HIGH_PRIORITY: Excellent news content for immediate processing"
        elif scout_score >= 0.6:
            return "MEDIUM_PRIORITY: Good quality news content"
        elif scout_score >= 0.4:
            return "LOW_PRIORITY: Borderline content, manual review recommended"
        else:
            return "REJECT: Poor quality or non-news content, exclude from pipeline"
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
