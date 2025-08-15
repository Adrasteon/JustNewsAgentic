#!/usr/bin/env python3
"""
🚀 JustNews V4 - Native Scout Agent with Enhanced Deep Crawl
Start Scout agent natively with BestFirstCrawlingStrategy integration
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for Scout agent"""
    logger.info("🔧 Setting up Scout agent environment...")
    
    # Set environment variables
    os.environ["PYTHONPATH"] = "/home/adra/JustNewsAgentic/agents/scout"
    os.environ["SCOUT_AGENT_PORT"] = "8002"
    os.environ["MCP_BUS_URL"] = "http://localhost:8000"
    os.environ["SCOUT_FEEDBACK_LOG"] = "./feedback_scout.log"
    
    # GPU configuration for Scout Intelligence
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TRANSFORMERS_CACHE"] = "/home/adra/.cache/huggingface"
    os.environ["HF_HOME"] = "/home/adra/.cache/huggingface"
    
    logger.info("✅ Environment configured for Scout agent")

def install_dependencies():
    """Install required dependencies for enhanced deep crawl"""
    logger.info("📦 Installing enhanced deep crawl dependencies...")
    
    try:
        # Install Crawl4AI if not available
        result = subprocess.run([
            sys.executable, "-c", "import crawl4ai; print('Crawl4AI available')"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.info("🔧 Installing Crawl4AI...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "crawl4ai>=0.7.0"
            ], check=True)
            logger.info("✅ Crawl4AI installed successfully")
        else:
            logger.info("✅ Crawl4AI already available")
            
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Could not install Crawl4AI: {e}")
        logger.info("Scout will use Docker fallback for deep crawling")

def start_scout_agent():
    """Start the Scout agent with enhanced deep crawl capability"""
    logger.info("🚀 Starting Scout Agent with Enhanced Deep Crawl...")
    
    # Change to Scout agent directory
    scout_dir = Path("/home/adra/JustNewsAgentic/agents/scout")
    os.chdir(scout_dir)
    
    # Start the Scout agent
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", "8002",
        "--reload"
    ]
    
    logger.info(f"📡 Starting Scout agent: {' '.join(cmd)}")
    logger.info("🧠 Enhanced Features:")
    logger.info("   ✅ BestFirstCrawlingStrategy deep crawling")
    logger.info("   ✅ Scout Intelligence analysis")
    logger.info("   ✅ User parameters: depth=3, pages=100, min_words=500")
    logger.info("   ✅ Native Crawl4AI integration with Docker fallback")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("\n🛑 Scout agent stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Scout agent failed to start: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🚀 JustNews V4 - Native Scout Agent Startup")
    print("Enhanced Deep Crawl with BestFirstCrawlingStrategy")
    print("=" * 80)
    
    try:
        setup_environment()
        install_dependencies()
        start_scout_agent()
    except Exception as e:
        logger.error(f"❌ Failed to start Scout agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
