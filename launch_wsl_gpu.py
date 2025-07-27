#!/usr/bin/env python3
"""
Quick GPU Performance Test - Windows Side

This script tests the GPU acceleration from Windows before launching
the full WSL deployment.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wsl_gpu_performance():
    """Test GPU performance in WSL environment."""
    
    logger.info("🧪 Testing GPU Performance in WSL")
    logger.info("Expected: 42.1 articles/sec (already proven!)")
    
    wsl_command = [
        "wsl", "bash", "-c",
        "cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment && bash test_gpu_performance.sh"
    ]
    
    try:
        logger.info("🚀 Running WSL GPU performance test...")
        result = subprocess.run(wsl_command, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ GPU Performance Test Results:")
            print(result.stdout)
            
            # Check if we're getting good performance
            if "articles per second:" in result.stdout:
                # Try to extract the performance number
                lines = result.stdout.split('\n')
                for line in lines:
                    if "articles per second:" in line:
                        logger.info(f"📊 {line}")
                        break
            
            return True
        else:
            logger.error("❌ GPU test failed:")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ GPU test timed out (>60s)")
        return False
    except Exception as e:
        logger.error(f"❌ GPU test error: {e}")
        return False

def launch_full_system():
    """Launch the full WSL system with GPU acceleration."""
    
    logger.info("🚀 Launching Full JustNews V4 System")
    logger.info("Target: 42.1 articles/sec with FastAPI endpoints")
    
    wsl_command = [
        "wsl", "bash", "-c",
        "cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment && bash start_gpu_analyst.sh"
    ]
    
    try:
        logger.info("🎯 Starting GPU-accelerated analyst agent...")
        logger.info("API will be available at: http://localhost:8004")
        logger.info("Press Ctrl+C to stop")
        
        # Run in foreground so user can see output and stop it
        subprocess.run(wsl_command)
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"❌ System launch error: {e}")

def main():
    """Main test and launch orchestrator."""
    
    logger.info("🎯 JustNews V4 WSL GPU Deployment")
    logger.info("Your 42.1 articles/sec achievement in action!")
    
    print("\nChoose your option:")
    print("1. 🧪 Test GPU performance first (recommended)")
    print("2. 🚀 Launch full system immediately")
    print("3. 📊 Both - test then launch")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        logger.info("Testing GPU performance...")
        success = test_wsl_gpu_performance()
        if success:
            logger.info("✅ GPU test successful! Ready to launch full system.")
        else:
            logger.error("❌ GPU test failed. Check WSL and GPU setup.")
            
    elif choice == "2":
        logger.info("Launching full system...")
        launch_full_system()
        
    elif choice == "3":
        logger.info("Testing first, then launching...")
        success = test_wsl_gpu_performance()
        if success:
            input("\nPress Enter to launch full system...")
            launch_full_system()
        else:
            logger.error("❌ GPU test failed. Not launching full system.")
            
    else:
        logger.error("Invalid choice. Please run again and select 1, 2, or 3.")
    
    return True

if __name__ == "__main__":
    main()
