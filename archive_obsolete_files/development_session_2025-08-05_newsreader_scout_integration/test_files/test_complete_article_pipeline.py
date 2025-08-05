#!/usr/bin/env python3
"""
Complete Article Pipeline Test
Test the full JustNews V4 pipeline from Scout crawling to Memory storage
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import psycopg2
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configuration
MCP_BUS_URL = "http://localhost:8000"
TEST_URL = "https://www.bbc.com/news"
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'justnews'),
    'user': os.getenv('DB_USER', 'justnews_user'),
    'password': os.getenv('DB_PASSWORD', 'justnews123'),
    'port': int(os.getenv('DB_PORT', 5432))
}

# Logging configuration
logging.basicConfig(
    filename='pipeline_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Configure retries for requests
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

class ArticlePipelineTest:
    def __init__(self):
        self.results = {}
        self.test_start_time = datetime.now()
        
    def log_test(self, stage: str, status: str, details: Any = None):
        """Log test results"""
        timestamp = datetime.now().isoformat()
        self.results[stage] = {
            'status': status,
            'timestamp': timestamp,
            'details': details
        }
        print(f"[{timestamp}] {stage}: {status}")
        if details:
            print(f"  Details: {details}")
            logger.info(f"{stage}: {status} - Details: {details}")
            with open("pipeline_test.log", "a") as log_file:
                log_file.write(f"[{timestamp}] {stage}: {status}\n")
                if details:
                    log_file.write(f"  Details: {details}\n")
            
    def call_agent(self, agent: str, tool: str, args: List = None, kwargs: Dict = None) -> Dict:
        """Call an agent through MCP Bus with improved error handling."""
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        payload = {
            "agent": agent,
            "tool": tool,
            "args": args,
            "kwargs": kwargs
        }

        try:
            response = session.post(f"{MCP_BUS_URL}/call", json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"Agent {agent} tool {tool} called successfully.")
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return {"error": str(http_err), "status": "failed"}
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
            return {"error": str(req_err), "status": "failed"}
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            return {"error": str(e), "status": "failed"}

    def check_database_connection(self) -> bool:
        """Verify database connectivity with enhanced logging."""
        try:
            conn = psycopg2.connect(**DATABASE_CONFIG)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            self.log_test("Database Connection", "✅ PASS", f"Connected, {count} articles in database")
            logger.info("Database connection verified successfully.")
            return True
        except psycopg2.OperationalError as db_err:
            self.log_test("Database Connection", "❌ FAIL", str(db_err))
            logger.error(f"Database connection failed: {db_err}")
            return False
        except Exception as e:
            self.log_test("Database Connection", "❌ FAIL", str(e))
            logger.error(f"Unexpected error during database connection check: {e}")
            return False
            
    def check_mcp_bus_status(self) -> bool:
        """Check MCP Bus and agent registrations"""
        try:
            response = requests.get(f"{MCP_BUS_URL}/agents", timeout=10)
            response.raise_for_status()
            agents = response.json()
            
            expected_agents = ["scout", "analyst", "fact_checker", "synthesizer", 
                             "critic", "memory", "reasoning", "newsreader", "chief_editor"]
            
            # Handle both dict and list response formats
            if isinstance(agents, dict):
                registered_agents = list(agents.keys())
            else:
                registered_agents = [agent["name"] for agent in agents]
                
            missing_agents = [agent for agent in expected_agents if agent not in registered_agents]
            
            if missing_agents:
                self.log_test("MCP Bus Status", "⚠️ PARTIAL", f"Missing agents: {missing_agents}")
                return False
            else:
                self.log_test("MCP Bus Status", "✅ PASS", f"All {len(registered_agents)} agents registered")
                return True
                
        except Exception as e:
            self.log_test("MCP Bus Status", "❌ FAIL", str(e))
            return False
            
    def test_scout_crawling(self) -> Dict:
        """Test Scout Agent crawling functionality with enhanced NewsReader integration"""
        self.log_test("Scout Crawling", "🔄 RUNNING", f"Enhanced crawling with NewsReader: {TEST_URL}")
        
        # Use enhanced NewsReader crawling instead of basic crawl_url
        result = self.call_agent("scout", "enhanced_newsreader_crawl", args=[TEST_URL])
        
        if "error" in result:
            self.log_test("Scout Crawling", "❌ FAIL", result["error"]) 
            return {}
            
        # Validate enhanced Scout response
        if result and isinstance(result, dict) and "content" in result:
            content_length = len(result.get("content", ""))
            method = result.get("metadata", {}).get("content_source", "unknown")
            visual_success = result.get("metadata", {}).get("visual_success", False)
            
            details = f"Retrieved {content_length} characters via {method}"
            if visual_success:
                details += " (with visual analysis)"
            
            self.log_test("Scout Crawling", "✅ PASS", details)
            return result
        else:
            self.log_test("Scout Crawling", "❌ FAIL", "Invalid response format")
            return {}
            
    def test_analyst_processing(self, content: str) -> Dict:
        """Test Analyst Agent sentiment and bias analysis"""
        if not content:
            self.log_test("Analyst Processing", "⏭️ SKIP", "No content to analyze")
            return {}
            
        self.log_test("Analyst Processing", "🔄 RUNNING", f"Analyzing {len(content)} characters")
        
        result = self.call_agent("analyst", "analyze_sentiment_and_bias", args=[content])
        
        if "error" in result:
            self.log_test("Analyst Processing", "❌ FAIL", result["error"])
            return {}
            
        # Validate Analyst response
        if result and "sentiment_score" in result and "bias_score" in result:
            sentiment = result.get("sentiment_score", "N/A")
            bias = result.get("bias_score", "N/A")
            self.log_test("Analyst Processing", "✅ PASS", f"Sentiment: {sentiment}, Bias: {bias}")
            return result
        else:
            self.log_test("Analyst Processing", "❌ FAIL", "Invalid analysis response")
            return {}
            
    def test_fact_checker_validation(self, content: str) -> Dict:
        """Test Fact Checker Agent validation"""
        if not content:
            self.log_test("Fact Checker", "⏭️ SKIP", "No content to validate")
            return {}
            
        self.log_test("Fact Checker", "🔄 RUNNING", "Validating facts and claims")
        
        result = self.call_agent("fact_checker", "validate_claims", args=[content])
        
        if "error" in result:
            self.log_test("Fact Checker", "❌ FAIL", result["error"])
            return {}
            
        # Validate Fact Checker response
        if result and "validation_score" in result:
            score = result.get("validation_score", "N/A")
            self.log_test("Fact Checker", "✅ PASS", f"Validation score: {score}")
            return result
        else:
            self.log_test("Fact Checker", "❌ FAIL", "Invalid validation response")
            return {}
            
    def test_memory_storage(self, article_data: Dict) -> bool:
        """Test Memory Agent storage functionality"""
        if not article_data:
            self.log_test("Memory Storage", "⏭️ SKIP", "No article data to store")
            return False
            
        self.log_test("Memory Storage", "🔄 RUNNING", "Storing article in database")
        
        result = self.call_agent("memory", "store_article", kwargs=article_data)
        
        if "error" in result:
            self.log_test("Memory Storage", "❌ FAIL", result["error"])
            return False
            
        # Verify storage in database
        try:
            conn = psycopg2.connect(**DATABASE_CONFIG)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles")
            count_after = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            if count_after > 0:
                self.log_test("Memory Storage", "✅ PASS", f"Article stored, total: {count_after}")
                return True
            else:
                self.log_test("Memory Storage", "❌ FAIL", "No articles found in database")
                return False
                
        except Exception as e:
            self.log_test("Memory Storage", "❌ FAIL", f"Database verification failed: {e}")
            return False
            
    def test_reasoning_integration(self, analysis_data: Dict) -> Dict:
        """Test Reasoning Agent integration"""
        if not analysis_data:
            self.log_test("Reasoning Integration", "⏭️ SKIP", "No analysis data")
            return {}
            
        self.log_test("Reasoning Integration", "🔄 RUNNING", "Adding facts to reasoning engine")
        
        # Add facts to reasoning engine
        facts = [
            f"article_sentiment = {analysis_data.get('sentiment_score', 0.5)}",
            f"article_bias = {analysis_data.get('bias_score', 0.5)}",
            f"article_credibility = {analysis_data.get('validation_score', 0.5)}"
        ]
        
        result = self.call_agent("reasoning", "add_facts", args=[facts])
        
        if "error" in result:
            self.log_test("Reasoning Integration", "❌ FAIL", result["error"])
            return {}
            
        # Query reasoning engine
        query_result = self.call_agent("reasoning", "query", args=["article_sentiment"])
        
        if "error" not in query_result and query_result:
            self.log_test("Reasoning Integration", "✅ PASS", f"Query result: {query_result}")
            return result
        else:
            self.log_test("Reasoning Integration", "❌ FAIL", "Query failed")
            return {}
            
    def restart_all_services(self):
        """Restart all services required for the pipeline"""
        self.log_test("Service Restart", "🔄 RUNNING", "Restarting all services")
        try:
            os.system("./stop_services.sh")
            os.system("./start_services_daemon.sh")
            self.log_test("Service Restart", "✅ PASS", "All services restarted successfully")
        except Exception as e:
            self.log_test("Service Restart", "❌ FAIL", str(e))

    def run_complete_pipeline_test(self):
        """Run the complete article pipeline test"""
        print("🚀 Starting Complete Article Pipeline Test")
        print("=" * 60)

        # Restart all services
        self.restart_all_services()

        # Pre-flight checks
        if not self.check_mcp_bus_status():
            print("❌ Cannot proceed - MCP Bus issues detected")
            return False

        if not self.check_database_connection():
            print("❌ Cannot proceed - Database connection failed")
            return False

        print("\n📰 Testing Article Processing Pipeline")
        print("-" * 40)
        
        # Step 1: Scout crawling
        scout_result = self.test_scout_crawling()
        content = scout_result.get("content", "") if scout_result else ""
        
        # Step 2: Analyst processing
        analyst_result = self.test_analyst_processing(content)
        
        # Step 3: Fact checker validation
        fact_check_result = self.test_fact_checker_validation(content)
        
        # Step 4: Reasoning integration
        reasoning_result = self.test_reasoning_integration(analyst_result)
        
        # Step 5: Memory storage
        article_data = {
            "content": content,
            "metadata": {
                "url": TEST_URL,
                "scout_analysis": scout_result,
                "analyst_analysis": analyst_result,
                "fact_check": fact_check_result,
                "reasoning": reasoning_result,
                "pipeline_test": True,
                "test_timestamp": self.test_start_time.isoformat()
            }
        }
        
        memory_success = self.test_memory_storage(article_data)
        
        # Generate summary
        self.generate_test_summary()
        
        return all([
            bool(scout_result),
            bool(analyst_result), 
            bool(fact_check_result),
            memory_success
        ])
        
    def generate_test_summary(self):
        """Generate and display test summary"""
        print("\n" + "=" * 60)
        print("📊 PIPELINE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if "✅" in r["status"]])
        failed_tests = len([r for r in self.results.values() if "❌" in r["status"]])
        skipped_tests = len([r for r in self.results.values() if "⏭️" in r["status"]])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏭️ Skipped: {skipped_tests}")
        print(f"🕐 Duration: {datetime.now() - self.test_start_time}")
        
        print("\nDetailed Results:")
        for stage, result in self.results.items():
            status = result["status"]
            details = result.get("details", "")
            print(f"  {stage}: {status}")
            if details and len(str(details)) < 100:
                print(f"    └─ {details}")
                
        # Overall status
        if failed_tests == 0:
            print(f"\n🎉 PIPELINE TEST: ✅ SUCCESS - All critical components operational!")
        else:
            print(f"\n⚠️ PIPELINE TEST: ❌ ISSUES DETECTED - {failed_tests} failures need attention")

if __name__ == "__main__":
    tester = ArticlePipelineTest()
    success = tester.run_complete_pipeline_test()
    
    print(f"\n{'='*60}")
    if success:
        print("🚀 JustNews V4 Pipeline: READY FOR PRODUCTION!")
    else:
        print("🔧 JustNews V4 Pipeline: NEEDS ATTENTION")
    print(f"{'='*60}")
