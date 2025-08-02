"""
Reasoning Agent (Nucleoid) - Production Implementation
Purpose: Symbolic reasoning, fact validation, contradiction detection, and explainability for news analysis
GPU Status: ❌ CPU Only (symbolic logic)
Performance: Fast for logic/rule queries; not GPU-accelerated
V4 Compliance: Designed for multi-agent orchestration, FastAPI, and MCP bus integration
Dependencies: git, networkx, fastapi, pydantic, uvicorn
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
import os
import sys
import subprocess
import tempfile
import json
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNucleoidImplementation:
    """Simple fallback implementation of Nucleoid for basic reasoning."""
    
    def __init__(self):
        self.facts = {}  # Store as key-value pairs
        self.rules = []
        
    def execute(self, statement):
        """Execute a Nucleoid statement."""
        statement = statement.strip()
        
        # Handle variable assignments (facts) - simple assignments only
        if "=" in statement and not "==" in statement and not any(op in statement for op in ["+", "-", "*", "/", "if", "then"]):
            parts = statement.split("=")
            if len(parts) == 2:
                var_name = parts[0].strip()
                value = parts[1].strip()
                
                # Try to convert to number if possible
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string if not a number
                    value = value.strip("\"'")
                
                self.facts[var_name] = value
                return {"success": True, "message": f"Variable {var_name} set to {value}"}
        
        # Handle rule definitions (y = x + 10, if-then statements)
        if "=" in statement and (any(op in statement for op in ["+", "-", "*", "/"]) or "if" in statement or "then" in statement):
            self.rules.append(statement)
            return {"success": True, "message": "Rule added"}
        
        # Handle queries (single variable)
        if statement.isalpha() or statement.replace("_", "").isalpha():
            # Check if it's a direct fact
            if statement in self.facts:
                return self.facts[statement]
            
            # Try to evaluate using rules
            for rule in self.rules:
                if "=" in rule and rule.split("=")[0].strip() == statement:
                    right_side = rule.split("=")[1].strip()
                    try:
                        # Simple expression evaluation
                        result = self._evaluate_expression(right_side)
                        if result is not None:
                            return result
                    except:
                        pass
            
            return {"success": False, "message": f"Unknown variable: {statement}"}
        
        # Handle boolean queries (==, !=, etc.)
        if any(op in statement for op in ["==", "!=", ">", "<", ">=", "<="]):
            try:
                return self._evaluate_boolean(statement)
            except:
                return {"success": False, "message": "Could not evaluate boolean expression"}
        
        return {"success": False, "message": "Unknown statement type"}
    
    def _evaluate_expression(self, expression):
        """Evaluate a simple mathematical expression."""
        try:
            # Replace variables with their values
            for var, val in self.facts.items():
                expression = expression.replace(var, str(val))
            
            # Simple evaluation (be careful in production!)
            result = eval(expression)
            return result
        except:
            return None
    
    def _evaluate_boolean(self, statement):
        """Evaluate a boolean statement."""
        try:
            # Replace variables with their values
            for var, val in self.facts.items():
                if isinstance(val, str):
                    statement = statement.replace(var, f"'{val}'")
                else:
                    statement = statement.replace(var, str(val))
            
            # Evaluate the boolean expression
            result = eval(statement)
            return result
        except:
            return False
    
    def run(self, statement):
        """Alias for execute method to match expected interface."""
        return self.execute(statement)
    
    def clear(self):
        """Clear all facts and rules."""
        self.facts = {}
        self.rules = []
        return {"success": True, "message": "Knowledge base cleared"}

app = FastAPI(title="JustNews V4 Reasoning Agent (Nucleoid)")

# --- Nucleoid GitHub Integration ---
class NucleoidEngine:
    """Wrapper for Nucleoid GitHub Python implementation."""
    
    def __init__(self):
        self.nucleoid = None
        self.facts_store = {}  # Store facts for retrieval
        self.rules_store = []  # Store rules for retrieval
        self.session_id = "reasoning_session"
        self._setup_nucleoid()
    
    def _setup_nucleoid(self):
        """Setup Nucleoid Python implementation from our local implementation."""
        try:
            # Try to use our complete local implementation first
            from nucleoid_implementation import Nucleoid
            self.nucleoid = Nucleoid()
            logger.info("✅ Complete Nucleoid implementation loaded successfully")
            return
            
        except ImportError as e:
            logger.warning(f"Local implementation failed: {e}, trying GitHub repository...")
            
            # Fallback to GitHub repository approach
            nucleoid_dir = Path(__file__).parent / "nucleoid_repo"
            
            if not nucleoid_dir.exists():
                logger.info("Cloning Nucleoid repository...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/nucleoidai/nucleoid.git", 
                    str(nucleoid_dir)
                ], check=True, capture_output=True)
                logger.info("✅ Nucleoid repository cloned successfully")
            
            # Add Python path for the nucleoid module
            python_path = str(nucleoid_dir / "python")
            if python_path not in sys.path:
                sys.path.insert(0, python_path)
            
            try:
                # Import the Nucleoid class using proper module path
                from nucleoid.nucleoid import Nucleoid
                self.nucleoid = Nucleoid()
                logger.info("✅ Nucleoid GitHub implementation loaded successfully")
                return
                
            except ImportError as e:
                logger.warning(f"GitHub implementation import failed: {e}, using fallback...")
                
                # Final fallback
                self.nucleoid = SimpleNucleoidImplementation()
                logger.info("✅ Simple Nucleoid fallback implementation loaded")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone Nucleoid repository: {e}")
            # Use fallback implementation
            self.nucleoid = SimpleNucleoidImplementation()
            logger.warning("Using fallback simple implementation")
        except Exception as e:
            logger.error(f"Unexpected error setting up Nucleoid: {e}")
            # Use fallback implementation
            self.nucleoid = SimpleNucleoidImplementation()
            logger.warning("Using fallback simple implementation")
    
    def add_fact(self, fact_data: Dict[str, Any]) -> Any:
        """Add a fact to the reasoning system."""
        try:
            # Convert fact to Nucleoid statement
            fact_id = f"fact_{len(self.facts_store)}"
            
            # Store fact for retrieval
            self.facts_store[fact_id] = fact_data
            
            # Create Nucleoid statement
            if "statement" in fact_data:
                # Direct statement execution
                result = self.nucleoid.run(fact_data["statement"])
            else:
                # Convert dict to variable assignments
                statements = []
                for key, value in fact_data.items():
                    if isinstance(value, str):
                        statements.append(f'{key} = "{value}"')
                    else:
                        statements.append(f'{key} = {value}')
                
                result = None
                for statement in statements:
                    result = self.nucleoid.run(statement)
            
            logger.info(f"Added fact {fact_id}: {fact_data}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            raise
    
    def add_rule(self, rule: str) -> Any:
        """Add a logical rule."""
        try:
            # Store rule for retrieval
            self.rules_store.append(rule)
            
            # Execute rule in Nucleoid
            result = self.nucleoid.run(rule)
            
            logger.info(f"Added rule: {rule}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
            raise
    
    def query(self, query_str: str) -> Any:
        """Execute a symbolic reasoning query."""
        try:
            result = self.nucleoid.run(query_str)
            logger.info(f"Executed query: {query_str} -> {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_facts(self) -> Dict[str, Any]:
        """Retrieve all stored facts."""
        return self.facts_store
    
    def get_rules(self) -> List[str]:
        """Retrieve all stored rules."""
        return self.rules_store
    
    def evaluate_contradiction(self, statements: List[str]) -> Dict[str, Any]:
        """Check for logical contradictions between statements."""
        try:
            contradictions = []
            
            # Extract variable assignments and check for direct contradictions
            variable_assignments = {}
            
            for stmt in statements:
                # Check for direct variable assignments (x = 5, x = 10)
                if "=" in stmt and not "==" in stmt and not any(op in stmt for op in ["+", "-", "*", "/", "if", "then", ">", "<"]):
                    parts = stmt.split("=")
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Try to convert to number for comparison
                        try:
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            value = value.strip("\"'")
                        
                        if var_name in variable_assignments:
                            if variable_assignments[var_name] != value:
                                contradictions.append({
                                    "statement1": f"{var_name} = {variable_assignments[var_name]}",
                                    "statement2": f"{var_name} = {value}",
                                    "conflict": "variable_reassignment_contradiction"
                                })
                        else:
                            variable_assignments[var_name] = value
            
            # Check for boolean contradictions (x == 5 vs x == 10)
            boolean_statements = [stmt for stmt in statements if any(op in stmt for op in ["==", "!=", ">", "<", ">=", "<="])]
            
            for i, stmt1 in enumerate(boolean_statements):
                for j, stmt2 in enumerate(boolean_statements[i+1:], i+1):
                    # Extract variable and values from boolean statements
                    try:
                        if self._are_contradictory_booleans(stmt1, stmt2):
                            contradictions.append({
                                "statement1": stmt1,
                                "statement2": stmt2,
                                "conflict": "boolean_contradiction"
                            })
                    except:
                        pass  # Skip if can't parse
            
            return {
                "has_contradictions": len(contradictions) > 0,
                "contradictions": contradictions,
                "total_statements": len(statements)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating contradictions: {e}")
            return {
                "has_contradictions": False,
                "contradictions": [],
                "total_statements": len(statements)
            }
    
    def _are_contradictory_booleans(self, stmt1: str, stmt2: str) -> bool:
        """Check if two boolean statements are contradictory."""
        # Simple check for directly contradictory statements
        # e.g., "temperature == 25" vs "temperature == 30"
        
        # Extract variable and operator for each statement
        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if op in stmt1 and op in stmt2:
                parts1 = stmt1.split(op)
                parts2 = stmt2.split(op)
                
                if len(parts1) == 2 and len(parts2) == 2:
                    var1, val1 = parts1[0].strip(), parts1[1].strip()
                    var2, val2 = parts2[0].strip(), parts2[1].strip()
                    
                    # Same variable, same operator, different values
                    if var1 == var2 and op == "==" and val1 != val2:
                        return True
        
        return False

# Initialize Nucleoid engine
engine = None
try:
    engine = NucleoidEngine()
    logger.info("✅ Reasoning Agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Reasoning Agent: {e}")
    engine = None

# --- Pydantic Models ---
class ToolCall(BaseModel):
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}

class Fact(BaseModel):
    data: Dict[str, Any]

class Facts(BaseModel):
    facts: List[Dict[str, Any]]

class Rule(BaseModel):
    rule: str

class Query(BaseModel):
    query: str

class Evaluate(BaseModel):
    expression: str

class ContradictionCheck(BaseModel):
    statements: List[str]

class FactValidation(BaseModel):
    claim: str
    context: Optional[Dict[str, Any]] = None

# --- Utility Functions ---
def log_feedback(event: str, details: Dict[str, Any]):
    """Log feedback for debugging and improvement."""
    feedback_log = Path(__file__).parent / "feedback_reasoning.log"
    try:
        with open(feedback_log, "a", encoding="utf-8") as f:
            timestamp = datetime.utcnow().isoformat()
            f.write(f"{timestamp}\t{event}\t{json.dumps(details)}\n")
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")

# --- API Endpoints ---
@app.post("/add_fact")
def add_fact_endpoint(call: ToolCall):
    """Add a fact to the reasoning system."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    try:
        # Extract fact from args or kwargs
        if call.args:
            fact_data = call.args[0] if isinstance(call.args[0], dict) else {"statement": str(call.args[0])}
        else:
            fact_data = call.kwargs.get("fact", call.kwargs.get("data", {}))
        
        result = engine.add_fact(fact_data)
        
        # Log feedback
        log_feedback("add_fact", {
            "fact_data": fact_data,
            "result": str(result),
            "success": True
        })
        
        return {"success": True, "result": result, "fact_id": len(engine.facts_store)}
        
    except Exception as e:
        error_msg = str(e)
        log_feedback("add_fact_error", {
            "fact_data": call.args if call.args else call.kwargs,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/add_facts")
def add_facts_endpoint(call: ToolCall):
    """Add multiple facts to the reasoning system."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    try:
        # Extract facts from args or kwargs
        if call.args and isinstance(call.args[0], list):
            facts_list = call.args[0]
        else:
            facts_list = call.kwargs.get("facts", [])
        
        results = []
        for fact_data in facts_list:
            if isinstance(fact_data, dict):
                result = engine.add_fact(fact_data)
                results.append(result)
            else:
                result = engine.add_fact({"statement": str(fact_data)})
                results.append(result)
        
        # Log feedback
        log_feedback("add_facts", {
            "facts_count": len(facts_list),
            "results": [str(r) for r in results],
            "success": True
        })
        
        return {"success": True, "count": len(results), "results": results}
        
    except Exception as e:
        error_msg = str(e)
        log_feedback("add_facts_error", {
            "facts_data": call.args if call.args else call.kwargs,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/add_rule")
def add_rule_endpoint(call: ToolCall):
    """Add a logical rule."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    try:
        # Extract rule from args or kwargs
        if call.args:
            rule = str(call.args[0])
        else:
            rule = call.kwargs.get("rule", "")
        
        if not rule:
            raise ValueError("Rule cannot be empty")
        
        result = engine.add_rule(rule)
        
        # Log feedback
        log_feedback("add_rule", {
            "rule": rule,
            "result": str(result),
            "success": True
        })
        
        return {"success": True, "result": result, "rule_count": len(engine.rules_store)}
        
    except Exception as e:
        error_msg = str(e)
        log_feedback("add_rule_error", {
            "rule_data": call.args if call.args else call.kwargs,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/query")
def query_endpoint(call: ToolCall):
    """Execute a symbolic reasoning query."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    try:
        # Extract query from args or kwargs
        if call.args:
            query = str(call.args[0])
        else:
            query = call.kwargs.get("query", "")
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        result = engine.query(query)
        
        # Log feedback
        log_feedback("query", {
            "query": query,
            "result": str(result),
            "success": True
        })
        
        return {"success": True, "result": result, "query": query}
        
    except Exception as e:
        error_msg = str(e)
        log_feedback("query_error", {
            "query_data": call.args if call.args else call.kwargs,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/evaluate")
def evaluate_endpoint(call: ToolCall):
    """Evaluate contradictions and logical consistency."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    try:
        # Extract evaluation request
        if call.args and isinstance(call.args[0], list):
            statements = call.args[0]
        elif call.args:
            statements = [str(call.args[0])]
        else:
            statements = call.kwargs.get("statements", [])
            if not statements:
                # Evaluate all current facts and rules
                statements = list(engine.facts_store.values()) + engine.rules_store
        
        if not statements:
            return {"success": True, "result": "No statements to evaluate"}
        
        # Convert non-string statements to strings
        str_statements = []
        for stmt in statements:
            if isinstance(stmt, dict):
                if "statement" in stmt:
                    str_statements.append(stmt["statement"])
                else:
                    str_statements.append(json.dumps(stmt))
            else:
                str_statements.append(str(stmt))
        
        result = engine.evaluate_contradiction(str_statements)
        
        # Log feedback
        log_feedback("evaluate", {
            "statements_count": len(str_statements),
            "has_contradictions": result.get("has_contradictions", False),
            "contradictions_count": len(result.get("contradictions", [])),
            "success": True
        })
        
        return {"success": True, "result": result}
        
    except Exception as e:
        error_msg = str(e)
        log_feedback("evaluate_error", {
            "evaluation_data": call.args if call.args else call.kwargs,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/facts")
def get_facts():
    """Retrieve all stored facts."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    return {"facts": engine.get_facts(), "count": len(engine.facts_store)}

@app.get("/rules")
def get_rules():
    """Retrieve all stored rules."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    return {"rules": engine.get_rules(), "count": len(engine.rules_store)}

@app.get("/status")
def get_status():
    """Get reasoning engine status."""
    if not engine:
        return {
            "status": "unavailable",
            "nucleoid_available": False,
            "facts_count": 0,
            "rules_count": 0
        }
    
    return {
        "status": "ok",
        "nucleoid_available": True,
        "facts_count": len(engine.facts_store),
        "rules_count": len(engine.rules_store),
        "session_id": engine.session_id
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    status = "ok" if engine else "unavailable"
    return {"status": status, "nucleoid_available": engine is not None}

# --- MCP Bus Integration ---
@app.post("/call")
def call_tool(request: Dict[str, Any]):
    """MCP bus integration - handles tool calls from other agents."""
    try:
        tool = request.get("tool", "")
        args = request.get("args", [])
        kwargs = request.get("kwargs", {})
        
        # Create ToolCall object
        call = ToolCall(args=args, kwargs=kwargs)
        
        # Route to appropriate endpoint
        if tool == "add_fact":
            return add_fact_endpoint(call)
        elif tool == "add_facts":
            return add_facts_endpoint(call)
        elif tool == "add_rule":
            return add_rule_endpoint(call)
        elif tool == "query":
            return query_endpoint(call)
        elif tool == "evaluate":
            return evaluate_endpoint(call)
        else:
            available_tools = ["add_fact", "add_facts", "add_rule", "query", "evaluate"]
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown tool: {tool}. Available tools: {available_tools}"
            )
    
    except Exception as e:
        log_feedback("mcp_call_error", {
            "request": request,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

# --- News Analysis Specific Functions ---
@app.post("/validate_claim")
def validate_claim(call: ToolCall):
    """Validate a news claim against known facts and rules."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    try:
        # Extract claim
        if call.args:
            claim = str(call.args[0])
            context = call.args[1] if len(call.args) > 1 else {}
        else:
            claim = call.kwargs.get("claim", "")
            context = call.kwargs.get("context", {})
        
        if not claim:
            raise ValueError("Claim cannot be empty")
        
        # Add claim as temporary fact and check for contradictions
        temp_fact = {"statement": claim, "type": "claim", "context": context}
        
        # Get existing statements
        existing_statements = []
        for fact in engine.facts_store.values():
            if isinstance(fact, dict) and "statement" in fact:
                existing_statements.append(fact["statement"])
        existing_statements.extend(engine.rules_store)
        
        # Add the claim and check for contradictions
        test_statements = existing_statements + [claim]
        contradiction_result = engine.evaluate_contradiction(test_statements)
        
        # Determine validation result
        validation_result = {
            "claim": claim,
            "context": context,
            "valid": not contradiction_result["has_contradictions"],
            "contradictions": contradiction_result["contradictions"],
            "confidence": 1.0 - (len(contradiction_result["contradictions"]) * 0.2)
        }
        
        # Log feedback
        log_feedback("validate_claim", {
            "claim": claim,
            "valid": validation_result["valid"],
            "contradictions_count": len(contradiction_result["contradictions"]),
            "confidence": validation_result["confidence"]
        })
        
        return {"success": True, "result": validation_result}
        
    except Exception as e:
        error_msg = str(e)
        log_feedback("validate_claim_error", {
            "claim_data": call.args if call.args else call.kwargs,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/explain_reasoning")
def explain_reasoning(call: ToolCall):
    """Provide explainable reasoning for a query or validation."""
    if not engine:
        raise HTTPException(status_code=503, detail="Nucleoid engine not available")
    
    try:
        # Extract query
        if call.args:
            query = str(call.args[0])
        else:
            query = call.kwargs.get("query", "")
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Execute query and provide explanation
        result = engine.query(query)
        
        # Generate explanation
        explanation = {
            "query": query,
            "result": result,
            "reasoning_steps": [
                f"1. Executed query: '{query}'",
                f"2. Applied {len(engine.rules_store)} logical rules",
                f"3. Checked against {len(engine.facts_store)} known facts",
                f"4. Result: {result}"
            ],
            "facts_used": list(engine.facts_store.keys()),
            "rules_applied": engine.rules_store,
            "confidence": 0.8 if result else 0.2
        }
        
        # Log feedback
        log_feedback("explain_reasoning", {
            "query": query,
            "result": str(result),
            "explanation_provided": True
        })
        
        return {"success": True, "result": explanation}
        
    except Exception as e:
        error_msg = str(e)
        log_feedback("explain_reasoning_error", {
            "query_data": call.args if call.args else call.kwargs,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=error_msg)

# --- Startup and Shutdown Events ---
@app.on_event("startup")
def startup_event():
    """Initialize reasoning agent on startup."""
    logger.info("🧠 Reasoning Agent starting up...")
    
    # Load preloaded rules if available
    preload_file = os.environ.get("REASONING_AGENT_PRELOAD")
    if preload_file and Path(preload_file).exists():
        logger.info(f"Loading preloaded rules from {preload_file}")
        try:
            with open(preload_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("RULE:"):
                        engine.add_rule(line[5:].strip())
                    elif line.startswith("FACT:"):
                        fact_data = json.loads(line[5:].strip())
                        engine.add_fact(fact_data)
            logger.info("✅ Preloaded rules and facts loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load preloaded rules: {e}")
    
    # Load news domain rules
    _load_news_domain_rules()
    
    logger.info("✅ Reasoning Agent startup complete")

def _load_news_domain_rules():
    """Load domain-specific rules for news analysis."""
    if not engine:
        return
    
    try:
        # Basic news validation rules
        news_rules = [
            # Source credibility rules
            "if source_type == 'government' then credibility = 0.9",
            "if source_type == 'academic' then credibility = 0.8",
            "if source_type == 'news_agency' then credibility = 0.7",
            "if source_type == 'blog' then credibility = 0.3",
            
            # Temporal consistency rules
            "if event_date > current_date then validity = false",
            "if publication_date < event_date then temporal_flag = true",
            
            # Basic fact checking rules
            "if claim_type == 'statistical' and source_provided == false then verify_needed = true",
            "if claim_type == 'quote' and source_provided == false then verify_needed = true"
        ]
        
        for rule in news_rules:
            try:
                engine.add_rule(rule)
            except Exception as e:
                logger.warning(f"Failed to load rule '{rule}': {e}")
        
        logger.info(f"✅ Loaded {len(news_rules)} news domain rules")
        
    except Exception as e:
        logger.warning(f"Failed to load news domain rules: {e}")

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🧠 Reasoning Agent shutting down...")
    
    # Save current state if needed
    try:
        state_file = Path(__file__).parent / "reasoning_state.json"
        state_data = {
            "facts": engine.get_facts() if engine else {},
            "rules": engine.get_rules() if engine else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"✅ Reasoning state saved to {state_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save reasoning state: {e}")
    
    logger.info("✅ Reasoning Agent shutdown complete")

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("REASONING_AGENT_PORT", 8008))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
