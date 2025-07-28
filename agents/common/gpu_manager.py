"""
Multi-Agent GPU Manager for JustNews V4
Manages GPU memory allocation across multiple agents to prevent crashes

Based on proven patterns from GPUAcceleratedAnalyst:
- Professional memory management preventing system crashes  
- Dynamic allocation based on agent requirements
- Intelligent fallback to CPU when GPU memory is exhausted
- Performance monitoring across all GPU agents
"""

import os
import logging
import torch
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("justnews.gpu_manager")

class AgentType(Enum):
    ANALYST = "analyst"
    FACT_CHECKER = "fact_checker"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    SCOUT = "scout"
    CHIEF_EDITOR = "chief_editor"
    MEMORY = "memory"

@dataclass
class AgentGPURequirements:
    agent_type: AgentType
    base_memory_gb: float
    peak_memory_gb: float
    models: List[str]
    priority: int  # 1=highest, 5=lowest
    batch_size: int
    
class GPUAllocationStatus(Enum):
    ALLOCATED = "allocated"
    QUEUED = "queued"
    FAILED = "failed"
    RELEASED = "released"

@dataclass
class AgentAllocation:
    agent_id: str
    agent_type: AgentType
    allocated_memory_gb: float
    gpu_device: int
    status: GPUAllocationStatus
    allocated_at: datetime
    last_activity: datetime

class MultiAgentGPUManager:
    """
    Professional GPU memory management for JustNews V4 multi-agent system
    
    Features:
    - RTX 3090 24GB VRAM optimal allocation
    - Priority-based agent scheduling  
    - Crash prevention through memory monitoring
    - Dynamic fallback to CPU when GPU exhausted
    - Performance tracking across all agents
    """
    
    def __init__(self):
        self.total_vram_gb = 24.0  # RTX 3090
        self.system_reserved_gb = 2.0  # OS + CUDA overhead
        self.available_vram_gb = self.total_vram_gb - self.system_reserved_gb
        
        # Agent allocation tracking
        self.allocations: Dict[str, AgentAllocation] = {}
        self.allocation_queue: List[str] = []
        
        # Performance tracking
        self.performance_stats = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'cpu_fallbacks': 0,
            'memory_warnings': 0,
            'peak_usage_gb': 0.0
        }
        
        # Agent requirements (based on proven analyst pattern)
        self.agent_requirements = {
            AgentType.ANALYST: AgentGPURequirements(
                agent_type=AgentType.ANALYST,
                base_memory_gb=6.0,  # Current proven allocation
                peak_memory_gb=8.0,
                models=["cardiffnlp/twitter-roberta-base-sentiment-latest", "HuggingFace models"],
                priority=1,  # Highest priority (already proven)
                batch_size=32
            ),
            AgentType.FACT_CHECKER: AgentGPURequirements(
                agent_type=AgentType.FACT_CHECKER,
                base_memory_gb=4.0,  # DialoGPT-large (774M params)
                peak_memory_gb=5.5,
                models=["microsoft/DialoGPT-large", "facebook/bart-large-mnli"],
                priority=2,  # High priority for critical fact checking
                batch_size=4
            ),
            AgentType.SYNTHESIZER: AgentGPURequirements(
                agent_type=AgentType.SYNTHESIZER,
                base_memory_gb=6.0,  # Multiple models: sentence-transformers + clustering
                peak_memory_gb=8.0,
                models=["sentence-transformers/all-MiniLM-L6-v2", "clustering models"],
                priority=3,  # Medium-high priority
                batch_size=16
            ),
            AgentType.CRITIC: AgentGPURequirements(
                agent_type=AgentType.CRITIC,
                base_memory_gb=4.0,  # DialoGPT-medium (355M params)
                peak_memory_gb=5.0,
                models=["microsoft/DialoGPT-medium"],
                priority=3,  # Medium-high priority
                batch_size=8
            ),
            AgentType.SCOUT: AgentGPURequirements(
                agent_type=AgentType.SCOUT,
                base_memory_gb=8.0,  # LLaMA-3-8B (larger model)
                peak_memory_gb=10.0,
                models=["meta-llama/Llama-3-8B-Instruct"],
                priority=4,  # Medium priority
                batch_size=2
            ),
            AgentType.CHIEF_EDITOR: AgentGPURequirements(
                agent_type=AgentType.CHIEF_EDITOR,
                base_memory_gb=12.0,  # LLaMA-3-70B (very large model)
                peak_memory_gb=16.0,
                models=["meta-llama/Llama-3-70B-Instruct"],
                priority=5,  # Lower priority due to size
                batch_size=1
            ),
            AgentType.MEMORY: AgentGPURequirements(
                agent_type=AgentType.MEMORY,
                base_memory_gb=3.0,  # Vector embeddings + search
                peak_memory_gb=4.0,
                models=["sentence-transformers/all-MiniLM-L6-v2"],
                priority=4,  # Medium priority
                batch_size=64
            )
        }
        
        logger.info("🚀 Multi-Agent GPU Manager initialized")
        logger.info(f"   Total VRAM: {self.total_vram_gb}GB")
        logger.info(f"   Available for agents: {self.available_vram_gb}GB")
        logger.info(f"   Supported agents: {len(self.agent_requirements)}")
        
        self._validate_gpu_availability()
    
    def _validate_gpu_availability(self):
        """Validate GPU is available and get actual memory info"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                actual_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"✅ GPU Available: {gpu_name}")
                logger.info(f"✅ Actual GPU Memory: {actual_memory:.1f}GB")
                
                # Update actual memory if different from expected
                if abs(actual_memory - self.total_vram_gb) > 1.0:
                    logger.warning(f"⚠️ Expected {self.total_vram_gb}GB, got {actual_memory:.1f}GB")
                    self.total_vram_gb = actual_memory
                    self.available_vram_gb = actual_memory - self.system_reserved_gb
            else:
                logger.warning("⚠️ GPU not available - all agents will use CPU fallback")
                
        except Exception as e:
            logger.error(f"❌ GPU validation failed: {e}")
    
    def request_gpu_allocation(self, agent_id: str, agent_type: AgentType) -> Dict[str, Any]:
        """
        Request GPU memory allocation for an agent
        
        Returns:
            Dict with allocation status, memory info, and fallback instructions
        """
        try:
            # Check if agent already has allocation
            if agent_id in self.allocations:
                existing = self.allocations[agent_id]
                logger.info(f"Agent {agent_id} already has GPU allocation: {existing.allocated_memory_gb}GB")
                return {
                    "status": "already_allocated",
                    "allocated_memory_gb": existing.allocated_memory_gb,
                    "gpu_device": existing.gpu_device,
                    "message": "Agent already has GPU allocation"
                }
            
            # Get requirements for this agent type
            if agent_type not in self.agent_requirements:
                logger.error(f"Unknown agent type: {agent_type}")
                return self._create_cpu_fallback_response(agent_id, "Unknown agent type")
            
            requirements = self.agent_requirements[agent_type]
            
            # Check if we have enough memory
            current_usage = self._get_current_memory_usage()
            needed_memory = requirements.base_memory_gb
            
            if current_usage + needed_memory > self.available_vram_gb:
                logger.warning(f"Insufficient GPU memory for {agent_id}: need {needed_memory}GB, available {self.available_vram_gb - current_usage}GB")
                
                # Try to free memory from lower priority agents
                if self._try_free_memory_for_priority(requirements.priority, needed_memory):
                    logger.info(f"Freed memory for higher priority agent {agent_id}")
                else:
                    return self._create_cpu_fallback_response(agent_id, "Insufficient GPU memory")
            
            # Allocate GPU memory
            allocation = AgentAllocation(
                agent_id=agent_id,
                agent_type=agent_type,
                allocated_memory_gb=needed_memory,
                gpu_device=0,  # RTX 3090 is single GPU
                status=GPUAllocationStatus.ALLOCATED,
                allocated_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.allocations[agent_id] = allocation
            
            # Update performance stats
            self.performance_stats['total_allocations'] += 1
            self.performance_stats['successful_allocations'] += 1
            self.performance_stats['peak_usage_gb'] = max(
                self.performance_stats['peak_usage_gb'],
                current_usage + needed_memory
            )
            
            logger.info(f"✅ GPU allocated to {agent_id}: {needed_memory}GB")
            logger.info(f"   Total GPU usage: {current_usage + needed_memory:.1f}GB / {self.available_vram_gb}GB")
            
            return {
                "status": "allocated",
                "allocated_memory_gb": needed_memory,
                "gpu_device": 0,
                "batch_size": requirements.batch_size,
                "models": requirements.models,
                "torch_dtype": "float16",  # Memory optimization
                "message": f"GPU memory allocated: {needed_memory}GB"
            }
            
        except Exception as e:
            logger.error(f"❌ Error allocating GPU for {agent_id}: {e}")
            self.performance_stats['failed_allocations'] += 1
            return self._create_cpu_fallback_response(agent_id, f"Allocation error: {e}")
    
    def release_gpu_allocation(self, agent_id: str) -> bool:
        """Release GPU allocation for an agent"""
        try:
            if agent_id not in self.allocations:
                logger.warning(f"Agent {agent_id} has no GPU allocation to release")
                return False
            
            allocation = self.allocations[agent_id]
            allocation.status = GPUAllocationStatus.RELEASED
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            del self.allocations[agent_id]
            
            logger.info(f"✅ Released GPU allocation for {agent_id}: {allocation.allocated_memory_gb}GB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error releasing GPU for {agent_id}: {e}")
            return False
    
    def update_agent_activity(self, agent_id: str):
        """Update last activity timestamp for agent (for memory management)"""
        if agent_id in self.allocations:
            self.allocations[agent_id].last_activity = datetime.now()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU allocation status"""
        current_usage = self._get_current_memory_usage()
        
        return {
            "total_vram_gb": self.total_vram_gb,
            "available_vram_gb": self.available_vram_gb,
            "current_usage_gb": current_usage,
            "free_memory_gb": self.available_vram_gb - current_usage,
            "utilization_percent": (current_usage / self.available_vram_gb) * 100,
            "active_allocations": len(self.allocations),
            "allocated_agents": list(self.allocations.keys()),
            "performance_stats": self.performance_stats,
            "gpu_available": torch.cuda.is_available() if torch is not None else False
        }
    
    def get_allocation_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for optimal agent deployment"""
        recommendations = {
            "high_priority_agents": [],
            "optimal_deployment": [],
            "memory_warnings": []
        }
        
        # Sort agents by priority and memory requirements
        sorted_agents = sorted(
            self.agent_requirements.items(),
            key=lambda x: (x[1].priority, x[1].base_memory_gb)
        )
        
        total_memory = 0
        for agent_type, req in sorted_agents:
            if total_memory + req.base_memory_gb <= self.available_vram_gb:
                recommendations["optimal_deployment"].append({
                    "agent_type": agent_type.value,
                    "memory_gb": req.base_memory_gb,
                    "priority": req.priority,
                    "models": req.models
                })
                total_memory += req.base_memory_gb
                
                if req.priority <= 2:
                    recommendations["high_priority_agents"].append(agent_type.value)
            else:
                recommendations["memory_warnings"].append({
                    "agent_type": agent_type.value,
                    "required_memory_gb": req.base_memory_gb,
                    "available_memory_gb": self.available_vram_gb - total_memory,
                    "recommendation": "Deploy with CPU fallback"
                })
        
        recommendations["total_optimal_agents"] = len(recommendations["optimal_deployment"])
        recommendations["total_memory_used_gb"] = total_memory
        
        return recommendations
    
    def _get_current_memory_usage(self) -> float:
        """Calculate current GPU memory usage from allocations"""
        return sum(alloc.allocated_memory_gb for alloc in self.allocations.values())
    
    def _try_free_memory_for_priority(self, required_priority: int, needed_memory: float) -> bool:
        """Try to free memory by releasing lower priority agents"""
        # Find lower priority agents that could be released
        candidates = [
            (agent_id, alloc) for agent_id, alloc in self.allocations.items()
            if self.agent_requirements[alloc.agent_type].priority > required_priority
        ]
        
        # Sort by priority (highest priority number = lowest priority)
        candidates.sort(key=lambda x: self.agent_requirements[x[1].agent_type].priority, reverse=True)
        
        freed_memory = 0
        for agent_id, allocation in candidates:
            if freed_memory >= needed_memory:
                break
                
            logger.info(f"Releasing lower priority agent {agent_id} to make room")
            if self.release_gpu_allocation(agent_id):
                freed_memory += allocation.allocated_memory_gb
        
        return freed_memory >= needed_memory
    
    def _create_cpu_fallback_response(self, agent_id: str, reason: str) -> Dict[str, Any]:
        """Create response for CPU fallback deployment"""
        self.performance_stats['cpu_fallbacks'] += 1
        
        logger.info(f"Agent {agent_id} will use CPU fallback: {reason}")
        
        return {
            "status": "cpu_fallback",
            "allocated_memory_gb": 0,
            "gpu_device": -1,  # -1 indicates CPU
            "reason": reason,
            "message": f"Using CPU fallback due to: {reason}",
            "fallback_config": {
                "device": -1,
                "batch_size": 1,  # Smaller batches for CPU
                "torch_dtype": "float32"  # CPU doesn't benefit from float16
            }
        }

# Global instance
_gpu_manager = None

def get_gpu_manager() -> MultiAgentGPUManager:
    """Get or create global GPU manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = MultiAgentGPUManager()
    return _gpu_manager

# Convenience functions for agents
def request_agent_gpu(agent_id: str, agent_type: str) -> Dict[str, Any]:
    """Request GPU allocation for an agent (string-based agent type)"""
    try:
        agent_type_enum = AgentType(agent_type)
        manager = get_gpu_manager()
        return manager.request_gpu_allocation(agent_id, agent_type_enum)
    except ValueError:
        return {"status": "error", "message": f"Unknown agent type: {agent_type}"}

def release_agent_gpu(agent_id: str) -> bool:
    """Release GPU allocation for an agent"""
    manager = get_gpu_manager()
    return manager.release_gpu_allocation(agent_id)

def get_system_gpu_status() -> Dict[str, Any]:
    """Get system-wide GPU status"""
    manager = get_gpu_manager()
    return manager.get_gpu_status()

def get_deployment_recommendations() -> Dict[str, Any]:
    """Get recommendations for optimal multi-agent deployment"""
    manager = get_gpu_manager()
    return manager.get_allocation_recommendations()
