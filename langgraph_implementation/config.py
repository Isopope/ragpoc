"""
Configuration and utility functions for the LangGraph implementation.
"""

import os
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Configuration management for the LangGraph system."""
    
    # Load environment variables
    load_dotenv()
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Models Configuration
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    # For litellm, model name should preferably include the provider (e.g., openai/gpt-4-turbo)
    # But usually litellm routes standard models fine.
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    
    # Weaviate Configuration
    WEAVIATE_URL = os.getenv("WCD_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WCD_API_KEY", "")
    
    # LangGraph Configuration
    LANGGRAPH_MODE = os.getenv("LANGGRAPH_MODE", "multibranch")
    MAX_TREE_DEPTH = int(os.getenv("MAX_TREE_DEPTH", "5"))
    RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "100"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Feature Flags
    USE_WEAVIATE = os.getenv("USE_WEAVIATE", "true").lower() == "true"
    USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
    ENABLE_ASYNC = os.getenv("ENABLE_ASYNC", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration settings."""
        if cls.USE_LLM and not (cls.OPENAI_API_KEY or cls.MISTRAL_API_KEY or cls.ANTHROPIC_API_KEY or cls.DEFAULT_LLM_PROVIDER == 'ollama'):
            print("WARNING: USE_LLM=true but no LLM API KEY is set")
            return False
        
        if cls.USE_WEAVIATE and not cls.WEAVIATE_URL:
            print("WARNING: USE_WEAVIATE=true but WCD_URL not set")
            return False
        
        return True
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert configuration to dictionary."""
        return {
            "openai_model": cls.OPENAI_MODEL,
            "weaviate_url": cls.WEAVIATE_URL,
            "max_tree_depth": cls.MAX_TREE_DEPTH,
            "mode": cls.LANGGRAPH_MODE,
            "debug": cls.DEBUG,
        }


def setup_weaviate_client():
    """Initialize Weaviate client if configured."""
    if not Config.USE_WEAVIATE:
        return None
    
    try:
        import weaviate
        
        client = weaviate.WeaviateClient(
            url=Config.WEAVIATE_URL,
            api_key=Config.WEAVIATE_API_KEY,
            additional_headers={},
        )
        
        return client
        
    except ImportError:
        print("WARNING: weaviate package not installed")
        return None
    except Exception as e:
        print(f"ERROR: Failed to connect to Weaviate: {e}")
        return None


def setup_llm_client():
    """Initialize LLM client if configured."""
    if not Config.USE_LLM:
        return None
    
    try:
        import sys
        import os
        # Add parent dir to path to import llm package
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from llm import get_langchain_llm
        
        client = get_langchain_llm(
            provider_model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY, # Pass if needed, or rely on env
            temperature=0.7,
            max_tokens=2000,
        )
        
        return client
        
    except ImportError:
        print("WARNING: llm_factory or litellm not available")
        return None
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM client: {e}")
        return None


class DebugHelper:
    """Utilities for debugging and development."""
    
    @staticmethod
    def print_state_snapshot(state: dict) -> None:
        """Print a formatted snapshot of the current state."""
        print("\n" + "=" * 60)
        print("STATE SNAPSHOT")
        print("=" * 60)
        
        print(f"\nUser Prompt: {state.get('user_prompt', 'N/A')}")
        print(f"Current Branch: {state.get('current_branch', 'N/A')}")
        print(f"Tree Depth: {state.get('tree_depth', 0)}/{state.get('max_tree_depth', 5)}")
        
        print(f"\nDecision History ({len(state.get('decision_history', []))} decisions):")
        for i, decision in enumerate(state.get('decision_history', [])[-5:]):
            print(f"  {i+1}. {decision}")
        
        environment = state.get('environment', {})
        print(f"\nEnvironment ({len(environment)} tool results):")
        for tool_name, collections in environment.items():
            for collection_name, results in collections.items():
                total_objs = sum(len(r.objects) for r in results)
                print(f"  - {tool_name} ({collection_name}): {total_objs} objects")
        
        errors = state.get('errors', [])
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for error in errors[-3:]:
                print(f"  - [{error.get('tool_name')}] {error.get('message')}")
        
        print("\n" + "=" * 60)
    
    @staticmethod
    def print_execution_summary(result: dict) -> None:
        """Print a summary of the execution result."""
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        
        print(f"\nQuery: {result.get('user_prompt', 'N/A')}")
        print(f"Response: {result.get('response', 'N/A')[:200]}...")
        print(f"\nDecision Path: {' → '.join(result.get('decision_history', []))}")
        print(f"Tree Depth: {result.get('tree_depth', 0)}")
        print(f"Objects Retrieved: {len(result.get('retrieved_objects', []))}")
        print(f"Errors: {len(result.get('errors', []))}")
        
        metadata = result.get('metadata', {})
        print(f"\nMetadata:")
        print(f"  Mode: {metadata.get('mode', 'N/A')}")
        print(f"  Executed: {metadata.get('executed_at', 'N/A')}")
        
        print("\n" + "=" * 60)
    
    @staticmethod
    def print_tree_structure(tree_builder) -> None:
        """Print the tree structure."""
        print("\n" + "=" * 60)
        print("TREE STRUCTURE")
        print("=" * 60)
        
        structure = tree_builder.get_branch_structure()
        
        for branch_id, branch_info in structure.items():
            indent = "  " if not branch_info["is_root"] else ""
            marker = "🌳" if branch_info["is_root"] else "📦"
            
            print(f"\n{indent}{marker} {branch_id}")
            print(f"{indent}  Instruction: {branch_info['instruction'][:50]}...")
            print(f"{indent}  Options: {', '.join(branch_info['options'])}")
            
            if branch_info['parent']:
                print(f"{indent}  Parent: {branch_info['parent']}")
        
        print("\n" + "=" * 60)


class MetricsCollector:
    """Collect and report metrics about executions."""
    
    def __init__(self):
        self.executions = []
        self.total_decisions = 0
        self.total_tool_calls = 0
        self.total_errors = 0
        self.avg_tree_depth = 0
    
    def record_execution(self, result: dict) -> None:
        """Record metrics from an execution result."""
        self.executions.append(result)
        self.total_decisions += len(result.get('decision_history', []))
        self.total_errors += len(result.get('errors', []))
        
        # Count tool calls
        for obj in result.get('retrieved_objects', []):
            self.total_tool_calls += 1
        
        # Update average depth
        if self.executions:
            all_depths = [e.get('tree_depth', 0) for e in self.executions]
            self.avg_tree_depth = sum(all_depths) / len(all_depths)
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            "total_executions": len(self.executions),
            "total_decisions": self.total_decisions,
            "total_tool_calls": self.total_tool_calls,
            "total_errors": self.total_errors,
            "avg_tree_depth": self.avg_tree_depth,
            "error_rate": self.total_errors / max(1, self.total_tool_calls),
        }
    
    def print_summary(self) -> None:
        """Print a formatted metrics summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)
        
        print(f"\nExecutions: {summary['total_executions']}")
        print(f"Total Decisions: {summary['total_decisions']}")
        print(f"Total Tool Calls: {summary['total_tool_calls']}")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Avg Tree Depth: {summary['avg_tree_depth']:.2f}")
        print(f"Error Rate: {summary['error_rate']:.2%}")
        
        print("\n" + "=" * 60)


# Global instances
config = Config()
_weaviate_client = None
_llm_client = None


def get_weaviate_client():
    """Get or create the global Weaviate client."""
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client = setup_weaviate_client()
    return _weaviate_client


def get_llm_client():
    """Get or create the global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = setup_llm_client()
    return _llm_client
