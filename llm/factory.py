"""
Centralized LLM Factory using LiteLLM.
Provides unified access to OpenAI, Mistral, Anthropic, Ollama, etc.
"""
from typing import Any, Optional
import os
import threading

from loguru import logger

def get_llm_completion(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """Wrapper to call litellm.completion with timeout."""
    try:
        import litellm
    except ImportError:
        logger.error("litellm is not installed.")
        raise ImportError("litellm package is required for get_llm_completion.")
        
    # Optional performance optimization/suppression of logs
    litellm.suppress_debug_info = True
    # Drop unsupported parameters if any provider doesn't support them
    litellm.drop_params = True
    
    result = {"response": None, "error": None}
    
    def _run():
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=api_base,
                **kwargs
            )
            result["response"] = resp
        except Exception as e:
            logger.error(f"LiteLLM completion error: {e}")
            result["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(f"LLM request timed out after {timeout} seconds.")
    if result["error"]:
        raise result["error"]
        
    return result["response"]


def get_langchain_llm(
    provider_model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs: Any
):
    """Factory to get a LangChain ChatLiteLLM instance."""
    try:
        from langchain_community.chat_models import ChatLiteLLM
    except ImportError:
        logger.warning("langchain_community not installed, ChatLiteLLM is unavailable.")
        return None
    
    return ChatLiteLLM(
        model=provider_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=api_base,
        **kwargs
    )
