"""
Simple LangSmith tracing setup for the music store support bot.

LangChain/LangGraph automatically enable tracing when these environment variables are set:
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_API_KEY (or LANGSMITH_API_KEY)
- LANGCHAIN_PROJECT (optional, for organizing traces)
"""

import os
import logging

logger = logging.getLogger(__name__)


def setup_langsmith_tracing():
    """
    Configure LangSmith tracing via environment variables.
    This is called automatically - no manual setup needed if env vars are set.
    """
    # Check if tracing is enabled
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    # Get API key (supports both LANGCHAIN_API_KEY and LANGSMITH_API_KEY)
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    
    # Get project name (optional)
    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT", "music-store-support-bot")
    
    if tracing_enabled and api_key:
        # Set environment variables for LangChain to pick up
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        
        logger.info(f"✅ LangSmith tracing enabled for project: {project}")
        logger.info("📊 Traces will appear in LangSmith dashboard")
        return True
    elif tracing_enabled and not api_key:
        logger.warning("⚠️ LANGCHAIN_TRACING_V2=true but no API key found. Set LANGCHAIN_API_KEY or LANGSMITH_API_KEY")
        return False
    else:
        logger.info("ℹ️ LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")
        return False
