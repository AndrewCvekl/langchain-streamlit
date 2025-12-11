"""
Entry point for LangGraph Studio.
Exports the compiled agent graph.
"""

from graph_with_verification import create_agent_with_memory

# Create and compile the agent with memory/checkpointing
graph = create_agent_with_memory()

# Export for LangGraph Studio
__all__ = ['graph']
