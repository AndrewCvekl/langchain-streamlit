"""
Entry point for LangGraph Studio.
Exports the compiled agent graph.
"""

from graph_with_verification import create_agent_with_verification

# Create and compile the agent without custom checkpointer
# LangGraph API handles persistence automatically
graph = create_agent_with_verification()

# Export for LangGraph Studio
__all__ = ['graph']
